from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Callable

from . import ffmpeg_util, job as jobmod
from .paths import job_dir

StopCheck = Callable[[], bool]


class Stopped(Exception):
    pass


NO_KEPT_MSG = "沒有保留任何段，無法輸出。請至少保留一段（D 切換保留／捨去）。"


def run_assemble(job_id: str, stop_check: StopCheck | None = None) -> None:
    stop_check = stop_check or (lambda: jobmod.should_stop(job_id))
    job = jobmod.load_job(job_id)
    src = Path(job["source_path"])
    if not src.is_file():
        raise FileNotFoundError(jobmod.source_missing_message(src))

    all_segs = jobmod.load_segments(job_id)
    if not all_segs:
        raise RuntimeError("尚未分析，沒有分段。")
    segs = [s for s in all_segs if jobmod.is_kept(s)]
    if not segs:
        raise RuntimeError(NO_KEPT_MSG)
    all_kept = len(segs) == len(all_segs)

    missing_restore = [s for s in segs if jobmod.needs_restore(s)]
    if missing_restore:
        ids = ", ".join(f"{int(s['index']):04d}" for s in missing_restore[:12])
        raise RuntimeError(f"還有未完成的修復段：{ids}")

    jobmod.update_job(
        job_id,
        phase="assemble",
        assemble_status="running",
        error=None,
        progress=jobmod.empty_progress()
        | {
            "message": "準備輸出",
            "started_at": time.time(),
            "total": len(segs),
        },
    )
    jobmod.append_log(
        job_id,
        "輸出開始" if all_kept else f"輸出開始（保留 {len(segs)}/{len(all_segs)} 段）",
    )
    started = time.time()

    try:
        clips: list[tuple[Path, float]] = []
        fps = float(job["fps"])
        for i, seg in enumerate(segs):
            if stop_check():
                raise Stopped()
            idx = int(seg["index"])
            dest = jobmod.segment_out_path(job_id, idx)
            kind = "restore" if seg.get("tag") == "restore" else "skip"
            nframes = ffmpeg_util.clip_frame_count(float(seg["t0"]), float(seg["t1"]), fps)
            dur = nframes / fps
            jobmod.set_progress(
                job_id,
                message=f"編碼分段 {idx:04d}（{kind}） {i + 1}/{len(segs)}",
                current=i,
                total=len(segs),
                segment_index=idx,
                eta_sec=_eta(started, i, len(segs)),
            )
            if kind == "restore":
                if not dest.is_file():
                    raise RuntimeError(f"修復檔不存在：{dest.name}")
                # codeformer or deblock output is used as-is
            else:
                need = True
                if dest.is_file() and seg.get("out_kind") == "skip_v2":
                    need = False
                if need:
                    if dest.exists():
                        dest.unlink()
                    ffmpeg_util.encode_skip_clip(
                        src,
                        float(seg["t0"]),
                        float(seg["t1"]),
                        dest,
                        width=int(job["width"]),
                        height=int(job["height"]),
                        fps=fps,
                        crf=int(job["params"].get("crf_skip") or 18),
                        preset=str(job["params"].get("preset_skip") or "veryfast"),
                    )
                    jobmod.update_segment(job_id, idx, status="done", out_kind="skip_v2", error=None)
            clips.append((dest, dur))

        if stop_check():
            raise Stopped()

        final_path = job_dir(job_id) / "final.mkv"
        list_path = job_dir(job_id) / "concat.txt"
        if all_kept:
            jobmod.set_progress(job_id, message="接回完整影片並 mux 原音訊", current=len(segs), total=len(segs))
            ffmpeg_util.concat_and_mux(
                clips,
                src,
                final_path,
                bool(job.get("has_audio")),
                list_path,
            )
        else:
            _assemble_kept(job_id, job, src, segs, clips, final_path, list_path, stop_check)
        jobmod.update_job(
            job_id,
            phase="done",
            assemble_status="done",
            final_path=str(final_path),
            progress=jobmod.empty_progress() | {"message": "成品已輸出"},
            error=None,
        )
        jobmod.append_log(job_id, f"成品：{final_path}")
    except Stopped:
        jobmod.update_job(
            job_id,
            assemble_status="stopped",
            progress=jobmod.empty_progress() | {"message": "輸出已停止，可繼續"},
        )
        jobmod.append_log(job_id, "輸出停止")
    except Exception as e:
        jobmod.update_job(
            job_id,
            assemble_status="failed",
            error=str(e),
            progress=jobmod.empty_progress() | {"message": f"輸出失敗：{e}"},
        )
        jobmod.append_log(job_id, f"輸出失敗：{e}")
        raise


def audio_runs(segs: list[dict], fps: float) -> list[tuple[float, float]]:
    """(start, duration) in seconds for each run of adjacent kept segments.

    Boundaries are the same frame grid as clip_frame_count, so each run's audio
    length equals the summed video length of its clips.
    """
    runs: list[list[int]] = []
    for s in segs:
        i0 = int(round(float(s["t0"]) * fps))
        i1 = i0 + ffmpeg_util.clip_frame_count(float(s["t0"]), float(s["t1"]), fps)
        if runs and runs[-1][1] == i0:
            runs[-1][1] = i1
        else:
            runs.append([i0, i1])
    return [(a / fps, (b - a) / fps) for a, b in runs]


def _assemble_kept(job_id, job, src: Path, segs, clips, final_path: Path, list_path: Path, stop_check) -> None:
    """Dropped segments present: cut source audio to the kept runs, then concat + AAC."""
    fps = float(job["fps"])
    audio_dir = job_dir(job_id) / "audio_cut"
    parts: list[Path] = []
    try:
        if job.get("has_audio"):
            # Segment times count from the first video frame; audio is cut on the source pts clock.
            info = ffmpeg_util.probe(src)
            v0 = float(info.get("video_start") or 0.0)
            rate = int(info.get("audio_rate") or 48000)
            runs = audio_runs(segs, fps)
            if audio_dir.exists():
                shutil.rmtree(audio_dir, ignore_errors=True)
            for ri, (start, dur) in enumerate(runs):
                if stop_check():
                    raise Stopped()
                jobmod.set_progress(
                    job_id,
                    message=f"切音訊 {ri + 1}/{len(runs)}",
                    current=len(segs),
                    total=len(segs),
                )
                dest = audio_dir / f"{ri:04d}.wav"
                ffmpeg_util.extract_audio_wav(src, v0 + start, dur, dest, rate)
                parts.append(dest)
        if stop_check():
            raise Stopped()
        jobmod.set_progress(
            job_id, message="接回保留段並編碼音訊（AAC）", current=len(segs), total=len(segs)
        )
        ffmpeg_util.concat_and_mux_cut(
            clips, parts, final_path, list_path, job_dir(job_id) / "concat_audio.txt"
        )
    finally:
        if audio_dir.exists():
            shutil.rmtree(audio_dir, ignore_errors=True)


def _eta(started: float, done: int, total: int) -> float | None:
    if done <= 0 or total <= done:
        return None
    elapsed = time.time() - started
    return round((total - done) * (elapsed / done), 1)
