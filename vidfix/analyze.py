from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Callable

from . import ffmpeg_util, job as jobmod
from .paths import job_dir
from .plan import plan_segments

StopCheck = Callable[[], bool]


class Stopped(Exception):
    pass


def run_analyze(job_id: str, stop_check: StopCheck | None = None) -> None:
    stop_check = stop_check or (lambda: jobmod.should_stop(job_id))
    job = jobmod.load_job(job_id)
    src = Path(job["source_path"])
    if not src.is_file():
        raise FileNotFoundError(jobmod.source_missing_message(src))

    jobmod.update_job(
        job_id,
        phase="analyze",
        analyze_status="running",
        error=None,
        progress=jobmod.empty_progress()
        | {
            "message": "開始分析",
            "started_at": time.time(),
        },
    )
    jobmod.append_log(job_id, "分析開始")
    t0 = time.time()

    try:
        _detect_cuts(job_id, job, src, stop_check)
        if stop_check():
            raise Stopped()
        _plan_and_write_segments(job_id, job)
        if stop_check():
            raise Stopped()
        _make_thumbs(job_id, src, stop_check, t0)
        jobmod.update_job(
            job_id,
            phase="review",
            analyze_status="done",
            progress=jobmod.empty_progress() | {"message": "分析完成"},
            error=None,
        )
        jobmod.append_log(job_id, "分析完成")
    except Stopped:
        jobmod.update_job(
            job_id,
            analyze_status="stopped",
            progress=jobmod.empty_progress() | {"message": "分析已停止，可繼續"},
        )
        jobmod.append_log(job_id, "分析停止")
    except Exception as e:
        jobmod.update_job(
            job_id,
            analyze_status="failed",
            error=str(e),
            progress=jobmod.empty_progress() | {"message": f"分析失敗：{e}"},
        )
        jobmod.append_log(job_id, f"分析失敗：{e}")
        raise


def run_thumbs_only(job_id: str, stop_check: StopCheck | None = None) -> None:
    """Fill missing thumbnails without re-running scene detection."""
    stop_check = stop_check or (lambda: jobmod.should_stop(job_id))
    job = jobmod.load_job(job_id)
    src = Path(job["source_path"])
    if not src.is_file():
        raise FileNotFoundError(jobmod.source_missing_message(src))
    jobmod.update_job(
        job_id,
        phase="analyze",
        analyze_status="running",
        error=None,
        progress=jobmod.empty_progress()
        | {"message": "補抽縮圖", "started_at": time.time()},
    )
    jobmod.append_log(job_id, "補抽縮圖開始")
    t0 = time.time()
    try:
        _make_thumbs(job_id, src, stop_check, t0)
        if stop_check():
            raise Stopped()
        missing = [
            s
            for s in jobmod.load_segments(job_id)
            if not jobmod.segment_thumb_path(job_id, s["index"]).is_file()
        ]
        jobmod.update_job(
            job_id,
            phase="review",
            analyze_status="done",
            progress=jobmod.empty_progress()
            | {"message": "縮圖完成" if not missing else f"縮圖完成（缺 {len(missing)}）"},
            error=None,
        )
        jobmod.append_log(job_id, "補抽縮圖結束")
    except Stopped:
        jobmod.update_job(
            job_id,
            analyze_status="stopped",
            phase="review",
            progress=jobmod.empty_progress() | {"message": "縮圖已停止，可繼續"},
        )
        jobmod.append_log(job_id, "補抽縮圖停止")
    except Exception as e:
        jobmod.update_job(
            job_id,
            analyze_status="failed",
            error=str(e),
            progress=jobmod.empty_progress() | {"message": f"縮圖失敗：{e}"},
        )
        jobmod.append_log(job_id, f"補抽縮圖失敗：{e}")
        raise


def _window_progress_path(job_id: str) -> Path:
    return job_dir(job_id) / "analyze_windows.jsonl"


def _cuts_path(job_id: str) -> Path:
    return job_dir(job_id) / "cuts.json"


def _done_windows(job_id: str) -> set[int]:
    p = _window_progress_path(job_id)
    done: set[int] = set()
    if not p.is_file():
        return done
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        done.add(int(rec["window"]))
    return done


def _detect_cuts(job_id: str, job: dict[str, Any], src: Path, stop_check: StopCheck) -> None:
    duration = float(job["duration"])
    window = float(job["params"].get("analyze_window_sec") or 240.0)
    threshold = float(job["params"].get("scene_threshold") or 27.0)
    n_windows = max(1, int((duration + window - 1e-6) // window))
    done = _done_windows(job_id)
    started = time.time()

    if len(done) >= n_windows:
        jobmod.append_log(job_id, "場景切點已存在，跳過偵測")
        return

    try:
        from scenedetect import ContentDetector, SceneManager, open_video
    except Exception as e:
        jobmod.append_log(job_id, f"PySceneDetect 不可用（{e}），改依最長 25 秒切段")
        jobmod.atomic_write_json(_cuts_path(job_id), {"cuts": [], "fallback": "fixed-length"})
        return

    for wi in range(n_windows):
        if stop_check():
            raise Stopped()
        if wi in done:
            continue
        w0 = wi * window
        w1 = min(duration, (wi + 1) * window)
        jobmod.set_progress(
            job_id,
            message=f"場景切分 視窗 {wi + 1}/{n_windows}（{jobmod.fmt_timecode(w0)}–{jobmod.fmt_timecode(w1)}）",
            current=wi,
            total=n_windows,
            eta_sec=_eta(started, len(done), n_windows),
        )
        try:
            cuts = _detect_window(src, w0, w1, threshold, open_video, SceneManager, ContentDetector)
        except Exception as e:
            jobmod.append_log(job_id, f"視窗 {wi + 1} 場景偵測失敗，改 ffmpeg：{e}")
            try:
                cuts = _detect_window_ffmpeg(src, w0, w1)
            except Exception as e2:
                jobmod.append_log(job_id, f"視窗 {wi + 1} ffmpeg 也失敗，此窗無切點：{e2}")
                cuts = []
        rec = {"window": wi, "t0": round(w0, 3), "t1": round(w1, 3), "cuts": cuts}
        with _window_progress_path(job_id).open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        done.add(wi)
        jobmod.append_log(job_id, f"視窗 {wi + 1}/{n_windows} 完成，切點 {len(cuts)}")

    all_cuts = _collect_cuts(job_id)
    jobmod.atomic_write_json(_cuts_path(job_id), {"cuts": all_cuts})


def _detect_window(
    src: Path,
    w0: float,
    w1: float,
    threshold: float,
    open_video,
    SceneManager,
    ContentDetector,
) -> list[float]:
    video = open_video(str(src))
    try:
        if w0 > 0:
            video.seek(w0)
        sm = SceneManager()
        sm.auto_downscale = True
        sm.add_detector(ContentDetector(threshold=threshold, min_scene_len="0.4s"))
        duration = max(0.1, w1 - w0)
        sm.detect_scenes(video, duration=duration, show_progress=False)
        scenes = sm.get_scene_list()
        cuts: list[float] = []
        for start, _end in scenes:
            t = round(float(start.get_seconds()), 3)
            # (w0, w1] so a cut exactly on a window boundary is not dropped
            if t > w0 + 1e-4 and t <= w1 + 1e-4:
                cuts.append(t)
        return cuts
    finally:
        close = getattr(video, "close", None)
        if callable(close):
            close()


_PTS_RE = re.compile(r"pts_time:\s*([0-9.]+)")


def _detect_window_ffmpeg(src: Path, w0: float, w1: float) -> list[float]:
    ffmpeg = ffmpeg_util.which_ffmpeg()
    dur = max(0.1, w1 - w0)
    proc = ffmpeg_util.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "info",
            "-ss",
            f"{w0:.3f}",
            "-t",
            f"{dur:.3f}",
            "-i",
            str(src),
            "-an",
            "-vf",
            "select='gt(scene,0.30)',showinfo",
            "-vsync",
            "vfr",
            "-f",
            "null",
            "-",
        ],
        check=False,
    )
    text = (proc.stderr or "") + (proc.stdout or "")
    cuts: list[float] = []
    span = w1 - w0
    for m in _PTS_RE.finditer(text):
        raw = float(m.group(1))
        t = raw + w0 if raw <= span + 0.5 else raw
        t = round(t, 3)
        if t > w0 + 1e-4 and t <= w1 + 1e-4:
            cuts.append(t)
    return sorted(set(cuts))


def _collect_cuts(job_id: str) -> list[float]:
    p = _window_progress_path(job_id)
    cuts: list[float] = []
    if not p.is_file():
        return cuts
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        cuts.extend(float(x) for x in rec.get("cuts") or [])
    return sorted(set(round(c, 3) for c in cuts))


def _plan_and_write_segments(job_id: str, job: dict[str, Any]) -> None:
    existing = jobmod.load_segments(job_id)
    if existing:
        jobmod.update_job(job_id, total_segments=len(existing))
        return
    cuts_file = _cuts_path(job_id)
    cuts: list[float] = []
    if cuts_file.is_file():
        cuts = json.loads(cuts_file.read_text(encoding="utf-8")).get("cuts") or []
    else:
        cuts = _collect_cuts(job_id)
    duration = float(job["duration"])
    params = job["params"]
    spans = plan_segments(
        cuts,
        duration,
        max_len=float(params.get("max_segment_sec") or 25.0),
        min_len=float(params.get("min_segment_sec") or 1.0),
    )
    segs = []
    for i, (a, b) in enumerate(spans):
        segs.append(
            {
                "index": i,
                "t0": a,
                "t1": b,
                "tag": "skip",
                "keep": True,
                "status": "pending",
                "thumb": f"thumbs/{i:04d}.jpg",
                "out": f"out/{i:04d}.mkv",
                "out_kind": None,
                "error": None,
            }
        )
    jobmod.save_segments(job_id, segs)
    jobmod.update_job(job_id, total_segments=len(segs))
    jobmod.append_log(job_id, f"切成 {len(segs)} 段（預設全部跳過）")


def _make_thumbs(job_id: str, src: Path, stop_check: StopCheck, started: float) -> None:
    job = jobmod.load_job(job_id)
    segs = jobmod.load_segments(job_id)
    duration = float(job["duration"])
    missing = []
    for s in segs:
        dest = jobmod.segment_thumb_path(job_id, s["index"])
        if not dest.is_file() or dest.stat().st_size < 32:
            missing.append(s)
    total = len(missing)
    if total == 0:
        return
    done = 0
    for s in missing:
        if stop_check():
            raise Stopped()
        mid = (float(s["t0"]) + float(s["t1"])) / 2.0
        dest = jobmod.segment_thumb_path(job_id, s["index"])
        try:
            ffmpeg_util.extract_thumbnail(src, mid, dest, duration)
        except Exception as e:
            jobmod.append_log(job_id, f"縮圖 {s['index']:04d} 失敗：{e}")
        done += 1
        if done == 1 or done == total or done % 8 == 0:
            jobmod.set_progress(
                job_id,
                message=f"抽縮圖 {done}/{total}",
                current=done,
                total=total,
                eta_sec=_eta(started, done, total) if done else None,
            )


def _eta(started: float, done: int, total: int) -> float | None:
    if done <= 0 or total <= done:
        return None
    elapsed = time.time() - started
    rate = done / max(elapsed, 1e-3)
    return round((total - done) / rate, 1)
