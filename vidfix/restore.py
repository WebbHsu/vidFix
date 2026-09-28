from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

import numpy as np

from . import ffmpeg_util, job as jobmod
from .stabilize import smooth_affine_track

StopCheck = Callable[[], bool]


class Stopped(Exception):
    pass


def run_restore(job_id: str, stop_check: StopCheck | None = None) -> None:
    stop_check = stop_check or (lambda: jobmod.should_stop(job_id))
    job = jobmod.load_job(job_id)
    src = Path(job["source_path"])
    if not src.is_file():
        raise FileNotFoundError(f"找不到原片：{src}")

    segs = jobmod.load_segments(job_id)
    queue = [
        s
        for s in segs
        if s.get("tag") == "restore" and s.get("status") != "done"
    ]
    if not queue:
        jobmod.update_job(
            job_id,
            restore_status="done",
            phase="review",
            progress=jobmod.empty_progress() | {"message": "沒有待修復的段"},
        )
        return

    params = job["params"]
    need_cf = any("codeformer" in jobmod.methods_for_segment(s, params) for s in queue)
    jobmod.update_job(
        job_id,
        phase="restore",
        restore_status="running",
        error=None,
        progress=jobmod.empty_progress()
        | {
            "message": f"開始修復（{len(queue)} 段）",
            "started_at": time.time(),
            "current": 0,
            "total": len(queue),
        },
    )
    jobmod.append_log(job_id, f"修復開始，佇列 {len(queue)} 段")

    restorer = None
    try:
        if need_cf:
            try:
                from .models.restorer import FaceRestorer
            except ModuleNotFoundError as e:
                if "torch" in str(e):
                    raise RuntimeError(
                        "尚未安裝 PyTorch。請關掉目前的 vidFix，改用 run.bat 啟動（會使用專案裡的 .venv）。"
                    ) from e
                raise
            restorer = FaceRestorer(
                fidelity=float(params.get("fidelity") or 0.40),
                visibility=float(params.get("visibility") or 0.60),
                tiny_face_px=int(params.get("tiny_face_px") or 75),
            )
        started = time.time()
        for qi, seg in enumerate(queue):
            if stop_check():
                raise Stopped()
            idx = int(seg["index"])
            methods = jobmod.methods_for_segment(seg, params)
            names = {"codeformer": "修臉", "deblock": "去塊", "deblur": "去糊", "denoise": "降噪"}
            label = "+".join(names[m] for m in methods if m in names) or "修臉"
            jobmod.set_progress(
                job_id,
                message=f"{label} {idx:04d}（{qi + 1}/{len(queue)}）",
                current=qi,
                total=len(queue),
                segment_index=idx,
                eta_sec=_eta(started, qi, len(queue)),
            )
            try:
                if "codeformer" in methods:
                    extra = ffmpeg_util.compression_core(
                        str(params.get("deblock_strength") or "medium"),
                        [m for m in methods if m in ("deblock", "deblur", "denoise")],
                        denoise_strength=str(params.get("denoise_strength") or "medium"),
                    )
                    _restore_one(job_id, job, src, seg, restorer, stop_check, extra_vf=extra)
                    jobmod.update_segment(
                        job_id, idx, status="done", out_kind="restore", methods=methods, error=None
                    )
                else:
                    _restore_deblock_one(job_id, job, src, seg, stop_check, methods=methods)
                    jobmod.update_segment(
                        job_id,
                        idx,
                        status="done",
                        out_kind="+".join(methods) or "deblock",
                        methods=methods,
                        error=None,
                    )
                jobmod.append_log(job_id, f"段 {idx:04d} {label}完成")
            except Stopped:
                raise
            except Exception as e:
                jobmod.update_segment(job_id, idx, status="failed", error=str(e))
                jobmod.append_log(job_id, f"段 {idx:04d} 失敗：{e}")

        segs = jobmod.load_segments(job_id)
        pending = [s for s in segs if s.get("tag") == "restore" and s.get("status") != "done"]
        failed = [s for s in segs if s.get("tag") == "restore" and s.get("status") == "failed"]
        if pending:
            jobmod.update_job(
                job_id,
                restore_status="stopped" if stop_check() else "pending",
                phase="review",
                progress=jobmod.empty_progress() | {"message": "修復未全部完成"},
            )
        else:
            msg = "修復完成" if not failed else f"修復結束（失敗 {len(failed)} 段）"
            jobmod.update_job(
                job_id,
                restore_status="done",
                phase="review",
                progress=jobmod.empty_progress() | {"message": msg},
            )
        jobmod.append_log(job_id, "修復階段結束")
    except Stopped:
        jobmod.update_job(
            job_id,
            restore_status="stopped",
            phase="review",
            progress=jobmod.empty_progress() | {"message": "修復已停止，可繼續"},
        )
        jobmod.append_log(job_id, "修復停止")
    except Exception as e:
        jobmod.update_job(
            job_id,
            restore_status="failed",
            error=str(e),
            progress=jobmod.empty_progress() | {"message": f"修復失敗：{e}"},
        )
        jobmod.append_log(job_id, f"修復失敗：{e}")
        raise
    finally:
        if restorer is not None:
            restorer.close()


def _restore_deblock_one(job_id, job, src: Path, seg, stop_check: StopCheck, methods=None) -> None:
    if stop_check():
        raise Stopped()
    dest = jobmod.segment_out_path(job_id, int(seg["index"]))
    params = job["params"]
    kinds = [m for m in (methods or jobmod.methods_for_segment(seg, params)) if m in ("deblock", "deblur", "denoise")]
    ffmpeg_util.encode_deblock_clip(
        src,
        float(seg["t0"]),
        float(seg["t1"]),
        dest,
        width=int(job["width"]),
        height=int(job["height"]),
        fps=float(job["fps"]),
        crf=int(params.get("crf_restore") or 16),
        preset=str(params.get("preset_restore") or "fast"),
        strength=str(params.get("deblock_strength") or "medium"),
        kinds=kinds or ["deblock"],
        denoise_strength=str(params.get("denoise_strength") or "medium"),
    )


def _restore_one(job_id, job, src: Path, seg, restorer, stop_check: StopCheck, extra_vf: str | None = None) -> None:
    reset = getattr(restorer, "reset", None)
    if callable(reset):
        reset()
    idx = int(seg["index"])
    dest = jobmod.segment_out_path(job_id, idx)
    tmp = dest.with_name(dest.name + ".partial")
    if tmp.exists():
        tmp.unlink()
    if dest.exists():
        dest.unlink()

    width = int(job["width"])
    height = int(job["height"])
    fps = float(job["fps"])
    t0 = float(seg["t0"])
    t1 = float(seg["t1"])
    params = job["params"]
    deblock = bool(params.get("deblock", True))
    crf = int(params.get("crf_restore") or 16)
    preset = str(params.get("preset_restore") or "fast")

    expected = ffmpeg_util.clip_frame_count(t0, t1, fps)

    jobmod.set_progress(
        job_id,
        message=f"修復 {idx:04d}  追蹤臉部",
        current=0,
        total=expected,
        segment_index=idx,
    )
    kps_list: list = []
    try:
        for i, frame in enumerate(
            _iter_frames(src, t0, t1, width, height, fps, deblock, expected, stop_check, extra_vf=extra_vf)
        ):
            kps_list.append(restorer.detect_kps(frame))
            if i == 0 or (i + 1) % 20 == 0 or i + 1 == expected:
                jobmod.set_progress(
                    job_id,
                    message=f"修復 {idx:04d}  追蹤 {i + 1}/{expected}",
                    current=i + 1,
                    total=expected * 2,
                    segment_index=idx,
                )
        while len(kps_list) < expected:
            kps_list.append(None)
        poses = smooth_affine_track(kps_list, fps)

        enc = ffmpeg_util.encode_process(tmp, width, height, fps, crf, preset)
        n = 0
        last = None
        last_progress = time.time()
        try:
            assert enc.stdin is not None
            for i, frame in enumerate(
                _iter_frames(src, t0, t1, width, height, fps, deblock, expected, stop_check, extra_vf=extra_vf)
            ):
                M = poses[i] if i < len(poses) else None
                out = restorer.restore_frame(frame, M=M)
                last = out.tobytes()
                enc.stdin.write(last)
                n += 1
                now = time.time()
                if now - last_progress >= 1.0:
                    last_progress = now
                    jobmod.set_progress(
                        job_id,
                        message=f"修復 {idx:04d}  幀 {n}/{expected}",
                        current=expected + n,
                        total=expected * 2,
                        segment_index=idx,
                    )
            while n < expected and last is not None:
                enc.stdin.write(last)
                n += 1
            enc.stdin.close()
            enc_rc = enc.wait(timeout=120)
            if enc_rc != 0:
                err = (enc.stderr.read() if enc.stderr else b"").decode("utf-8", "replace")[-1500:]
                raise RuntimeError(f"編碼失敗：{err}")
        except Exception:
            _kill(enc)
            raise
        finally:
            if enc.stderr:
                enc.stderr.close()

        if n == 0:
            raise RuntimeError("沒有解出任何影格")
        if not tmp.exists() or tmp.stat().st_size < 64:
            raise RuntimeError("修復輸出為空")
        tmp.replace(dest)
    except Stopped:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        raise
    except Exception:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        raise


def _iter_frames(src, t0, t1, width, height, fps, deblock, expected, stop_check, extra_vf=None):
    frame_bytes = width * height * 3
    dec = ffmpeg_util.decode_process(
        src, t0, t1, width, height, fps, deblock, nframes=expected, extra_vf=extra_vf
    )
    n = 0
    try:
        assert dec.stdout is not None
        while n < expected:
            if stop_check():
                raise Stopped()
            buf = dec.stdout.read(frame_bytes)
            if not buf or len(buf) < frame_bytes:
                break
            yield np.frombuffer(buf, dtype=np.uint8).reshape((height, width, 3)).copy()
            n += 1
    finally:
        _kill(dec)
        if dec.stdout:
            dec.stdout.close()
        if dec.stderr:
            dec.stderr.close()


def _kill(proc) -> None:
    if proc is None:
        return
    try:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)
    except Exception:
        pass


def _eta(started: float, done: int, total: int) -> float | None:
    if done <= 0 or total <= done:
        return None
    elapsed = time.time() - started
    return round((total - done) * (elapsed / done), 1)
