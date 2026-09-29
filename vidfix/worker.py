from __future__ import annotations

import threading
import traceback
from pathlib import Path
from typing import Callable

from . import analyze, assemble, ffmpeg_util, job as jobmod, restore

_guard = threading.Lock()
_threads: dict[str, threading.Thread] = {}
_busy_gpu = threading.Lock()


def _is_running(job_id: str) -> bool:
    t = _threads.get(job_id)
    return bool(t and t.is_alive())


def any_running() -> bool:
    with _guard:
        return any(t.is_alive() for t in _threads.values())


def running_job_id() -> str | None:
    with _guard:
        for jid, t in _threads.items():
            if t.is_alive():
                return jid
    return None


_STOPPED = (analyze.Stopped, restore.Stopped, assemble.Stopped)
# label -> (status key, message prefix)
_TASKS = {
    "analyze": ("analyze_status", "分析失敗"),
    "thumbs": ("analyze_status", "縮圖失敗"),
    "restore": ("restore_status", "修復失敗"),
    "assemble": ("assemble_status", "輸出失敗"),
}


def _record_failure(job_id: str, label: str, exc: BaseException) -> None:
    """Persist a background task error so the UI shows it and the step can be retried.

    run_* already log errors raised inside their own try; errors raised before that
    (missing source, nothing kept, ...) only reach here.
    """
    key, prefix = _TASKS.get(label, ("", "任務失敗"))
    msg = str(exc) or type(exc).__name__
    job = jobmod.load_job(job_id)
    if not (job.get(key) == "failed" and job.get("error") == msg):
        jobmod.append_log(job_id, f"{prefix}：{msg}")
    patch: dict = {
        "error": msg,
        "phase": "review" if int(job.get("total_segments") or 0) > 0 else "idle",
        "progress": jobmod.empty_progress() | {"message": f"{prefix}：{msg}"},
    }
    # A thumbs refill failing before it started must not undo a finished analysis.
    if key and not (label == "thumbs" and job.get(key) == "done"):
        patch[key] = "failed"
    jobmod.update_job(job_id, **patch)


def _spawn(job_id: str, target: Callable[[], None], label: str) -> None:
    with _guard:
        if _is_running(job_id):
            raise RuntimeError("此任務正在執行中")
        other = next((j for j, t in _threads.items() if t.is_alive()), None)
        if other:
            raise RuntimeError(f"另一個任務 {other} 正在執行，請先停止")
        jobmod.clear_stop(job_id)
        jobmod.delete_partials(jobmod.job_dir(job_id))

        def runner():
            try:
                target()
            except _STOPPED:
                pass
            except Exception as e:
                traceback.print_exc()
                try:
                    _record_failure(job_id, label, e)
                except Exception:
                    traceback.print_exc()
            finally:
                jobmod.clear_stop(job_id)

        t = threading.Thread(target=runner, name=f"vidfix-{label}-{job_id}", daemon=True)
        _threads[job_id] = t
        t.start()


def start_analyze(job_id: str) -> None:
    job = jobmod.load_job(job_id)
    if job.get("analyze_status") == "done":
        segs = jobmod.load_segments(job_id)
        missing = [
            s for s in segs if not jobmod.segment_thumb_path(job_id, int(s["index"])).is_file()
        ]
        if not missing:
            raise RuntimeError("分析已完成")
        _check_source(job_id, job)
        _spawn(job_id, lambda: analyze.run_thumbs_only(job_id), "thumbs")
        return
    _check_source(job_id, job)
    _spawn(job_id, lambda: analyze.run_analyze(job_id), "analyze")


def start_restore(job_id: str) -> None:
    job = jobmod.load_job(job_id)
    if job.get("analyze_status") != "done":
        raise RuntimeError("請先完成分析")
    _check_source(job_id, job)
    params = job.get("params") or {}
    need_gpu = any(
        jobmod.needs_restore(s)
        and (
            "codeformer" in jobmod.methods_for_segment(s, params)
            or "realesrgan" in jobmod.methods_for_segment(s, params)
        )
        for s in jobmod.load_segments(job_id)
    )

    def go():
        if need_gpu:
            with _busy_gpu:
                restore.run_restore(job_id)
        else:
            restore.run_restore(job_id)

    _spawn(job_id, go, "restore")


def start_assemble(job_id: str) -> None:
    job = jobmod.load_job(job_id)
    if job.get("analyze_status") != "done":
        raise RuntimeError("請先完成分析")
    if not any(jobmod.is_kept(s) for s in jobmod.load_segments(job_id)):
        raise RuntimeError(assemble.NO_KEPT_MSG)
    _check_source(job_id, job)
    _spawn(job_id, lambda: assemble.run_assemble(job_id), "assemble")


def _check_source(job_id: str, job: dict) -> None:
    try:
        jobmod.require_source(job)
    except jobmod.SourceMissing as e:
        jobmod.append_log(job_id, str(e))
        raise


def relink_source(job_id: str, path: str, tol_sec: float = 0.5) -> dict:
    """Point the job at a moved copy of the same video. Thumbs, segments and out/ stay."""
    job = jobmod.load_job(job_id)
    raw = (path or "").strip().strip('"')
    if not raw:
        raise ValueError("請輸入原始影片的完整路徑")
    src = Path(raw).expanduser()
    try:
        src = src.resolve()
    except OSError:
        pass
    if not src.is_file():
        raise ValueError(f"找不到檔案：{src}")
    if src.suffix.lower() != ".mkv":
        raise ValueError("輸入必須是 .mkv")
    try:
        info = ffmpeg_util.probe(src)
    except ffmpeg_util.FFmpegError as e:
        raise ValueError(f"讀取影片失敗：{e}") from e
    problems = []
    dur_old, dur_new = float(job.get("duration") or 0), float(info["duration"])
    if abs(dur_new - dur_old) > tol_sec:
        problems.append(f"長度 {dur_new:.3f}s（任務 {dur_old:.3f}s）")
    if (int(info["width"]), int(info["height"])) != (int(job.get("width") or 0), int(job.get("height") or 0)):
        problems.append(f"解析度 {info['width']}×{info['height']}（任務 {job.get('width')}×{job.get('height')}）")
    fps_old, fps_new = float(job.get("fps") or 0), float(info["fps"])
    if abs(fps_new - fps_old) > max(0.01, fps_old * 1e-3):
        problems.append(f"fps {fps_new:.3f}（任務 {fps_old:.3f}）")
    if bool(info["has_audio"]) != bool(job.get("has_audio")):
        problems.append("音軌 " + ("有" if info["has_audio"] else "無") + "（任務 " + ("有" if job.get("has_audio") else "無") + "）")
    if problems:
        raise ValueError("這不是同一支影片，未更換：" + "；".join(problems))
    old = job.get("source_path")
    with _guard:
        if _is_running(job_id) or any(
            job.get(k) == "running" for k in ("analyze_status", "restore_status", "assemble_status")
        ):
            raise RuntimeError("任務執行中無法更換原始影片，請先停止")
        patch: dict = {
            "source_path": str(src),
            "source_name": src.name,
            "error": None,
            "progress": jobmod.empty_progress() | {"message": "已重新指定原始影片"},
        }
        for k in ("analyze_status", "restore_status", "assemble_status"):
            if job.get(k) == "failed":
                patch[k] = "pending"
        job = jobmod.update_job(job_id, **patch)
    jobmod.append_log(job_id, f"重新指定原始影片：{old} → {src}")
    return job


def stop(job_id: str) -> None:
    jobmod.request_stop(job_id)


def set_tag(job_id: str, index: int, tag: str, method: str | None = None) -> dict:
    res = set_tags(job_id, [index], tag, method)
    return res["segments"][0]


def set_tags(job_id: str, indices: list[int], tag: str, method=None) -> dict:
    """Tag several segments in one locked read-modify-write of segments.jsonl.

    Per-segment rules match the single-segment path: skip deletes processed outputs,
    restore keeps a done clip only when its methods are unchanged.
    """
    if tag not in ("skip", "restore"):
        raise ValueError("標籤只能是 skip 或 restore")
    wanted = [int(i) for i in dict.fromkeys(int(i) for i in (indices or []))]
    if not wanted:
        raise ValueError("請至少指定一段")
    job = jobmod.load_job(job_id)
    if job.get("restore_status") == "running" or job.get("assemble_status") == "running":
        raise RuntimeError("執行中無法改標籤，請先停止")
    methods: list[str] = []
    if tag == "restore":
        raw = method if method is not None else (job.get("params") or {}).get("restore_methods")
        methods = jobmod.normalize_methods(raw)
        if not methods:
            methods = jobmod.methods_for_segment({}, job.get("params") or {})
    # Real-ESRGAN (with or without CodeFormer) outputs are out_kind "restore".
    processed = ("restore", "deblock", "deblur", "denoise")
    with jobmod._lock(job_id):
        segs = jobmod.load_segments(job_id)
        by_index = {int(s["index"]): s for s in segs}
        missing = [i for i in wanted if i not in by_index]
        if missing:
            raise KeyError(f"沒有第 {min(missing)} 段")
        updated = []
        for index in wanted:
            seg = by_index[index]
            fields: dict = {"tag": tag}
            outp = jobmod.segment_out_path(job_id, index)
            if tag == "skip":
                if outp.is_file() and seg.get("out_kind") in processed:
                    try:
                        outp.unlink()
                    except OSError:
                        pass
                fields["status"] = "pending"
                fields["out_kind"] = None
                fields["method"] = None
                fields["error"] = None
            else:
                fields["methods"] = list(methods)
                fields["method"] = "+".join(methods)
                if (
                    seg.get("status") == "done"
                    and outp.is_file()
                    and jobmod.normalize_methods(seg.get("methods") or seg.get("method")) == methods
                ):
                    fields["status"] = "done"
                else:
                    fields["status"] = "pending"
                    if outp.is_file() and seg.get("out_kind") in (*processed, "skip", "skip_v2"):
                        try:
                            outp.unlink()
                        except OSError:
                            pass
                        fields["out_kind"] = None
                    fields["error"] = None
            seg.update(fields)
            updated.append(seg)
        jobmod.save_segments(job_id, segs)
    restore_pending = any(jobmod.needs_restore(s) for s in segs)
    patch = {
        "assemble_status": "pending",
        "final_path": None,
        "phase": "review",
    }
    if restore_pending:
        patch["restore_status"] = "pending"
    jobmod.update_job(job_id, **patch)
    if len(updated) > 1:
        label = "修復" if tag == "restore" else "跳過"
        jobmod.append_log(job_id, f"批次標{label} {len(updated)} 段")
    return {
        "count": len(updated),
        "restore_count": sum(1 for s in segs if s.get("tag") == "restore"),
        "segments": updated,
    }


def set_keep(
    job_id: str,
    indices: list[int] | None = None,
    keep: bool | None = None,
    invert: bool = False,
) -> dict:
    """Set the output keep flag. indices=None means every segment.

    Outputs in out/ are left alone so a dropped segment can be re-kept for free.
    """
    if keep is None and not invert:
        raise ValueError("請指定保留或捨去")
    job = jobmod.load_job(job_id)
    if job.get("assemble_status") == "running":
        raise RuntimeError("輸出中無法改保留／捨去，請先停止")
    with jobmod._lock(job_id):
        segs = jobmod.load_segments(job_id)
        if not segs:
            raise RuntimeError("尚未分析，沒有分段。")
        wanted = None if indices is None else {int(i) for i in indices}
        if wanted is not None:
            missing = wanted - {int(s["index"]) for s in segs}
            if missing:
                raise KeyError(f"沒有第 {min(missing)} 段")
        changed = 0
        for s in segs:
            if wanted is not None and int(s["index"]) not in wanted:
                continue
            cur = jobmod.is_kept(s)
            new = (not cur) if invert else bool(keep)
            if cur != new:
                changed += 1
            s["keep"] = new
        if changed:
            jobmod.save_segments(job_id, segs)
    if changed:
        patch: dict = {"assemble_status": "pending", "final_path": None}
        if job.get("restore_status") != "running":
            patch["phase"] = "review"
            if job.get("restore_status") == "done" and any(jobmod.needs_restore(s) for s in segs):
                patch["restore_status"] = "pending"
        jobmod.update_job(job_id, **patch)
        kept = sum(1 for s in segs if jobmod.is_kept(s))
        jobmod.append_log(job_id, f"保留／捨去變更 {changed} 段，目前保留 {kept}/{len(segs)} 段")
    return {"changed": changed, "kept_count": sum(1 for s in segs if jobmod.is_kept(s))}


def clear_restore_outputs(job_id: str) -> int:
    """Delete restored files for currently selected restore segments so they can be rerun."""
    job = jobmod.load_job(job_id)
    if job.get("restore_status") == "running" or job.get("assemble_status") == "running":
        raise RuntimeError("執行中無法清除，請先停止")
    n = 0
    segs = jobmod.load_segments(job_id)
    for s in segs:
        if s.get("tag") != "restore":
            continue
        outp = jobmod.segment_out_path(job_id, s["index"])
        if outp.exists():
            outp.unlink()
            n += 1
        s["status"] = "pending"
        s["out_kind"] = None
        s["error"] = None
    jobmod.save_segments(job_id, segs)
    jobmod.update_job(
        job_id,
        restore_status="pending",
        assemble_status="pending",
        final_path=None,
        phase="review",
        progress=jobmod.empty_progress() | {"message": f"已清除 {n} 個修復檔，可重跑"},
    )
    jobmod.append_log(job_id, f"清除修復結果 {n} 檔")
    return n


def update_params(job_id: str, patch: dict) -> dict:
    job = jobmod.load_job(job_id)
    params = dict(job.get("params") or {})
    allowed = {
        "fidelity": lambda v: max(0.0, min(1.0, float(v))),
        "visibility": lambda v: max(0.0, min(1.0, float(v))),
        "tiny_face_px": lambda v: int(v),
        "deblock": lambda v: bool(v),
        "scene_threshold": lambda v: float(v),
        "max_segment_sec": lambda v: float(v),
        "min_segment_sec": lambda v: float(v),
        "crf_skip": lambda v: int(v),
        "crf_restore": lambda v: int(v),
        "restore_method": lambda v: v if v in jobmod.RESTORE_METHODS else "codeformer",
        "restore_methods": lambda v: jobmod.normalize_methods(v) or ["codeformer"],
        "deblock_strength": lambda v: v if v in ("mild", "medium", "strong") else "medium",
        "denoise_strength": lambda v: v if v in ("mild", "medium", "strong") else "medium",
        "realesrgan_model": lambda v: v if v in ("realesr-general-x4v3", "RealESRGAN_x2plus") else "realesr-general-x4v3",
        "realesrgan_strength": lambda v: v if v in ("mild", "medium", "strong") else "medium",
        "realesrgan_tile": lambda v: 0 if int(v) <= 0 else max(64, int(v)),
    }
    for k, conv in allowed.items():
        if k in patch and patch[k] is not None:
            params[k] = conv(patch[k])
    if "restore_methods" in params:
        params["restore_method"] = "+".join(params["restore_methods"])
    elif "restore_method" in patch and patch["restore_method"] is not None:
        params["restore_methods"] = jobmod.normalize_methods(params.get("restore_method"))
    job["params"] = params
    jobmod.save_job(job)
    return params
