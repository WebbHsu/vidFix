from __future__ import annotations

import threading
from typing import Callable

from . import analyze, assemble, job as jobmod, restore

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
            except Exception:
                pass
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
        _spawn(job_id, lambda: analyze.run_thumbs_only(job_id), "thumbs")
        return
    _spawn(job_id, lambda: analyze.run_analyze(job_id), "analyze")


def start_restore(job_id: str) -> None:
    job = jobmod.load_job(job_id)
    if job.get("analyze_status") != "done":
        raise RuntimeError("請先完成分析")
    params = job.get("params") or {}
    need_gpu = any(
        jobmod.needs_restore(s)
        and "codeformer" in jobmod.methods_for_segment(s, params)
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
    _spawn(job_id, lambda: assemble.run_assemble(job_id), "assemble")


def stop(job_id: str) -> None:
    jobmod.request_stop(job_id)


def set_tag(job_id: str, index: int, tag: str, method: str | None = None) -> dict:
    if tag not in ("skip", "restore"):
        raise ValueError("標籤只能是 skip 或 restore")
    job = jobmod.load_job(job_id)
    if job.get("restore_status") == "running" or job.get("assemble_status") == "running":
        raise RuntimeError("執行中無法改標籤，請先停止")
    seg = None
    for s in jobmod.load_segments(job_id):
        if int(s["index"]) == int(index):
            seg = s
            break
    if seg is None:
        raise KeyError(f"沒有第 {index} 段")

    fields: dict = {"tag": tag}
    outp = jobmod.segment_out_path(job_id, index)
    processed = ("restore", "deblock", "deblur", "denoise")
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
        raw = method if method is not None else (job.get("params") or {}).get("restore_methods")
        methods = jobmod.normalize_methods(raw)
        if not methods:
            methods = jobmod.methods_for_segment({}, job.get("params") or {})
        fields["methods"] = methods
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
    updated = jobmod.update_segment(job_id, index, **fields)
    segs = jobmod.load_segments(job_id)
    restore_pending = any(jobmod.needs_restore(s) for s in segs)
    patch = {
        "assemble_status": "pending",
        "final_path": None,
        "phase": "review",
    }
    if restore_pending:
        patch["restore_status"] = "pending"
    jobmod.update_job(job_id, **patch)
    return updated


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
