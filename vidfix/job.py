from __future__ import annotations

import json
import os
import re
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import ffmpeg_util
from .paths import WORK_DIR, job_dir

JOB_ID_RE = re.compile(r"^[0-9A-Za-z_-]{4,64}$")
RESTORE_METHODS = ("codeformer", "deblock", "deblur", "denoise")


def normalize_methods(raw) -> list[str]:
    if raw is None or raw == "":
        return []
    if isinstance(raw, str):
        parts = [p.strip() for p in raw.replace("+", ",").split(",") if p.strip()]
    else:
        parts = [str(p).strip() for p in raw]
    out: list[str] = []
    for m in parts:
        if m in RESTORE_METHODS and m not in out:
            out.append(m)
    return out


def methods_for_segment(seg: dict | None, params: dict | None = None) -> list[str]:
    ms = normalize_methods((seg or {}).get("methods"))
    if not ms:
        ms = normalize_methods((seg or {}).get("method"))
    if not ms and params:
        ms = normalize_methods(params.get("restore_methods") or params.get("restore_method"))
    return ms or ["codeformer"]


def is_kept(seg: dict | None) -> bool:
    """Output keep flag. Missing (older jobs) means keep."""
    return (seg or {}).get("keep", True) is not False


def needs_restore(seg: dict | None) -> bool:
    """Kept restore-tagged segment without a finished output."""
    s = seg or {}
    return s.get("tag") == "restore" and s.get("status") != "done" and is_kept(s)


def kept_frames(segs: list[dict[str, Any]], fps: float) -> int:
    """Frames the output will contain (same tessellation as assemble)."""
    return sum(
        ffmpeg_util.clip_frame_count(float(s["t0"]), float(s["t1"]), fps) for s in segs if is_kept(s)
    )


DEFAULT_PARAMS = {
    "fidelity": 0.40,
    "visibility": 0.60,
    "max_segment_sec": 25.0,
    "min_segment_sec": 1.0,
    "tiny_face_px": 75,
    "scene_threshold": 27.0,
    "deblock": True,
    "crf_skip": 18,
    "crf_restore": 16,
    "preset_skip": "veryfast",
    "preset_restore": "fast",
    "analyze_window_sec": 240.0,
    "kps_smooth": 0.42,
    "face_smooth": 0.70,
    "hold_miss": 2,
    "restore_method": "codeformer",
    "restore_methods": ["codeformer"],
    "deblock_strength": "medium",
    "denoise_strength": "medium",
}

_locks: dict[str, threading.RLock] = {}
_locks_guard = threading.Lock()


def _lock(job_id: str) -> threading.RLock:
    with _locks_guard:
        if job_id not in _locks:
            _locks[job_id] = threading.RLock()
        return _locks[job_id]


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def atomic_write_json(path: Path, data: Any) -> None:
    atomic_write_text(path, json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def new_job_id() -> str:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{stamp}-{uuid.uuid4().hex[:4]}"


def validate_job_id(job_id: str) -> str:
    if not JOB_ID_RE.match(job_id or ""):
        raise ValueError("無效的任務編號")
    return job_id


def create_job(source_path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    src = Path(source_path).expanduser().resolve()
    if not src.is_file():
        raise FileNotFoundError(f"找不到檔案：{src}")
    if src.suffix.lower() != ".mkv":
        raise ValueError("輸入必須是 .mkv")
    info = ffmpeg_util.probe(src)
    job_id = new_job_id()
    d = job_dir(job_id)
    d.mkdir(parents=True, exist_ok=True)
    (d / "thumbs").mkdir(exist_ok=True)
    (d / "out").mkdir(exist_ok=True)
    (d / "previews").mkdir(exist_ok=True)
    merged = dict(DEFAULT_PARAMS)
    if params:
        for k, v in params.items():
            if k in merged and v is not None:
                merged[k] = v
    job = {
        "job_id": job_id,
        "source_path": str(src),
        "source_name": src.name,
        "created_at": _now(),
        "duration": info["duration"],
        "width": info["width"],
        "height": info["height"],
        "fps": info["fps"],
        "video_codec": info["video_codec"],
        "audio_codec": info["audio_codec"],
        "has_audio": info["has_audio"],
        "params": merged,
        "total_segments": 0,
        "phase": "idle",
        "analyze_status": "pending",
        "restore_status": "pending",
        "assemble_status": "pending",
        "progress": empty_progress(),
        "error": None,
        "final_path": None,
    }
    save_job(job)
    (d / "segments.jsonl").write_text("", encoding="utf-8")
    return job


def empty_progress() -> dict[str, Any]:
    return {
        "message": "",
        "current": 0,
        "total": 0,
        "eta_sec": None,
        "started_at": None,
        "segment_index": None,
    }


def job_json_path(job_id: str) -> Path:
    return job_dir(job_id) / "job.json"


def load_job(job_id: str) -> dict[str, Any]:
    job_id = validate_job_id(job_id)
    p = job_json_path(job_id)
    if not p.is_file():
        raise FileNotFoundError(f"找不到任務：{job_id}")
    with _lock(job_id):
        return json.loads(p.read_text(encoding="utf-8"))


def save_job(job: dict[str, Any]) -> None:
    job_id = validate_job_id(job["job_id"])
    with _lock(job_id):
        atomic_write_json(job_json_path(job_id), job)


def update_job(job_id: str, **fields: Any) -> dict[str, Any]:
    with _lock(job_id):
        job = load_job(job_id)
        job.update(fields)
        save_job(job)
        return job


def set_progress(job_id: str, **progress: Any) -> None:
    with _lock(job_id):
        job = load_job(job_id)
        pg = dict(job.get("progress") or empty_progress())
        pg.update(progress)
        job["progress"] = pg
        save_job(job)


def list_jobs() -> list[dict[str, Any]]:
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    out = []
    for p in sorted(WORK_DIR.iterdir(), reverse=True):
        jp = p / "job.json"
        if not jp.is_file():
            continue
        try:
            job = json.loads(jp.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        segs = load_segments(job["job_id"]) if (p / "segments.jsonl").exists() else []
        restore_n = sum(1 for s in segs if s.get("tag") == "restore")
        done_n = sum(1 for s in segs if s.get("tag") == "restore" and s.get("status") == "done")
        job["restore_selected"] = restore_n
        job["restore_done"] = done_n
        out.append(job)
    return out


def find_existing_for_source(source_path: str) -> dict[str, Any] | None:
    """Return an existing job for this file, preferring one that already has segments."""
    try:
        src = str(Path(source_path).expanduser().resolve())
    except OSError:
        src = str(source_path)
    ranked: list[dict[str, Any]] = []
    for job in list_jobs():
        stored = str(job.get("source_path") or "")
        if stored == src or stored.replace("/", "\\") == src.replace("/", "\\"):
            ranked.append(job)
    if not ranked:
        return None
    ranked.sort(key=lambda j: int(j.get("total_segments") or 0), reverse=True)
    return ranked[0]


def serialize_segments(job_id: str, segs: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    if segs is None:
        segs = load_segments(job_id)
    out: list[dict[str, Any]] = []
    for s in segs:
        t0, t1 = float(s["t0"]), float(s["t1"])
        idx = int(s["index"])
        thumb = segment_thumb_path(job_id, idx)
        outp = segment_out_path(job_id, idx)
        out.append(
            {
                **s,
                "duration": round(t1 - t0, 3),
                "t0_tc": fmt_timecode(t0),
                "t1_tc": fmt_timecode(t1),
                "keep": is_kept(s),
                "has_thumb": thumb.is_file(),
                "has_out": outp.is_file(),
            }
        )
    return out


def load_segments(job_id: str) -> list[dict[str, Any]]:
    job_id = validate_job_id(job_id)
    p = job_dir(job_id) / "segments.jsonl"
    if not p.is_file():
        return []
    segs = []
    with p.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            segs.append(json.loads(line))
    segs.sort(key=lambda s: int(s["index"]))
    return segs


def save_segments(job_id: str, segs: list[dict[str, Any]]) -> None:
    job_id = validate_job_id(job_id)
    with _lock(job_id):
        p = job_dir(job_id) / "segments.jsonl"
        tmp = p.with_name("segments.jsonl.tmp")
        lines = [json.dumps(s, ensure_ascii=False, separators=(",", ":")) for s in segs]
        tmp.write_text(("\n".join(lines) + ("\n" if lines else "")), encoding="utf-8")
        os.replace(tmp, p)


def update_segment(job_id: str, index: int, **fields: Any) -> dict[str, Any]:
    with _lock(job_id):
        segs = load_segments(job_id)
        found = None
        for s in segs:
            if int(s["index"]) == int(index):
                s.update(fields)
                found = s
                break
        if found is None:
            raise KeyError(f"沒有第 {index} 段")
        save_segments(job_id, segs)
        return found


def segment_out_path(job_id: str, index: int) -> Path:
    return job_dir(job_id) / "out" / f"{int(index):04d}.mkv"


def segment_thumb_path(job_id: str, index: int) -> Path:
    return job_dir(job_id) / "thumbs" / f"{int(index):04d}.jpg"


def stop_flag_path(job_id: str) -> Path:
    return job_dir(job_id) / "STOP"


def request_stop(job_id: str) -> None:
    job_id = validate_job_id(job_id)
    stop_flag_path(job_id).write_text("1", encoding="utf-8")


def clear_stop(job_id: str) -> None:
    p = stop_flag_path(job_id)
    if p.exists():
        p.unlink()


def should_stop(job_id: str) -> bool:
    return stop_flag_path(job_id).exists()


def append_log(job_id: str, message: str) -> None:
    p = job_dir(job_id) / "job.log"
    ts = time.strftime("%H:%M:%S")
    with p.open("a", encoding="utf-8") as f:
        f.write(f"[{ts}] {message}\n")


def delete_partials(root: Path) -> int:
    n = 0
    if not root.exists():
        return 0
    for p in root.rglob("*"):
        if p.is_file() and (".partial" in p.name or p.name.endswith(".tmp")):
            try:
                p.unlink()
                n += 1
            except OSError:
                pass
    return n


def recover_jobs_on_startup() -> None:
    """Disk is the source of truth. Drop in-flight partials and mark running as stopped."""
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    for d in WORK_DIR.iterdir():
        if not d.is_dir():
            continue
        jp = d / "job.json"
        if not jp.is_file():
            continue
        delete_partials(d)
        stop = d / "STOP"
        if stop.exists():
            try:
                stop.unlink()
            except OSError:
                pass
        try:
            job = json.loads(jp.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        changed = False
        for key in ("analyze_status", "restore_status", "assemble_status"):
            if job.get(key) == "running":
                job[key] = "stopped"
                changed = True
        if job.get("phase") in ("analyze", "restore", "assemble"):
            # leave phase as-is so the user can resume the same step
            pass
        if changed:
            job["progress"] = empty_progress() | {"message": "上次中斷，可繼續"}
            job["error"] = None
            try:
                atomic_write_json(jp, job)
            except OSError:
                pass
        # sync restore done flags from files
        try:
            segs = load_segments(job["job_id"])
        except Exception:
            continue
        dirty = False
        for s in segs:
            outp = segment_out_path(job["job_id"], s["index"])
            if s.get("tag") == "restore" and s.get("status") != "done" and outp.is_file():
                s["status"] = "done"
                s["out_kind"] = "restore"
                dirty = True
            if s.get("tag") == "restore" and s.get("status") == "done" and not outp.is_file():
                s["status"] = "pending"
                s["out_kind"] = None
                dirty = True
        if dirty:
            save_segments(job["job_id"], segs)


def fmt_timecode(sec: float) -> str:
    sec = max(0.0, float(sec))
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = sec % 60
    if h:
        return f"{h}:{m:02d}:{s:06.3f}"
    return f"{m:02d}:{s:06.3f}"


def public_job(job: dict[str, Any], segs: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    if segs is None:
        try:
            segs = load_segments(job["job_id"])
        except Exception:
            segs = []
    restore = [s for s in segs if s.get("tag") == "restore"]
    thumbs_missing = sum(
        1 for s in segs if not segment_thumb_path(job["job_id"], int(s["index"])).is_file()
    )
    cover = None
    for s in segs:
        p = segment_thumb_path(job["job_id"], int(s["index"]))
        if p.is_file() and p.stat().st_size >= 2500:
            cover = int(s["index"])
            break
    if cover is None:
        for s in segs:
            p = segment_thumb_path(job["job_id"], int(s["index"]))
            if p.is_file() and p.stat().st_size >= 32:
                cover = int(s["index"])
                break
    jid = job["job_id"]
    kept = [s for s in segs if is_kept(s)]
    try:
        fps = float(job.get("fps") or 0)
    except (TypeError, ValueError):
        fps = 0.0
    kept_duration = round(kept_frames(segs, fps) / fps, 3) if fps > 0 else None
    return {
        **job,
        "kept_count": len(kept),
        "kept_duration": kept_duration,
        "restore_selected": len(restore),
        "restore_done": sum(1 for s in restore if s.get("status") == "done"),
        "restore_failed": sum(1 for s in restore if s.get("status") == "failed"),
        "thumbs_missing": thumbs_missing,
        "cover_thumb": (f"/api/jobs/{jid}/thumbs/{cover:04d}.jpg" if cover is not None else None),
        "source_exists": Path(job.get("source_path") or "").is_file(),
        "final_exists": bool(job.get("final_path") and Path(job["final_path"]).is_file()),
    }
