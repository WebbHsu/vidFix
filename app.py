from __future__ import annotations

import socket
import sys
import threading
import webbrowser
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from vidfix import ffmpeg_util, job as jobmod, worker
from vidfix.paths import STATIC_DIR, CODEFORMER_WEIGHTS, INSIGHTFACE_ROOT, REALESRGAN_DIR, ensure_dirs


@asynccontextmanager
async def lifespan(_app: FastAPI):
    ensure_dirs()
    jobmod.recover_jobs_on_startup()
    yield


app = FastAPI(title="vidFix", docs_url=None, redoc_url=None, lifespan=lifespan)
_preview_locks: dict[str, threading.Lock] = {}
_preview_guard = threading.Lock()


class CreateJobBody(BaseModel):
    source_path: str
    params: dict[str, Any] | None = None


class TagBody(BaseModel):
    tag: str
    method: str | None = None
    methods: list[str] | None = None


class KeepBody(BaseModel):
    keep: bool


class KeepBulkBody(BaseModel):
    action: str
    indices: list[int] | None = None


class ParamsBody(BaseModel):
    fidelity: float | None = Field(default=None, ge=0, le=1)
    visibility: float | None = Field(default=None, ge=0, le=1)
    deblock: bool | None = None
    tiny_face_px: int | None = None
    restore_method: str | None = None
    restore_methods: list[str] | None = None
    deblock_strength: str | None = None
    denoise_strength: str | None = None
    realesrgan_model: str | None = None
    realesrgan_strength: str | None = None
    realesrgan_tile: int | None = Field(default=None, ge=0, le=4096)


def _err(status: int, message: str) -> HTTPException:
    return HTTPException(status_code=status, detail=message)


@app.get("/api/health")
def health() -> dict[str, Any]:
    ffmpeg_ok = False
    ffprobe_ok = False
    ffmpeg_msg = ""
    try:
        ffmpeg_util.which_ffmpeg()
        ffmpeg_ok = True
    except Exception as e:
        ffmpeg_msg = str(e)
    try:
        ffmpeg_util.which_ffprobe()
        ffprobe_ok = True
    except Exception as e:
        ffmpeg_msg = ffmpeg_msg or str(e)
    cuda = False
    cuda_name = None
    torch_ok = False
    torch_ver = None
    python_exe = sys.executable
    try:
        import torch

        torch_ok = True
        torch_ver = torch.__version__
        cuda = bool(torch.cuda.is_available())
        if cuda:
            cuda_name = torch.cuda.get_device_name(0)
    except Exception:
        pass
    det_dir = INSIGHTFACE_ROOT / "models" / "buffalo_l"
    scenedetect_ok = False
    try:
        import scenedetect  # noqa: F401

        scenedetect_ok = True
    except Exception:
        pass
    return {
        "ffmpeg": ffmpeg_ok,
        "ffprobe": ffprobe_ok,
        "ffmpeg_message": ffmpeg_msg,
        "torch": torch_ok,
        "torch_version": torch_ver,
        "python": python_exe,
        "cuda": cuda,
        "cuda_name": cuda_name,
        "scenedetect": scenedetect_ok,
        "codeformer": CODEFORMER_WEIGHTS.is_file(),
        "insightface": det_dir.is_dir() and any(det_dir.glob("*.onnx")),
        "realesrgan": (REALESRGAN_DIR / "realesr-general-x4v3.pth").is_file(),
        "running_job": worker.running_job_id(),
    }


@app.post("/api/browse")
def browse() -> dict[str, str]:
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception as e:
        raise _err(500, f"無法開啟檔案對話框：{e}") from e
    root = tk.Tk()
    root.withdraw()
    try:
        root.attributes("-topmost", True)
        path = filedialog.askopenfilename(
            title="選擇 MKV",
            filetypes=[("MKV 影片", "*.mkv"), ("所有檔案", "*.*")],
        )
    finally:
        root.destroy()
    return {"path": path or ""}


@app.post("/api/jobs")
def create_job(body: CreateJobBody) -> dict[str, Any]:
    existing = jobmod.find_existing_for_source(body.source_path)
    if existing and int(existing.get("total_segments") or 0) > 0:
        segs = jobmod.load_segments(existing["job_id"])
        return {
            **jobmod.public_job(existing, segs),
            "reused": True,
            "segments": jobmod.serialize_segments(existing["job_id"], segs),
        }
    try:
        job = jobmod.create_job(body.source_path, body.params)
    except FileNotFoundError as e:
        raise _err(400, str(e)) from e
    except ValueError as e:
        raise _err(400, str(e)) from e
    except ffmpeg_util.FFmpegError as e:
        raise _err(400, f"讀取影片失敗：{e}") from e
    except Exception as e:
        raise _err(500, str(e)) from e
    return {**jobmod.public_job(job, []), "reused": False, "segments": []}


@app.get("/api/jobs")
def list_jobs() -> dict[str, Any]:
    jobs = [jobmod.public_job(j, None) for j in jobmod.list_jobs()]
    return {"jobs": jobs}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    try:
        job = jobmod.load_job(job_id)
    except (FileNotFoundError, ValueError) as e:
        raise _err(404, str(e)) from e
    segs = jobmod.load_segments(job_id)
    packed = jobmod.serialize_segments(job_id, segs)
    return {
        **jobmod.public_job(job, segs),
        "segments": packed,
        "running": worker.running_job_id() == job_id,
    }


@app.get("/api/jobs/{job_id}/segments")
def get_segments(job_id: str) -> dict[str, Any]:
    try:
        jobmod.load_job(job_id)
    except (FileNotFoundError, ValueError) as e:
        raise _err(404, str(e)) from e
    segs = jobmod.load_segments(job_id)
    return {"segments": jobmod.serialize_segments(job_id, segs)}


@app.patch("/api/jobs/{job_id}/segments/{index}")
def patch_segment(job_id: str, index: int, body: TagBody) -> dict[str, Any]:
    try:
        raw = body.methods if body.methods is not None else body.method
        seg = worker.set_tag(job_id, index, body.tag, raw)
    except RuntimeError as e:
        raise _err(409, str(e)) from e
    except (KeyError, ValueError) as e:
        raise _err(400, str(e)) from e
    return seg


@app.patch("/api/jobs/{job_id}/segments/{index}/keep")
def patch_segment_keep(job_id: str, index: int, body: KeepBody) -> dict[str, Any]:
    try:
        res = worker.set_keep(job_id, [index], keep=body.keep)
        seg = next(s for s in jobmod.load_segments(job_id) if int(s["index"]) == int(index))
    except RuntimeError as e:
        raise _err(409, str(e)) from e
    except FileNotFoundError as e:
        raise _err(404, str(e)) from e
    except KeyError as e:
        raise _err(400, str(e.args[0]) if e.args else str(e)) from e
    except ValueError as e:
        raise _err(400, str(e)) from e
    return {**seg, "keep": jobmod.is_kept(seg), "kept_count": res["kept_count"]}


@app.post("/api/jobs/{job_id}/keep")
def bulk_keep(job_id: str, body: KeepBulkBody) -> dict[str, Any]:
    """action: keep | drop | invert; indices omitted means every segment."""
    if body.action not in ("keep", "drop", "invert"):
        raise _err(400, "action 只能是 keep、drop 或 invert")
    try:
        return worker.set_keep(
            job_id,
            body.indices,
            keep=None if body.action == "invert" else body.action == "keep",
            invert=body.action == "invert",
        )
    except RuntimeError as e:
        raise _err(409, str(e)) from e
    except FileNotFoundError as e:
        raise _err(404, str(e)) from e
    except KeyError as e:
        raise _err(400, str(e.args[0]) if e.args else str(e)) from e
    except ValueError as e:
        raise _err(400, str(e)) from e


@app.post("/api/jobs/{job_id}/params")
def patch_params(job_id: str, body: ParamsBody) -> dict[str, Any]:
    try:
        jobmod.load_job(job_id)
        params = worker.update_params(job_id, body.model_dump(exclude_none=True))
    except (FileNotFoundError, ValueError) as e:
        raise _err(404, str(e)) from e
    return {"params": params}


@app.post("/api/jobs/{job_id}/analyze")
def start_analyze(job_id: str) -> dict[str, str]:
    try:
        worker.start_analyze(job_id)
    except RuntimeError as e:
        raise _err(409, str(e)) from e
    except (FileNotFoundError, ValueError) as e:
        raise _err(404, str(e)) from e
    return {"ok": "analyze"}


@app.post("/api/jobs/{job_id}/restore")
def start_restore(job_id: str) -> dict[str, str]:
    try:
        worker.start_restore(job_id)
    except RuntimeError as e:
        raise _err(409, str(e)) from e
    except (FileNotFoundError, ValueError) as e:
        raise _err(404, str(e)) from e
    return {"ok": "restore"}


@app.post("/api/jobs/{job_id}/assemble")
def start_assemble(job_id: str) -> dict[str, str]:
    try:
        worker.start_assemble(job_id)
    except RuntimeError as e:
        raise _err(409, str(e)) from e
    except (FileNotFoundError, ValueError) as e:
        raise _err(404, str(e)) from e
    return {"ok": "assemble"}


@app.post("/api/jobs/{job_id}/stop")
def stop_job(job_id: str) -> dict[str, str]:
    try:
        jobmod.load_job(job_id)
    except (FileNotFoundError, ValueError) as e:
        raise _err(404, str(e)) from e
    worker.stop(job_id)
    return {"ok": "stop"}


@app.post("/api/jobs/{job_id}/clear-restore")
def clear_restore(job_id: str) -> dict[str, Any]:
    try:
        n = worker.clear_restore_outputs(job_id)
    except RuntimeError as e:
        raise _err(409, str(e)) from e
    except (FileNotFoundError, ValueError) as e:
        raise _err(404, str(e)) from e
    return {"cleared": n}


@app.get("/api/jobs/{job_id}/thumbs/{name}")
def get_thumb(job_id: str, name: str) -> FileResponse:
    try:
        jobmod.validate_job_id(job_id)
    except ValueError as e:
        raise _err(400, str(e)) from e
    if not name.endswith(".jpg") or "/" in name or "\\" in name:
        raise _err(400, "無效的縮圖名稱")
    path = jobmod.job_dir(job_id) / "thumbs" / name
    if not path.is_file():
        raise _err(404, "沒有縮圖")
    return FileResponse(path, media_type="image/jpeg")


@app.get("/api/jobs/{job_id}/segments/{index}/media")
def get_media(job_id: str, index: int) -> FileResponse:
    try:
        job = jobmod.load_job(job_id)
    except (FileNotFoundError, ValueError) as e:
        raise _err(404, str(e)) from e
    segs = jobmod.load_segments(job_id)
    seg = next((s for s in segs if int(s["index"]) == int(index)), None)
    if seg is None:
        raise _err(404, "沒有此段")
    dest = jobmod.job_dir(job_id) / "previews" / f"{int(index):04d}.mp4"
    key = f"{job_id}:{index}"
    with _preview_guard:
        lock = _preview_locks.setdefault(key, threading.Lock())
    with lock:
        if not dest.is_file():
            try:
                ffmpeg_util.preview_mp4(
                    job["source_path"],
                    float(seg["t0"]),
                    float(seg["t1"]),
                    dest,
                    bool(job.get("has_audio")),
                )
            except Exception as e:
                raise _err(500, f"產生預覽失敗：{e}") from e
    return FileResponse(
        dest,
        media_type="video/mp4",
        headers={"Accept-Ranges": "bytes", "Cache-Control": "no-cache"},
    )


@app.get("/api/jobs/{job_id}/final")
def get_final(job_id: str) -> FileResponse:
    try:
        job = jobmod.load_job(job_id)
    except (FileNotFoundError, ValueError) as e:
        raise _err(404, str(e)) from e
    path = job.get("final_path")
    if not path or not Path(path).is_file():
        raise _err(404, "尚未輸出成品")
    name = Path(job.get("source_name") or "output.mkv").stem + "_vidfix.mkv"
    return FileResponse(path, media_type="video/x-matroska", filename=name)


@app.get("/api/jobs/{job_id}/log")
def get_log(job_id: str) -> dict[str, str]:
    try:
        jobmod.validate_job_id(job_id)
    except ValueError as e:
        raise _err(400, str(e)) from e
    p = jobmod.job_dir(job_id) / "job.log"
    if not p.is_file():
        return {"text": ""}
    lines = p.read_text(encoding="utf-8", errors="replace").splitlines()[-100:]
    return {"text": "\n".join(lines)}


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html", headers={"Cache-Control": "no-store"})


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def _port_in_use(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex((host, port)) == 0


def main() -> None:
    import uvicorn

    ensure_dirs()
    host = "127.0.0.1"
    port = 8765
    url = f"http://{host}:{port}"
    if _port_in_use(host, port):
        print(f"vidFix 已經在跑：{url}")
        print("沒有再開第二個。正在打開瀏覽器。")
        webbrowser.open(url)
        return
    threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
