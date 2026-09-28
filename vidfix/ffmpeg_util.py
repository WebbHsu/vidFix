from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

def _popen_flags() -> dict:
    if os.name == "nt":
        return {"creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0)}
    return {}


class FFmpegError(RuntimeError):
    pass


def which_ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if not exe:
        raise FFmpegError("找不到 ffmpeg。請安裝並加入 PATH。")
    return exe


def which_ffprobe() -> str:
    exe = shutil.which("ffprobe")
    if not exe:
        raise FFmpegError("找不到 ffprobe。請安裝 ffmpeg 並加入 PATH。")
    return exe


def run(
    args: list[str],
    *,
    check: bool = True,
    capture: bool = True,
    timeout: float | None = None,
) -> subprocess.CompletedProcess[str]:
    args = list(args)
    exe = os.path.basename(str(args[0])).lower() if args else ""
    if "ffmpeg" in exe and "-nostdin" not in args:
        args[1:1] = ["-nostdin"]
    kw: dict[str, Any] = {
        "args": args,
        "text": True,
        "encoding": "utf-8",
        "errors": "replace",
        "timeout": timeout,
        "stdin": subprocess.DEVNULL,
    }
    if capture:
        kw["stdout"] = subprocess.PIPE
        kw["stderr"] = subprocess.PIPE
    kw.update(_popen_flags())
    proc = subprocess.run(**kw)
    if check and proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip()
        tail = err[-2000:] if err else f"exit {proc.returncode}"
        raise FFmpegError(tail)
    return proc


def probe(path: str | Path) -> dict[str, Any]:
    ffprobe = which_ffprobe()
    src = str(Path(path))
    proc = run(
        [
            ffprobe,
            "-v",
            "error",
            "-show_format",
            "-show_streams",
            "-print_format",
            "json",
            src,
        ]
    )
    data = json.loads(proc.stdout or "{}")
    streams = data.get("streams") or []
    fmt = data.get("format") or {}
    v = next((s for s in streams if s.get("codec_type") == "video"), None)
    a = next((s for s in streams if s.get("codec_type") == "audio"), None)
    if v is None:
        raise FFmpegError("輸入檔沒有影像軌。")

    fps = _parse_rate(v.get("avg_frame_rate") or v.get("r_frame_rate") or "0/1")
    duration = _parse_float(v.get("duration")) or _parse_float(fmt.get("duration")) or 0.0
    width = int(v.get("width") or 0)
    height = int(v.get("height") or 0)
    if width <= 0 or height <= 0:
        raise FFmpegError("無法讀取解析度。")
    if duration <= 0:
        raise FFmpegError("無法讀取影片長度。")
    if fps <= 0:
        fps = 25.0

    return {
        "duration": round(float(duration), 3),
        "width": width,
        "height": height,
        "fps": fps,
        "video_codec": v.get("codec_name") or "",
        "audio_codec": (a or {}).get("codec_name") or "",
        "has_audio": a is not None,
        "nb_frames": _parse_int(v.get("nb_frames")),
    }


def extract_thumbnail(source: str | Path, t_sec: float, dest: Path, duration: float) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    # Must keep a real image extension; ffmpeg cannot guess JPEG from "*.jpg.partial".
    tmp = dest.with_name(f"{dest.stem}.partial.jpg")
    if tmp.exists():
        tmp.unlink()
    t = max(0.0, min(float(t_sec), max(duration - 0.05, 0.0)))
    try:
        _extract_one_jpeg(source, t, tmp, fast=True)
        if not tmp.exists() or tmp.stat().st_size < 32:
            raise FFmpegError("縮圖為空")
        tmp.replace(dest)
    except Exception:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        try:
            _extract_one_jpeg(source, t, tmp, fast=False)
            if not tmp.exists() or tmp.stat().st_size < 32:
                raise FFmpegError("縮圖為空")
            tmp.replace(dest)
        except Exception:
            if tmp.exists():
                tmp.unlink(missing_ok=True)
            raise


def _extract_one_jpeg(source: str | Path, t: float, tmp: Path, *, fast: bool) -> None:
    ffmpeg = which_ffmpeg()
    ts = f"{t:.3f}"
    args = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y"]
    if fast:
        args += ["-ss", ts, "-i", str(source)]
    else:
        args += ["-i", str(source), "-ss", ts]
    args += [
        "-frames:v",
        "1",
        "-an",
        "-vf",
        "scale=320:-2",
        "-q:v",
        "4",
        "-f",
        "image2",
        "-update",
        "1",
        str(tmp),
    ]
    run(args)


def clip_frame_count(t0: float, t1: float, fps: float) -> int:
    """Frame count so adjacent clips tessellate: [round(t0*fps), round(t1*fps))."""
    i0 = int(round(float(t0) * float(fps)))
    i1 = int(round(float(t1) * float(fps)))
    return max(1, i1 - i0)


def _seek_split(t0: float, preroll: float = 2.5) -> tuple[list[str], list[str]]:
    """Keyframe-accurate hybrid seek: coarse -ss before -i, fine -ss after."""
    t0 = max(0.0, float(t0))
    if t0 <= 1e-4:
        return [], []
    fast = max(0.0, t0 - preroll)
    fine = t0 - fast
    before = ["-ss", f"{fast:.6f}"]
    after = ["-ss", f"{fine:.6f}"] if fine > 1e-4 else []
    return before, after


def _h264_args(crf: int, preset: str, fps: float) -> list[str]:
    g = max(12, int(round(float(fps))))
    return [
        "-c:v",
        "libx264",
        "-preset",
        preset,
        "-crf",
        str(crf),
        "-profile:v",
        "high",
        "-level",
        "4.1",
        "-pix_fmt",
        "yuv420p",
        "-r",
        _fps_filter(fps),
        "-g",
        str(g),
        "-keyint_min",
        str(g),
        "-sc_threshold",
        "0",
        "-bf",
        "2",
        "-video_track_timescale",
        "90000",
        "-muxdelay",
        "0",
        "-muxpreload",
        "0",
    ]


def encode_skip_clip(
    source: str | Path,
    t0: float,
    t1: float,
    dest: Path,
    *,
    width: int,
    height: int,
    fps: float,
    crf: int = 18,
    preset: str = "veryfast",
) -> None:
    """Re-encode a skip segment to the shared H.264 profile (no face model)."""
    ffmpeg = which_ffmpeg()
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".partial")
    if tmp.exists():
        tmp.unlink()
    nframes = clip_frame_count(t0, t1, fps)
    before, after = _seek_split(t0)
    vf = (
        f"scale={width}:{height}:flags=bicubic,"
        f"fps={_fps_filter(fps)},format=yuv420p,setpts=PTS-STARTPTS"
    )
    args = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        *before,
        "-i",
        str(source),
        *after,
        "-an",
        "-vf",
        vf,
        "-frames:v",
        str(nframes),
        *_h264_args(crf, preset, fps),
        "-f",
        "matroska",
        str(tmp),
    ]
    try:
        run(args)
        if not tmp.exists() or tmp.stat().st_size < 64:
            raise FFmpegError("跳過段輸出為空")
        tmp.replace(dest)
    except Exception:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        raise


def _level(name: str | None, default: str = "medium") -> str:
    if name in ("mild", "medium", "strong"):
        return name
    return default


def compression_core(
    strength: str = "medium",
    kinds: list[str] | None = None,
    denoise_strength: str | None = None,
) -> str | None:
    """Middle of the vf chain (no scale/fps). kinds: deblock, denoise, deblur."""
    kinds = set(kinds or [])
    has_b = "deblock" in kinds
    has_n = "denoise" in kinds
    has_s = "deblur" in kinds
    if not (has_b or has_n or has_s):
        return None
    filt = _level(strength)
    den = _level(denoise_strength or strength)
    parts: list[str] = []
    if has_b:
        parts.append(
            {
                "mild": "deblock=filter=weak:block=8,gradfun=0.8:16",
                "medium": "deblock=filter=strong:block=8,gradfun=1.2:16",
                "strong": "deblock=filter=strong:block=8,gradfun=1.4:16",
            }[filt]
        )
    elif has_s:
        parts.append("deblock=filter=weak:block=8,gradfun=0.9:16")
    if has_n:
        parts.append(
            {
                "mild": "atadenoise=0a=0.02:0b=0.04:1a=0.02:1b=0.04:2a=0.02:2b=0.04:s=5",
                "medium": "atadenoise=0a=0.03:0b=0.06:1a=0.03:1b=0.05:2a=0.03:2b=0.05:s=7",
                "strong": "atadenoise=0a=0.04:0b=0.08:1a=0.03:1b=0.06:2a=0.03:2b=0.06:s=9,nlmeans=s=1.0:p=5:pc=3:r=5",
            }[den]
        )
    elif has_b and not has_s:
        parts.append(
            {
                "mild": "hqdn3d=1.5:1:2.5:1.5",
                "medium": "hqdn3d=2.2:1.4:3.5:2",
                "strong": "hqdn3d=3:2:5:3",
            }[filt]
        )
    if has_s:
        parts.append(
            {
                "mild": "unsharp=5:5:0.28:5:5:0.0",
                "medium": "unsharp=5:5:0.36:5:5:0.0",
                "strong": "unsharp=5:5:0.45:5:5:0.0",
            }[filt]
        )
    return ",".join(p for p in parts if p)


def compression_vf(
    width: int,
    height: int,
    fps: float,
    strength: str = "medium",
    kind: str = "deblock",
    kinds: list[str] | None = None,
    denoise_strength: str | None = None,
) -> str:
    """Non-generative whole-frame cleanup."""
    use = kinds or ([kind] if kind else ["deblock"])
    core = compression_core(strength, use, denoise_strength) or compression_core("medium", ["deblock"])
    return (
        f"scale={width}:{height}:flags=bicubic,{core},"
        f"fps={_fps_filter(fps)},format=yuv420p,setpts=PTS-STARTPTS"
    )


def encode_deblock_clip(
    source: str | Path,
    t0: float,
    t1: float,
    dest: Path,
    *,
    width: int,
    height: int,
    fps: float,
    crf: int = 16,
    preset: str = "fast",
    strength: str = "medium",
    kind: str = "deblock",
    kinds: list[str] | None = None,
    denoise_strength: str | None = None,
) -> None:
    """Re-encode a clip with compression-artifact filters (no face model)."""
    ffmpeg = which_ffmpeg()
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".partial")
    if tmp.exists():
        tmp.unlink()
    nframes = clip_frame_count(t0, t1, fps)
    before, after = _seek_split(t0)
    args = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        *before,
        "-i",
        str(source),
        *after,
        "-an",
        "-vf",
        compression_vf(
            width, height, fps, strength, kind=kind, kinds=kinds, denoise_strength=denoise_strength
        ),
        "-frames:v",
        str(nframes),
        *_h264_args(crf, preset, fps),
        "-f",
        "matroska",
        str(tmp),
    ]
    try:
        run(args, timeout=max(120.0, (t1 - t0) * 8 + 60))
        if not tmp.exists() or tmp.stat().st_size < 64:
            raise FFmpegError("去壓縮輸出為空")
        tmp.replace(dest)
    except Exception:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        raise


def preview_mp4(
    source: str | Path,
    t0: float,
    t1: float,
    dest: Path,
    has_audio: bool,
) -> None:
    ffmpeg = which_ffmpeg()
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".partial")
    if tmp.exists():
        tmp.unlink()
    dur = max(0.05, t1 - t0)
    base = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{t0:.3f}",
        "-i",
        str(source),
        "-t",
        f"{dur:.3f}",
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-crf",
        "28",
        "-pix_fmt",
        "yuv420p",
        "-sn",
        "-movflags",
        "+faststart",
    ]
    if has_audio:
        args = base + ["-c:a", "aac", "-ac", "2", "-b:a", "96k", "-f", "mp4", str(tmp)]
    else:
        args = base + ["-an", "-f", "mp4", str(tmp)]
    try:
        run(args)
        if not tmp.exists() or tmp.stat().st_size < 64:
            raise FFmpegError("預覽檔為空")
        tmp.replace(dest)
    except Exception:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        if has_audio:
            preview_mp4(source, t0, t1, dest, has_audio=False)
            return
        raise


def concat_and_mux(
    clips: list[tuple[Path, float]],
    source: str | Path,
    dest: Path,
    has_audio: bool,
    list_path: Path,
) -> None:
    ffmpeg = which_ffmpeg()
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".partial")
    if tmp.exists():
        tmp.unlink()

    lines = []
    for p, dur in clips:
        posix = p.resolve().as_posix().replace("'", "'\\''")
        lines.append(f"file '{posix}'")
        lines.append(f"duration {float(dur):.6f}")
    list_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    args = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-fflags",
        "+genpts",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(list_path),
    ]
    if has_audio:
        args += ["-i", str(source), "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "copy"]
    else:
        args += ["-map", "0:v:0", "-c:v", "copy"]
    args += ["-avoid_negative_ts", "make_zero", "-f", "matroska", str(tmp)]
    try:
        run(args)
        if not tmp.exists() or tmp.stat().st_size < 64:
            raise FFmpegError("成品為空")
        tmp.replace(dest)
    except Exception:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        raise


def decode_process(
    source: str | Path,
    t0: float,
    t1: float,
    width: int,
    height: int,
    fps: float,
    deblock: bool,
    nframes: int | None = None,
    extra_vf: str | None = None,
) -> subprocess.Popen[bytes]:
    ffmpeg = which_ffmpeg()
    if nframes is None:
        nframes = clip_frame_count(t0, t1, fps)
    before, after = _seek_split(t0)
    filters = [f"scale={width}:{height}:flags=bicubic"]
    if extra_vf:
        filters.append(extra_vf)
    elif deblock:
        filters.append("deblock=filter=weak:block=8")
    filters.append(f"fps={_fps_filter(fps)}")
    filters.append("format=bgr24")
    filters.append("setpts=PTS-STARTPTS")
    vf = ",".join(filters)
    args = [
        ffmpeg,
        "-hide_banner",
        "-nostdin",
        "-loglevel",
        "error",
        *before,
        "-i",
        str(source),
        *after,
        "-an",
        "-vf",
        vf,
        "-frames:v",
        str(nframes),
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "pipe:1",
    ]
    return subprocess.Popen(
        args,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        **_popen_flags(),
    )


def encode_process(
    dest: Path,
    width: int,
    height: int,
    fps: float,
    crf: int,
    preset: str,
) -> subprocess.Popen[bytes]:
    ffmpeg = which_ffmpeg()
    dest.parent.mkdir(parents=True, exist_ok=True)
    args = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{width}x{height}",
        "-r",
        _fps_filter(fps),
        "-i",
        "pipe:0",
        "-an",
        *_h264_args(crf, preset, fps),
        "-f",
        "matroska",
        str(dest),
    ]
    return subprocess.Popen(args, stdin=subprocess.PIPE, stderr=subprocess.PIPE, **_popen_flags())


def _parse_rate(s: str) -> float:
    try:
        if "/" in s:
            a, b = s.split("/", 1)
            aa, bb = float(a), float(b)
            if bb == 0:
                return 0.0
            return aa / bb
        return float(s)
    except (TypeError, ValueError):
        return 0.0


def _parse_float(v: Any) -> float | None:
    try:
        if v is None or v == "N/A":
            return None
        return float(v)
    except (TypeError, ValueError):
        return None


def _parse_int(v: Any) -> int | None:
    try:
        if v is None or v == "N/A":
            return None
        return int(v)
    except (TypeError, ValueError):
        return None


def _fps_filter(fps: float) -> str:
    # Keep a compact, ffmpeg-friendly rate string.
    r = round(float(fps), 6)
    if abs(r - 24000 / 1001) < 1e-4:
        return "24000/1001"
    if abs(r - 30000 / 1001) < 1e-4:
        return "30000/1001"
    if abs(r - 60000 / 1001) < 1e-4:
        return "60000/1001"
    if abs(r - round(r)) < 1e-6:
        return str(int(round(r)))
    return f"{r:.6f}".rstrip("0").rstrip(".")
