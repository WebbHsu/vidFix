from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WORK_DIR = ROOT / "work"
WEIGHTS_DIR = ROOT / "weights"
CODEFORMER_WEIGHTS = WEIGHTS_DIR / "CodeFormer" / "codeformer.pth"
INSIGHTFACE_ROOT = WEIGHTS_DIR / "insightface"
REALESRGAN_DIR = WEIGHTS_DIR / "RealESRGAN"
STATIC_DIR = ROOT / "static"


def ensure_dirs() -> None:
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    (WEIGHTS_DIR / "CodeFormer").mkdir(parents=True, exist_ok=True)
    INSIGHTFACE_ROOT.mkdir(parents=True, exist_ok=True)
    REALESRGAN_DIR.mkdir(parents=True, exist_ok=True)


def job_dir(job_id: str) -> Path:
    return WORK_DIR / job_id
