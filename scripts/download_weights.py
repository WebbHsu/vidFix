#!/usr/bin/env python3
"""Download CodeFormer + InsightFace + Real-ESRGAN weights for offline use."""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from vidfix.models.realesrgan import WEIGHT_URLS, weight_path  # noqa: E402
from vidfix.paths import CODEFORMER_WEIGHTS, INSIGHTFACE_ROOT, ensure_dirs  # noqa: E402

CODEFORMER_URL = "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth"
BUFFALO_URL = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"

UA = "vidFix-weight-downloader"

# Default Real-ESRGAN set (general-x4v3 + wdn for denoise DNI).
REALESRGAN_NAMES = ("realesr-general-x4v3", "realesr-general-wdn-x4v3")


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".partial")
    print(f"下載 {url}")
    req = Request(url, headers={"User-Agent": UA})
    with urlopen(req) as resp, tmp.open("wb") as f:
        total = int(resp.headers.get("Content-Length") or 0)
        got = 0
        while True:
            chunk = resp.read(1024 * 256)
            if not chunk:
                break
            f.write(chunk)
            got += len(chunk)
            if total:
                pct = got * 100 / total
                print(f"\r  {got / 1e6:.1f} / {total / 1e6:.1f} MB ({pct:.0f}%)", end="", flush=True)
            else:
                print(f"\r  {got / 1e6:.1f} MB", end="", flush=True)
    print()
    tmp.replace(dest)


def main() -> int:
    ensure_dirs()
    if CODEFORMER_WEIGHTS.is_file() and CODEFORMER_WEIGHTS.stat().st_size > 1_000_000:
        print(f"已有 CodeFormer：{CODEFORMER_WEIGHTS}")
    else:
        download(CODEFORMER_URL, CODEFORMER_WEIGHTS)
        print(f"已存 {CODEFORMER_WEIGHTS}")

    buffalo_dir = INSIGHTFACE_ROOT / "models" / "buffalo_l"
    onnx_files = list(buffalo_dir.glob("*.onnx")) if buffalo_dir.is_dir() else []
    if any(p.name.startswith("det_") for p in onnx_files):
        print(f"已有 InsightFace buffalo_l：{buffalo_dir}")
    else:
        zpath = INSIGHTFACE_ROOT / "buffalo_l.zip"
        if not zpath.is_file():
            download(BUFFALO_URL, zpath)
        print(f"解壓到 {buffalo_dir}")
        buffalo_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zpath) as zf:
            zf.extractall(buffalo_dir)
        print("InsightFace 權重就緒")
        try:
            zpath.unlink()
        except OSError:
            pass

    for name in REALESRGAN_NAMES:
        dest = weight_path(name)
        url = WEIGHT_URLS[name]
        if dest.is_file() and dest.stat().st_size > 1_000_000:
            print(f"已有 Real-ESRGAN {name}：{dest}")
        else:
            download(url, dest)
            print(f"已存 {dest}")
    # Optional heavier x2 RRDB model (not downloaded by default).
    x2 = weight_path("RealESRGAN_x2plus")
    if x2.is_file():
        print(f"已有可選 RealESRGAN_x2plus：{x2}")
    else:
        print(
            "（可選）RealESRGAN_x2plus 未下載；預設使用 realesr-general-x4v3。"
            f" 需要時可手動下載：{WEIGHT_URLS['RealESRGAN_x2plus']}"
        )

    print("\n完成。CodeFormer 權重授權為 S-Lab License 1.0；Real-ESRGAN 權重為 BSD-3-Clause。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
