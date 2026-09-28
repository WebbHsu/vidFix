#!/usr/bin/env python3
"""Download CodeFormer + InsightFace detection weights for offline use."""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from vidfix.paths import CODEFORMER_WEIGHTS, INSIGHTFACE_ROOT, ensure_dirs  # noqa: E402

CODEFORMER_URL = "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth"
BUFFALO_URL = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"

UA = "vidFix-weight-downloader"


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

    print("\n完成。CodeFormer 權重授權為 S-Lab License 1.0（非商業研究用途請自行確認）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
