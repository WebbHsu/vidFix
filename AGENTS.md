# vidFix — agent instructions

Read this file first. Product contract: `spec.md`. Human install/use: `README.md`. Implementation map: `docs/architecture.md`. Session-to-session state, pitfalls, and open work: `docs/handoff.md`.

You are continuing a local Windows Python app for **selective face restoration of old compressed footage**. Talk to the user in Traditional Chinese. Keep UI copy in Traditional Chinese.

## What this is

- FastAPI on `127.0.0.1:8765` + vanilla `static/index.html` / `app.js` / `app.css`.
- Jobs live on disk under `work/<job_id>/`. Progress is the files, not RAM.
- Default: every segment is **skip**. Only manually tagged restore clips run processing.
- Analysis does **not** detect faces. CodeFormer runs only on tagged clips that include the `codeformer` method.
- Target: Windows, RTX 4070 12GB, ~720p H.264 `.mkv`, clips ~80 minutes.

## Hard rules (do not violate)

- Do not dump an 80-minute video to PNG/JPG frames.
- Do not use ComfyUI, cloud APIs, SeedVR2, RIFE, demosaic, face-swap, or Deepfake.
- Do not auto-select restore clips from face size / presence.
- Do not upscale the **output** (no 2×/4× export). Keep source resolution. Exception (user-approved): Real-ESRGAN may temporarily upscale a restore-tagged frame then downscale back to the original size — the finished clip stays at source resolution.
- Do not run a second restore job at the same time (VRAM).
- Do not restart a vidFix / uvicorn process the user killed. Tell them to run `run.bat`.
- Do not bind a second server if `8765` is already in use. `app.py:main()` already opens the existing URL and returns.
- Parameter changes must **not** auto-delete old `out/*.mkv`. Re-run is explicit via「清除修復結果並重跑已選段」.
- UI language stays Traditional Chinese.

## Runtime

- Always use project `.venv` (Python 3.12). System Python 3.14 cannot load torch/InsightFace.
- Start with `run.bat`, never `python app.py` from 3.14.
- Torch is installed separately (CUDA wheel, not in `requirements.txt`).
- ffmpeg / ffprobe must be on `PATH`.
- Weights: `weights/CodeFormer/codeformer.pth`, `weights/insightface/models/buffalo_l/*.onnx`, `weights/RealESRGAN/realesr-general-x4v3.pth` (+ optional `realesr-general-wdn-x4v3.pth` for denoise DNI).
- After changing `static/app.js` or `app.css`, bump the `?v=` query in `static/index.html` (currently js `v=21`, css `v=18`). Ask the user to Ctrl+F5.

## Where to edit

| Change | Files |
|---|---|
| HTTP API, browse dialog, preview, health | `app.py` |
| Job JSON/JSONL, params defaults, resume | `vidfix/job.py` |
| Scene cuts, thumbs | `vidfix/analyze.py`, `vidfix/plan.py` |
| ffmpeg seek/encode/filters | `vidfix/ffmpeg_util.py` |
| Restore queue, two-pass CodeFormer, extra_vf | `vidfix/restore.py` |
| Face detect / CodeFormer / paste | `vidfix/models/restorer.py` |
| Real-ESRGAN frame enhance | `vidfix/models/realesrgan.py` |
| Offline affine smoothing | `vidfix/stabilize.py` |
| Background threads, GPU lock, tags | `vidfix/worker.py` |
| Concat + original audio | `vidfix/assemble.py` |
| UI | `static/index.html`, `static/app.js`, `static/app.css` |

## Restore methods

Checkboxes, combinable, applied in this order:

**deblock → denoise → deblur → realesrgan → codeformer**

- `deblock_strength` (mild/medium/strong): deblock + deblur (and light `hqdn3d` when deblock is on without the denoise method).
- `denoise_strength` (mild/medium/strong): **only** the denoise method. Independent of deblock_strength.
- `realesrgan_strength` (mild/medium/strong): blend between original and enhanced (0.45 / 0.75 / 1.0) plus DNI denoise for x4v3.
- Filters are CPU libavfilter. GPU lock is taken if any queued clip includes `codeformer` **or** `realesrgan`.
- Filters always go in `extra_vf` on the decode pipe when CodeFormer or Real-ESRGAN is selected — do not encode filters then decode again.
- Real-ESRGAN: upscale with the model, then `INTER_AREA` back to the original frame size. Output resolution never changes.

Do not add CAS. Do not use `deblock` `block=4`. Unsharp must stay luma-only and modest. Skin-smoothing / 美肌 is **not** a compression fix; do not add it as one.

## ffmpeg / output pitfalls (already paid for)

- Thumbnail temp files must be `0000.partial.jpg`, not `0000.jpg.partial` (ffmpeg needs a real image extension). Muxer: `image2`.
- Hybrid seek: coarse `-ss` before `-i`, fine `-ss` after. Frame counts from `clip_frame_count` so adjacent clips tessellate. Skip clips are re-encoded (`out_kind=skip_v2`), not stream-copied.
- Concat uses listed durations + `+genpts`. Audio is muxed once from the source with `-c:a copy` when every segment is kept. When segments are dropped (keep=false), audio is cut per kept run to PCM with the same frame-grid boundaries and encoded AAC once (`assemble._assemble_kept`).
- Face paste: 512 warp + inscribed-ellipse mask + `cv2.seamlessClone`. A rectangular mask shows a box around the face. A tiny ellipse looks weak.
- CodeFormer clips are two-pass: detect landmarks, Savitzky–Golay-smooth rigid affine (~0.45s), then restore with smoothed `M`. Do not replace this with causal EMA as the primary tracker.
- Thumb grid CSS: `grid-auto-rows: max-content` and `img` `min-height: 88px`. `overflow:hidden` + a squeezed row height makes thumbs look like lines.

## UI verification

If you change anything the user sees, exercise it in the browser (click, tag, preview, params). A screenshot is not verification. Check desktop layout of the thumb grid. If browser tools are missing, say so.

## Scope

Match existing style: small factual comments, no process narration in code, no unrelated refactors. Prefer editing the files above over adding new frameworks.
