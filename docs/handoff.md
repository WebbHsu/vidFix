# vidFix handoff

For the **next agent**. Read `../AGENTS.md` and `../spec.md` first, then this file for *why the code is this way* and what is still open.

Last updated: 2026-09-28.

## Current product state

The MVP in `spec.md` is implemented and has been used on a real ~80 minute H.264 MKV.

Working today:

- Create/reuse job, analyze (PySceneDetect or 25s fallback), resumable thumbs, Traditional Chinese UI.
- Tag F/S, preview from source, restore whitelist, stop/resume from disk.
- CodeFormer + InsightFace (largest face), fidelity 0.40, visibility 0.60, no upscale.
- Combinable restore methods: deblock, denoise, deblur, realesrgan, codeformer.
- Independent **deblock/deblur strength** vs **denoise strength** vs **realesrgan strength** (mild/medium/strong).
- Real-ESRGAN: enhance-at-original-resolution (upscale then INTER_AREA down). Default model `realesr-general-x4v3`.
- Assemble: skip clips re-encoded `skip_v2`, concat durations + genpts, original audio copy.
- Offline face-track smoothing (two-pass, Savitzky–Golay when scipy is installed).
- Ellipse + Poisson paste (no square “face box” if mask is left as-is).
- Keep/drop editing (user request, extends spec D): per-segment `keep` (missing = keep), D / card button / timeline right-click / bulk buttons. Dropped segments are not restored and not assembled; their `out/` files stay. All-kept assemble runs the old stream-copy code unchanged (same packets as before); with drops, audio is cut per kept run to PCM with sample-exact boundaries and re-encoded to AAC (see architecture “Assemble”).

There is at least one live job under `work/` with ~350 segments, thumbs, skip outputs, and a `final.mkv`. Treat `work/` as user data. Do not delete it.

## Environment (this machine)

- Windows, PowerShell, RTX 4070 12GB.
- Project `.venv` is Python **3.12.6** with CUDA PyTorch. `run.bat` must be used.
- System Python **3.14** is on PATH. If uvicorn is started with it: `No module named 'torch'` and CUDA pill stays off.
- ffmpeg/ffprobe on PATH. scenedetect 0.7.x and opencv are installed in the venv.
- Port **8765**. If something is already listening, `main()` opens the existing tab and does **not** start a second app. Stale 3.14 uvicorn on that port is a common “CUDA unchecked” cause — kill that process, then `run.bat`. Do **not** auto-restart a server the user just killed.

## Quality history (do not regress)

These were all user-visible bugs. The current code is the result. Re-read the cited files before “simplifying”.

### Thumbs

1. ffmpeg refused `0000.jpg.partial` because the muxer could not infer JPEG. Temp name is `0000.partial.jpg`, muxer `image2`. Then `replace` to `0000.jpg`.
2. UI “broken / 0 段”: `GET /api/jobs/{id}` must return packed segments; thumbs live below the fold; leftover `card.appendChild(img)` when `img` was undefined threw `img is not defined`.
3. Line-thin thumbs: CSS grid + `overflow:hidden` + ~42vh pane + `1fr` rows crushed 350 cards. Fix: `grid-auto-rows: max-content`, card `min-height: 118px`, img `height/min-height: 88px`, `object-fit: cover`.

### Join gap (~2 seconds of missing/wrong picture at cuts)

Keyframe-inaccurate `-ss` plus stream-copy skip clips. Fix: hybrid seek, `clip_frame_count` tessellation, re-encode skip as `skip_v2`, concat list `duration` lines, `-fflags +genpts`.

### Face look

| User report | Failed approach | Kept approach |
|---|---|---|
| Flicker | per-frame detect, no track | two-pass + SG affine (`stabilize.py`) |
| Square box around face | rectangular / eroded-square mask | inscribed ellipse on 512, corners at 0 |
| Weak restore after shrinking mask | tiny ellipse + heavy color-match | large ellipse + `seamlessClone` + visibility 0.60 |
| Face jitter remaining after causal EMA | landmark EMA only | offline smooth of (scale, rot, tx, ty), restore with `M=` |
| Geometric 8×8 lines on skin when deblock+deblur | CAS or chroma unsharp after incomplete deblock | `gradfun` after deblock; luma-only modest unsharp; never `block=4`; never CAS |

CodeFormer itself was **not** replaced. Compression cleanup is optional ffmpeg before / instead of CF.

### Methods

User asked, in order:

1. Keep CodeFormer; add a second path for over-compression blur/blocking.
2. More ways to reduce compression blur.
3. Apply any two or all methods on the same clip (checkboxes).
4. Confirm deblock/deblur/denoise are CPU (yes, libavfilter).
5. Independent denoise strength vs other filter strength.

Order is fixed: **deblock → denoise → deblur → realesrgan → codeformer**.

When CodeFormer or Real-ESRGAN is on, filters are `extra_vf` on the decode pipe (`restore.py` + `ffmpeg_util.decode_process`). Do not filter-encode then decode again.

GPU lock (`worker._busy_gpu`) if any queued clip’s methods include `codeformer` or `realesrgan`. Filter-only clips can run without it.

### Real-ESRGAN (2026-09-28)

User asked for RealScaler-class enhancement on restore-tagged clips only, without changing
output resolution. Implemented as method `realesrgan` (BSD-3 Real-ESRGAN, not RealScaler code).
Default `realesr-general-x4v3` (compact, real-world, denoise DNI). Strength = blend + DNI.
Verified on box (CPU): tile≈full (max abs 5), 720p ~6 s/frame x4v3 / ~16 s x2plus, assemble
keeps 1280×720 / 250 frames / sequential frame bar / joins intact with skip segments.

### Rejected ideas

- **美肌 / skin smoothing** as a compression fix: overlaps denoise, removes remaining grain that hides blocks, plastic skin vs blocked background makes compression *more* obvious. After CodeFormer it undoes restored texture. Do not add it unless the user wants a separate beauty look, and do not sell it as deblock.
- ComfyUI batch, full-film PNG dump, auto face-based preselect, 2×/4× **output**, RIFE.
- Copying RealScaler source (no declared license). Implement against Real-ESRGAN (BSD-3) only.

## What to do when the user asks to “make it look better”

1. Ask which clip and which methods were on. Params do not invalidate `out/*.mkv`.
2. They must「清除修復結果並重跑已選段」or re-tag after method changes.
3. Prefer retuning `compression_core` / strengths over new generative models.
4. If they see a **box**, inspect the paste mask, not CodeFormer.
5. If they see a **grid**, inspect unsharp/CAS/block size, not fidelity.
6. If they see **shake**, inspect two-pass `M` smoothing, not visibility.
7. If they see a **gap at scene cuts**, inspect seek + `skip_v2` + concat durations, not restore.

## Open / known gaps

- `scipy` is not in `requirements.txt`. `stabilize.smooth_affine_track` tries `savgol_filter` and silently falls back to interpolation. Adding scipy to the venv is a reasonable hardening step.
- README / UI copy: five methods including Real-ESRGAN. Keep in sync if you touch user-facing docs.
- `params.deblock` (bool) is a leftover. CF decode still applies weak deblock when `extra_vf` is empty. Easy to confuse with the deblock **method**.
- FaceRestorer still has causal landmark EMA + restored-face EMA for the `M is None` path. The production CF path always passes `M`. Do not delete EMA without checking that fallback.
- No automated tests except `python -m vidfix.plan`. There is no pytest suite.
- Preview MP4 uses simple `-ss` before `-i` (not hybrid). Fine for watching; do not use previews as assemble sources.
- Changing strengths/methods never auto-rebuilds outputs (spec). Users forget this constantly — remind them.
- Static cache: bump `app.js?v=` / `app.css?v=` in `index.html` on every frontend change (js is `v=19`, css `v=16` as of this file).
- Keep/drop: only the dropped-segments assemble path re-encodes audio. Do not “simplify” it to `-c:a copy` + cut (AAC frame granularity drifts at each join) or to a single `asplit`/`atrim`/`concat` graph (RAM on long films). The temp name must stay `NNNN.partial.wav` (real extension, same lesson as thumbs).
- Seen while testing keep/drop (not changed): the all-kept copy path puts the source AAC’s first packet at the video start and loses the ~21 ms encoder-delay offset, so audio is ~21 ms late (under one frame). And the hybrid seek can start a skip clip one frame late when `t0 - 2.5` falls between frames (it showed up on a video-only 25 fps test file with segment starts at whole seconds), probably because `setpts=PTS-STARTPTS` resets to the first frame after the coarse seek, not to the coarse time. Both affect the old path too; not touched here. The last segment’s `t1` is the container duration, which can be one frame past the last video frame.
- `torch` version on this machine has been CUDA 12.x (cu124 / cu128). Do not `pip install torch` from PyPI CPU wheels into `.venv`.

## Typical next tasks (if asked)

- Tune filter coefficients in `ffmpeg_util.compression_core` (keep independent denoise vs deblock strengths).
- Optional scipy in requirements + import error message.
- README/UI copy for the five methods and three strength dropdowns.
- Do **not** add 美肌 unless explicitly requested as a beauty look.
- Real-ESRGAN is the approved second GPU path (same-res enhance). Do not add a third GPU restorer without another GPU-budget discussion (12GB, one clip, one frame).
- Prefer retuning `realesrgan_strength` / `realesrgan_tile` over swapping to heavier RRDB models by default.

## How to run a change

```powershell
# from repo root, venv already exists
.\.venv\Scripts\python.exe -m compileall vidfix app.py -q
# user starts: run.bat
# user reloads UI: Ctrl+F5
```

If you change UI: exercise tag, preview, method checkboxes, both strength dropdowns, and confirm `POST /api/jobs/{id}/params` persists `denoise_strength`.

If you change restore/ffmpeg: remind the user to clear restore outputs for the clips they care about; old mkvs will otherwise be reused.
