# vidFix architecture

Entry for **how the code is structured**. Product rules: `../spec.md`. Agent constraints: `../AGENTS.md`. Current status and pitfalls: `handoff.md`.

## Process

```
run.bat
  → .venv\Scripts\python.exe app.py
      → if 127.0.0.1:8765 already bound: open browser, exit
      → else uvicorn FastAPI, open http://127.0.0.1:8765
```

One process. Background work is daemon threads in `vidfix/worker.py`. Only one job thread at a time. A process-wide `_busy_gpu` lock is taken when the restore queue includes CodeFormer or Real-ESRGAN.

On startup (`app` lifespan): `ensure_dirs()`, then `job.recover_jobs_on_startup()`:

- delete any `*.partial` / `*.tmp` under `work/`
- clear `STOP` files
- mark `running` phases as `stopped`

## Layout

```
app.py                 FastAPI routes + main()
run.bat                must use .venv (not system 3.14)
requirements.txt       no torch (install CUDA wheel separately)
scripts/download_weights.py
static/                index.html, app.js, app.css
vidfix/
  paths.py             ROOT, WORK_DIR, WEIGHTS_DIR, STATIC_DIR
  job.py               job.json, segments.jsonl, params, locks
  worker.py            start/stop threads, tags, params patch, GPU lock
  analyze.py           scene cuts + thumbs (no face detection)
  plan.py              cut times → (t0,t1) 25s max, <1s merge
  restore.py           restore queue; two-pass CodeFormer; filter-only path
  stabilize.py         offline Savitzky–Golay on rigid affine
  assemble.py          skip re-encode + concat + mux original audio
  ffmpeg_util.py       probe, seek, encode, compression_core, pipes
  models/
    codeformer.py      network + VQGAN
    vqgan.py
    restorer.py        InsightFace detect, CodeFormer, ellipse paste
    realesrgan.py      SRVGGNetCompact / RRDBNet, FrameEnhancer (tile + blend)
weights/               CodeFormer + buffalo_l + RealESRGAN (gitignored in practice)
work/<job_id>/         all durable state
```

Frontend is vanilla JS. No build step. Cache-bust with `?v=` on `/static/app.js` and `/static/app.css` in `index.html`. `/` is served with `Cache-Control: no-store`.

## Disk job

`work/<job_id>/`

| File | Role |
|---|---|
| `job.json` | source path, probe, params, phase, statuses, progress, error, final_path |
| `segments.jsonl` | one JSON object per line |
| `cuts.json` | scene cut times (or `fallback: fixed-length`) |
| `thumbs/0000.jpg` | one JPEG per segment (midpoint) |
| `out/0000.mkv` | finished clip; in-flight is `0000.mkv.partial` |
| `previews/0000.mp4` | on-demand original-slice preview |
| `concat.txt` | ffmpeg concat list for assemble |
| `final.mkv` | muxed output |
| `job.log` | append-only log |
| `STOP` | cooperative stop flag |

`job_id` pattern: `YYYYMMDD-HHMMSS` + 4 hex, validated by `JOB_ID_RE`.

Creating a job for a source that already has a job with segments **reuses** that job (`find_existing_for_source`).

### Segment record

```
index, t0, t1          seconds, millisecond decimals (not frame numbers)
tag                    skip | restore     default skip
keep                   bool, default true; missing (older jobs) = keep. Independent of tag.
status                 pending | done | failed
methods                list, e.g. ["deblock","codeformer"]
method                 joined string for older readers, e.g. "deblock+codeformer"
out_kind               restore | skip_v2 | deblock | deblur | denoise | "deblock+…"
error, thumb path implied by index
```

Times stay in seconds. Frame counts for encode/decode use:

```
clip_frame_count(t0, t1, fps) = round(t1*fps) - round(t0*fps)
```

so neighbouring clips tessellate with no 1-frame hole/overlap.

### Params (`job.params`, defaults in `job.DEFAULT_PARAMS`)

| Key | Default | Notes |
|---|---|---|
| fidelity | 0.40 | CodeFormer `w`, UI 0.30–0.55 |
| visibility | 0.60 | paste mix, UI 0.50–0.70 |
| tiny_face_px | 75 | small faces: bump fidelity, cap visibility |
| max_segment_sec | 25 | plan |
| min_segment_sec | 1 | merge short scenes |
| scene_threshold | 27 | PySceneDetect |
| deblock | True | leftover flag: weak deblock on CodeFormer decode if `extra_vf` empty |
| restore_methods | `["codeformer"]` | checkboxes |
| restore_method | `"codeformer"` | joined form, kept in sync |
| deblock_strength | medium | deblock + deblur (+ companion hqdn3d) |
| denoise_strength | medium | denoise method only |
| realesrgan_model | realesr-general-x4v3 | or RealESRGAN_x2plus |
| realesrgan_strength | medium | blend + DNI denoise (mild/medium/strong) |
| realesrgan_tile | 512 | input tile size; 0 = whole frame (VRAM!) |
| crf_skip / preset_skip | 18 / veryfast | assemble skip clips |
| crf_restore / preset_restore | 16 / fast | restored clips |
| kps_smooth, face_smooth, hold_miss | 0.42, 0.70, 2 | still on FaceRestorer; two-pass path passes `M` so landmark EMA is secondary |

Allowed restore method ids: `codeformer`, `deblock`, `deblur`, `denoise`, `realesrgan`.

## HTTP (all local)

| Method | Path | |
|---|---|---|
| GET | `/api/health` | ffmpeg, torch, cuda, codeformer/insightface/realesrgan weights, scenedetect, running job |
| POST | `/api/browse` | tkinter file dialog (needs a desktop session) |
| POST | `/api/jobs` | create or reuse |
| GET | `/api/jobs` | list |
| GET | `/api/jobs/{id}` | job + packed segments |
| PATCH | `/api/jobs/{id}/segments/{i}` | tag skip/restore; optional methods |
| PATCH | `/api/jobs/{id}/segments/{i}/keep` | `{keep: bool}` |
| POST | `/api/jobs/{id}/keep` | `{action: keep\|drop\|invert, indices?}`; no indices = all segments |
| POST | `/api/jobs/{id}/params` | fidelity, visibility, methods, strengths |
| POST | `/api/jobs/{id}/analyze` \| `restore` \| `assemble` \| `stop` | |
| POST | `/api/jobs/{id}/clear-restore` | delete selected restore outputs |
| GET | `/api/jobs/{id}/thumbs/{name}.jpg` | |
| GET | `/api/jobs/{id}/segments/{i}/media` | lazy preview mp4 |
| GET | `/api/jobs/{id}/final` | download `final.mkv` |
| GET | `/api/jobs/{id}/log` | last 100 log lines |

`GET /api/jobs/{id}` also returns `kept_count` and `kept_duration` (frame-grid seconds). Serialized segments always carry an explicit `keep`.

UI polls `GET /api/jobs/{id}`. `get_job` always returns `serialize_segments` (includes `has_thumb`, timecodes). Do not assume the client will call `/segments` separately.

## Pipelines

### Analyze

1. `analyze._detect_cuts` — PySceneDetect ContentDetector. On ImportError/failure: empty cuts + `fallback: fixed-length` (25s pieces via `plan_segments`).
2. `plan_segments` — split long scenes at `max_segment_sec`, merge `< min_segment_sec`.
3. Write `segments.jsonl` with `tag=skip`.
4. One thumbnail per segment at midpoint. Resume: skip existing `thumbs/NNNN.jpg`.
5. If analyze is already `done` but thumbs missing: `run_thumbs_only`.

No face detection in this phase.

### Preview / tag

- Space / click plays `previews/NNNN.mp4` generated from the **source** with `-ss t0 -t dur` (not hybrid seek; previews are not used for concat).
- F = restore, S = skip. Tagging restore snapshots current `restore_methods` onto the segment.
- D = keep/drop toggle (also card button, timeline right-click, bulk keep-all / drop-all / invert). `worker.set_keep` writes `segments.jsonl` via `save_segments` (tmp + `os.replace`) under the job lock, never touches `out/`, and resets `assemble_status`/`final_path` when something changed. Blocked only while assemble runs.
- Changing a done restore clip’s methods, or skip→restore, deletes that `out/` file and sets `pending`.
- Restore→skip deletes processed outputs (`out_kind` in restore/deblock/deblur/denoise).

### Restore

Queue: `job.needs_restore(seg)` = `tag==restore`, `status != done`, and kept. A segment dropped while restore is running is skipped when its turn comes.

Per clip, `methods_for_segment` (segment methods, else job params, else `["codeformer"]`).

**If `codeformer` in methods** (optional `realesrgan` before CF)

1. Pass 1: decode frames (`decode_process` raw BGR pipe + `extra_vf`), `detect_kps` (largest InsightFace face) on the filtered frames.
2. `smooth_affine_track` on the whole clip (scale, rotation, tx, ty). Savitzky–Golay if scipy is present; otherwise interpolated track.
3. Pass 2: decode again with the same `extra_vf`; if `realesrgan` is selected, `FrameEnhancer.enhance` each frame (same HxW); then `restore_frame(..., M=smoothed)`.
4. Encode H.264 mkv via `encode_process` stdin pipe. Last frame padded if decode ran short.

`extra_vf = compression_core(deblock_strength, filter_kinds, denoise_strength)`. If that is empty, decode may still apply original weak `deblock=filter=weak:block=8` when `params.deblock` is true.

**If `realesrgan` without CodeFormer**

Single decode pipe with `extra_vf` → `FrameEnhancer.enhance` per frame → encode. Same resolution as source.

**If neither CodeFormer nor Real-ESRGAN**

`encode_deblock_clip` — whole-frame ffmpeg filters only, one encode.

Filter chain (`compression_core`), CPU:

| Method | Filter |
|---|---|
| deblock | `deblock` block=8 + `gradfun` (flatten leftover 8×8) |
| deblur without deblock | still a **weak** deblock+gradfun first (unsharp on residual blocks paints a grid) |
| denoise | `atadenoise`; strong also `nlmeans=s=1.0:p=5:pc=3:r=5` |
| deblock on, denoise method off, deblur off | light `hqdn3d` using **deblock_strength** |
| deblur | luma-only `unsharp=5:5:…:5:5:0.0` (chroma gain 0) |

Never CAS. Never `block=4`.

### Real-ESRGAN (`vidfix/models/realesrgan.py`)

Default model **`realesr-general-x4v3`** (SRVGGNetCompact, ~1.2M params, ~5 MB weights):
built for real-world content, compact enough to share a 12 GB card with CodeFormer, and
supports denoise DNI with the companion `realesr-general-wdn-x4v3.pth`. Alternative:
`RealESRGAN_x2plus` (RRDBNet, heavier; set `params.realesrgan_model`).

Per frame: BGR → RGB float → network (x4 or x2) → `cv2.INTER_AREA` back to original HxW →
blend with the input by `realesrgan_strength` (mild 0.45 / medium 0.75 / strong 1.0). For
x4v3, the same strength also drives DNI denoise (0.35 / 0.55 / 0.75).

Tiling: default `realesrgan_tile=512` with `tile_pad=10` on the **input**. CUDA uses fp16;
CPU fp32. Expected VRAM on RTX 4070 12GB, 720p, one frame, fp16 (excl. CUDA context):
x4v3 tile 512 ≈ 0.3 GB peak tensors; with CodeFormer also resident ≈ 3–4 GB total.
x2plus full-frame 720p ≈ ~2 GB tensors — keep tiling on. Missing weights raise a Chinese
`FileNotFoundError` only when a clip actually needs the method.

Order: **deblock → denoise → deblur → realesrgan → codeformer**.

Face paste (`restorer._paste_face`): warp 512 restored face with inverse affine, inscribed-ellipse mask (corners of the 512 square stay zero), Poisson `seamlessClone`, then visibility mix.

Tiny face (`_face_w < tiny_face_px`): fidelity raised toward 0.50–0.55, visibility capped at 0.55.

### Assemble

For every **kept** segment (`job.is_kept`; none kept → Chinese error, also rejected up front by `worker.start_assemble`):

- restore tag: use existing `out/NNNN.mkv`
- skip tag: `encode_skip_clip` unless `out_kind==skip_v2` already exists

Then, if every segment is kept, `concat_and_mux` (unchanged): concat demuxer with per-file `duration`, `+genpts`, then mux source audio `-c:a copy`. Output `work/<id>/final.mkv`.

If any segment is dropped (`assemble._assemble_kept`):

1. `audio_runs` merges adjacent kept segments into runs on the same frame grid as `clip_frame_count` (`start = round(t0*fps)/fps`, `dur = frames/fps`).
2. `ffmpeg_util.extract_audio_wav` (`video_start` / `audio_rate` come from `probe()`) cuts each run from the source to `audio_cut/NNNN.wav` (temp `NNNN.partial.wav`): coarse `-ss` before `-i`, `-copyts`, `atrim=start=<video_start + start>` on the source pts clock, then `apad`+`atrim=end_sample` to exactly `round(dur*sample_rate)` samples. PCM s16.
3. `concat_and_mux_cut`: video concat list as before (durations + genpts, `-c:v copy`), second concat input of the WAVs (PCM concat is sample-exact), audio encoded AAC (96 kb/s per channel, max 512k). `audio_cut/` is deleted afterwards.

Why not copy audio here: stream copy can only cut on codec frames (AAC 1024 samples ≈ 21 ms), so every join would drift or click. Why not `atrim`+`concat` in one filter graph from the source: `asplit` into N branches buffers decoded audio for later branches in RAM (hours of PCM on an 80-minute film).

Skip clips must share the same H.264 profile (720p-class, yuv420p, same fps/timebase). Do not `-c:v copy` skip slices from the source (keyframe-inaccurate cuts caused ~2s gaps).

## Seek convention

`_seek_split(t0, preroll=2.5)`:

- `-ss (t0-2.5)` before `-i` (fast, keyframe)
- `-ss remainder` after `-i` (accurate)

Then `-frames:v N` with tessellating `N`. Used by skip encode, deblock encode, and CodeFormer decode.

## Frontend notes

- Method checkboxes: `mCodeformer`, `mDeblock`, `mDeblur`, `mDenoise`, `mRealesrgan`.
- `deblockRow` visible if deblock or deblur checked. `denoiseRow` / `realesrganRow` visible only if their method is checked.
- `onParam` POSTs fidelity, visibility, restore_methods, deblock_strength, denoise_strength, realesrgan_strength (debounced 300ms).
- Thumb cards must not `appendChild` an undefined `img` when `has_thumb` is false (use placeholder).
- Grid must not be `grid-auto-rows: 1fr` inside a short `overflow:hidden` pane.

## Dependencies

- `requirements.txt`: FastAPI, uvicorn, opencv-headless, numpy<2, insightface, onnxruntime, scenedetect, Pillow, pydantic.
- **Not** in requirements: `torch` / `torchvision` (CUDA wheel), `scipy` (optional; stabilize falls back).
- Python 3.10–3.12. 3.14 is known-broken for this stack.
- InsightFace detection uses onnxruntime; GPU EP if present, else CPU. `det_size=(640,640)`, `allowed_modules=["detection"]`.
