# vidFix

本機應用：舊實拍影片的**選擇性臉部模糊修復**。一次只修整部片子裡的 3–4 段，其餘保持原片時間軸。分析、修復、輸出都可以中斷甚至關機，開機後從磁碟進度續跑。

目標環境：Windows、NVIDIA RTX 4070 12GB、約 720p H.264 `.mkv`。UI 為繁體中文。

## 原則

- 分析後每一段預設都是「跳過」，只有手動標成「修復」的段才跑修復方法（去塊／降噪／去糊／AI 強化／CodeFormer，可複選）。
- 分析階段不做臉部偵測；只切場景、抽縮圖、可預覽原片。
- 進度只寫在 `work/<job_id>/`，不靠 RAM／瀏覽器／GPU 狀態。
- 不要把整部 80 分鐘解成圖片；不要用 ComfyUI 當批次引擎。

## 依賴

1. **ffmpeg / ffprobe**（必須在 `PATH`）
   - Windows 可從 https://www.gyan.dev/ffmpeg/builds/ 下載，把 `bin` 加進 PATH。
   - 終端機執行 `ffmpeg -version`、`ffprobe -version` 確認。
2. **Python 3.10–3.12**（3.13/3.14 可能沒有 InsightFace／onnxruntime 輪子，請用 3.11 或 3.12）
3. **NVIDIA 驅動 + CUDA 版 PyTorch**（修復時用）

## 安裝

在專案目錄：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 先裝對應 CUDA 的 PyTorch（範例為 cu124，請依驅動調整）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

pip install -r requirements.txt
python scripts/download_weights.py
```

權重位置：

- `weights/CodeFormer/codeformer.pth`
- `weights/insightface/models/buffalo_l/*.onnx`
- `weights/RealESRGAN/realesr-general-x4v3.pth`（預設 AI 強化）
- `weights/RealESRGAN/realesr-general-wdn-x4v3.pth`（選用，控制降噪強度）
- （選用）`weights/RealESRGAN/RealESRGAN_x2plus.pth`

沒有權重仍可**分析、預覽、標籤**；按「開始修復」且該段勾了需要的方法才會載入模型。
缺少 Real-ESRGAN 權重時，只有勾了「AI 強化」的段會報錯（繁中提示執行 `download_weights.py`）。

InsightFace 偵測預設走 CPU 版 `onnxruntime`（夠用）。若要 GPU 偵測可再裝 `onnxruntime-gpu`。

## 第一次使用

1. 雙擊 `run.bat`，或 `python app.py`。瀏覽器會開 `http://127.0.0.1:8765`。
2. 「瀏覽 MKV」或貼上本機 `.mkv` 路徑 → **建立任務**。
3. 按 **分析**。可關機；重開後開啟同一個任務再按分析，會補尚未完成的切段／縮圖。
4. 在時間軸或縮圖裡找到 3–4 個重點，按 **F** 標修復、**S** 改回跳過、**Space** 預覽該段原片、左右鍵切段。
   - 不想要的段按 **D** 捨去（再按一次改回保留），也可以點縮圖右上角的「保留／捨去」或在時間軸上按右鍵。捨去的段會變暗，不修復、也不進成品。
   - 「全部保留」「全部捨去」「反選」可一次改全部段；旁邊會顯示保留段數與成品長度。
   - 保留／捨去和「修復／跳過」是分開的：捨去已修好的段不會刪修復檔，改回保留就直接沿用。
5. 按 **開始修復**。只跑白名單；可中斷續跑。已 `done` 的段不會重跑。
6. 修復完成後按 **輸出成品**。跳過段用很快的 H.264 重編（不跑臉模），得到 `work/<job_id>/final.mkv`。
   - 全部段都保留（預設）：完整長度，原片音訊 `-c:a copy` mux 回去，和以前完全一樣。
   - 有捨去的段：只按順序接回保留的段。音訊依同樣的影格邊界從原片精確切出（PCM，逐取樣），接起來再編成 AAC，接點不會音畫偏移。
   - 全部捨去時無法輸出，會顯示錯誤。

快捷鍵：`F` 修復、`S` 跳過、`D` 保留／捨去、`Space` 預覽、`←` `→` 切段。時間軸右鍵也可切換保留／捨去。

## 參數

- **修復方式（可複選）**：去方塊、降噪、去壓縮模糊、AI 強化 (Real-ESRGAN)、CodeFormer 修臉。
  順序固定：去塊 → 降噪 → 去糊 → AI 強化 → 修臉。
- **CodeFormer fidelity** 預設 `0.40`（建議 0.30–0.55）。較低較「修」、較高較保真。
- **貼回 visibility** 預設 `0.60`（建議 0.50–0.70）。
- **去塊／去糊強度**、**降噪強度**、**AI 強化強度**各自 mild/medium/strong。
- **解析度維持原片**。AI 強化會用 Real-ESRGAN 先升解析再縮回原尺寸（成品解析度不變；這是使用者核准的例外，不是輸出 2×／4×）。
- 臉太小（寬邊小於約 75px）仍修，但提高保真、降低生成感。
- 改參數**不會**自動作廢舊輸出。要重跑已選段請按「清除修復結果並重跑已選段」。

## 工作目錄與續跑

```
work/<job_id>/
  job.json
  segments.jsonl      # 一段一行：index, t0, t1（秒、毫秒）, tag, keep, status（舊任務沒有 keep＝保留）
  thumbs/0000.jpg
  out/0000.mkv
  out/0001.mkv.partial   # 寫入中；啟動時會刪掉所有 .partial
  previews/0000.mp4
  final.mkv
```

- 輸出先寫 `*.partial`，成功後 rename。
- 開機恢復：以磁碟為準；刪 `.partial`；從第一個未完成的分析／修復段繼續。
- 已修完又改回跳過：刪該段修復檔。跳過改修復：加入佇列。

## VRAM

一次只修**一段、一幀**。RTX 4070 12GB 跑 720p 單人短段足夠。不要同時開第二個修復任務。

Real-ESRGAN（預設 `realesr-general-x4v3`、fp16、tile 512）本身大約幾百 MB；與 CodeFormer 同時常駐約 3–4 GB，遠低於 12 GB。較重的 `RealESRGAN_x2plus` 請保持 tiling。

## 明確不做

解馬賽克、換臉、Deepfake、自動預選重點鏡頭、SeedVR2 / RIFE 補幀、整段載入記憶體、雲端 API。

## 開發／接手

給下一個開發者或 agent：

- 產品契約：`spec.md`
- Agent 必讀：`AGENTS.md`
- 模組與資料格式：`docs/architecture.md`
- 現況、踩過的坑、未完成項：`docs/handoff.md`

請用 `run.bat`（專案 `.venv` / Python 3.12）。系統 Python 3.14 裝不了這套 torch／InsightFace。

## 授權

應用程式碼供本機使用。CodeFormer 權重為 [S-Lab License 1.0](https://github.com/sczhou/CodeFormer/blob/master/LICENSE)；Real-ESRGAN 權重／架構為 [BSD-3-Clause](https://github.com/xinntao/Real-ESRGAN/blob/master/LICENSE)。使用前請自行確認授權是否符合你的用途。
