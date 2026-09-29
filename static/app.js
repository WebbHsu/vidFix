const $ = (id) => document.getElementById(id);

const state = {
  jobId: null,
  job: null,
  segs: [],
  selected: 0,
  sel: new Set([0]),
  anchor: 0,
  poll: null,
  busy: false,
  segFp: "",
};

function fmtEta(sec) {
  if (sec == null || Number.isNaN(sec)) return "";
  const s = Math.max(0, Math.round(sec));
  const m = Math.floor(s / 60);
  const r = s % 60;
  if (m >= 60) {
    const h = Math.floor(m / 60);
    return `剩餘約 ${h} 小時 ${m % 60} 分`;
  }
  if (m > 0) return `剩餘約 ${m} 分 ${r} 秒`;
  return `剩餘約 ${r} 秒`;
}

async function api(path, opts = {}) {
  const headers = { ...(opts.headers || {}) };
  if (opts.body && !headers["Content-Type"]) headers["Content-Type"] = "application/json";
  const res = await fetch(path, { ...opts, headers });
  let data = null;
  try {
    data = await res.json();
  } catch {
    data = null;
  }
  if (!res.ok) {
    const msg = (data && (data.detail || data.message)) || res.statusText;
    throw new Error(typeof msg === "string" ? msg : JSON.stringify(msg));
  }
  return data;
}

async function refreshHealth() {
  try {
    const h = await api("/api/health");
    const pills = [
      ["ffmpeg", h.ffmpeg],
      ["場景切分", h.scenedetect],
      ["CUDA", h.cuda],
      ["CodeFormer", h.codeformer],
      ["InsightFace", h.insightface],
      ["RealESRGAN", h.realesrgan],
    ];
    $("health").innerHTML = pills
      .map(([n, ok]) => `<span class="pill ${ok ? "ok" : "bad"}">${n} ${ok ? "✓" : "✕"}</span>`)
      .join("");
    if (h.cuda_name) {
      $("health").insertAdjacentHTML(
        "beforeend",
        `<span class="pill ok">${h.cuda_name}</span>`
      );
    } else if (!h.torch) {
      $("health").insertAdjacentHTML(
        "beforeend",
        '<span class="pill bad">請用 run.bat 啟動（目前不是 .venv）</span>'
      );
    }
  } catch {
    $("health").innerHTML = '<span class="pill bad">服務未連線</span>';
  }
}

async function refreshJobs() {
  const { jobs } = await api("/api/jobs");
  const sel = $("jobSelect");
  const cur = state.jobId || "";
  sel.innerHTML = '<option value="">（新任務）</option>';
  for (const j of jobs) {
    const opt = document.createElement("option");
    opt.value = j.job_id;
    opt.textContent = `${j.source_name} · ${j.job_id} · ${j.phase} · ${j.total_segments || 0} 段`;
    sel.appendChild(opt);
  }
  sel.value = cur;
  renderJobCards(jobs);
  return jobs;
}

function renderJobCards(jobs) {
  const el = $("jobCards");
  if (!el) return;
  el.innerHTML = "";
  if (!jobs.length) {
    el.innerHTML = '<div class="hint">尚無既有任務。分析完成後會出現在這裡，點卡片即可打開縮圖。</div>';
    return;
  }
  for (const j of jobs) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "job-card" + (j.job_id === state.jobId ? " on" : "");
    const missing = j.thumbs_missing || 0;
    const n = j.total_segments || 0;
    const imgHtml = j.cover_thumb
      ? `<img src="${j.cover_thumb}" alt="">`
      : `<div class="thumb-ph">無縮圖</div>`;
    btn.innerHTML =
      `${imgHtml}<div class="meta"><b>${j.source_name}</b>` +
      `<div class="tc">${n} 段 · 分析 ${j.analyze_status}` +
      (missing ? ` · 缺縮圖 ${missing}` : "") +
      `</div></div>`;
    btn.onclick = () => {
      $("jobSelect").value = j.job_id;
      loadJob(j.job_id);
    };
    el.appendChild(btn);
  }
}

async function loadJob(id) {
  state.jobId = id;
  state.lastErr = undefined;
  state.segFp = "";
  state.selected = 0;
  state.sel = new Set([0]);
  state.anchor = 0;
  $("workspace").classList.remove("hidden");
  try {
    await refreshJob();
    if (!state.segs.length) {
      await refreshSegments(true);
    }
  } catch (e) {
    alert("開啟任務失敗：" + e.message);
    return;
  }
  startPoll();
  const panel = document.querySelector(".grid-panel");
  if (panel) panel.scrollIntoView({ block: "nearest" });
}

function renderSourceAndError(job) {
  const missing = !!job.source_missing;
  const info = $("sourcePathInfo");
  info.textContent = missing
    ? `⚠ 找不到原始影片：${job.source_path || "（未設定）"}，請確認檔案位置或重新指定`
    : `原始影片：${job.source_path || ""}`;
  info.classList.toggle("missing", missing);
  $("btnRelink").classList.toggle("warn", missing);
  $("btnRelink").disabled = !!job.running;

  const banner = $("errorBanner");
  banner.textContent = job.error ? `錯誤：${job.error}（修正後可再按一次同一個按鈕重試）` : "";
  banner.classList.toggle("hidden", !job.error);
  // Alert once when a background task fails after the job was opened.
  const err = job.error || null;
  if (state.lastErr !== undefined && err && err !== state.lastErr) {
    state.lastErr = err;
    setTimeout(() => alert("任務失敗：" + err), 0);
  }
  state.lastErr = err;
}

async function relinkSource() {
  const job = state.job;
  if (!job) return;
  let path = "";
  try {
    const r = await api("/api/browse", { method: "POST", body: "{}" });
    path = (r && r.path) || "";
  } catch {
    path = "";
  }
  if (!path) {
    const typed = prompt(
      "輸入原始影片的新完整路徑（必須是同一支影片：長度、解析度、fps 相同）",
      job.source_path || ""
    );
    if (typed === null) return;
    path = typed.trim();
  }
  if (!path) return;
  try {
    await api(`/api/jobs/${state.jobId}/source`, {
      method: "POST",
      body: JSON.stringify({ path }),
    });
    await refreshJob();
    await refreshJobs();
    alert("已更換原始影片：" + path);
  } catch (e) {
    alert(e.message);
  }
}

async function refreshJob() {
  if (!state.jobId) return;
  const job = await api(`/api/jobs/${state.jobId}`);
  state.job = job;
  if (Array.isArray(job.segments)) {
    applySegments(job.segments, false);
  }
  const running = !!job.running;
  $("fileMeta").textContent =
    `${job.source_name}　${job.width}×${job.height}　${job.fps.toFixed(3)} fps　` +
    `${jobmodFmt(job.duration)}　段數 ${job.total_segments || 0}`;
  $("fidelity").value = job.params.fidelity;
  $("visibility").value = job.params.visibility;
  $("fidelityVal").textContent = Number(job.params.fidelity).toFixed(2);
  $("visibilityVal").textContent = Number(job.params.visibility).toFixed(2);
  const methods = Array.isArray(job.params.restore_methods)
    ? job.params.restore_methods
    : String(job.params.restore_method || "codeformer").split("+").filter(Boolean);
  $("mCodeformer").checked = methods.includes("codeformer");
  $("mDeblock").checked = methods.includes("deblock");
  $("mDeblur").checked = methods.includes("deblur");
  $("mDenoise").checked = methods.includes("denoise");
  $("mRealesrgan").checked = methods.includes("realesrgan");
  $("deblockStrength").value = job.params.deblock_strength || "medium";
  $("denoiseStrength").value = job.params.denoise_strength || "medium";
  $("realesrganStrength").value = job.params.realesrgan_strength || "medium";
  syncMethodUi();

  const pg = job.progress || {};
  const total = pg.total || 0;
  const current = pg.current || 0;
  const pct = total > 0 ? Math.min(100, (current / total) * 100) : running ? 5 : 0;
  $("progressFill").style.width = `${pct}%`;
  const eta = fmtEta(pg.eta_sec);
  const msg =
    `${pg.message || job.phase || ""}　${eta}　` +
    `分析 ${job.analyze_status}　修復 ${job.restore_status}　輸出 ${job.assemble_status}` +
    (job.error ? `　錯誤：${job.error}` : "");
  $("progressText").textContent = msg;
  $("progressText").classList.toggle("err", !!job.error);
  renderSourceAndError(job);

  const thumbsMissing = (job.thumbs_missing || 0) > 0;
  $("btnAnalyze").disabled = running || (job.analyze_status === "done" && !thumbsMissing);
  $("btnAnalyze").textContent = thumbsMissing && job.analyze_status === "done" ? "補抽縮圖" : "分析";
  $("btnRestore").disabled = running || job.analyze_status !== "done";
  $("btnAssemble").disabled = running || job.analyze_status !== "done";
  $("btnStop").disabled = !running;
  $("btnClear").disabled = running;
  $("btnDownload").classList.toggle("hidden", !job.final_exists);
  $("btnDownload").href = `/api/jobs/${state.jobId}/final`;

  try {
    const log = await api(`/api/jobs/${state.jobId}/log`);
    $("log").textContent = log.text || "";
    $("log").scrollTop = $("log").scrollHeight;
  } catch {
    /* ignore */
  }
}

function jobmodFmt(sec) {
  const s = Math.max(0, Number(sec) || 0);
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const r = s % 60;
  if (h) return `${h}:${String(m).padStart(2, "0")}:${r.toFixed(1).padStart(4, "0")}`;
  return `${String(m).padStart(2, "0")}:${r.toFixed(1).padStart(4, "0")}`;
}

function pad4(n) {
  return String(n).padStart(4, "0");
}

function fingerprint(segs) {
  const list = Array.isArray(segs) ? segs : [];
  return list
    .map((s) => `${s.index}:${s.tag}:${s.status}:${s.has_thumb ? 1 : 0}:${isKept(s) ? 1 : 0}`)
    .join("|");
}

function isKept(s) {
  return !!s && s.keep !== false;
}

function keptStats() {
  const fps = (state.job && Number(state.job.fps)) || 0;
  let n = 0;
  let frames = 0;
  let sec = 0;
  for (const s of state.segs) {
    if (!isKept(s)) continue;
    n += 1;
    if (fps > 0) frames += Math.max(1, Math.round(s.t1 * fps) - Math.round(s.t0 * fps));
    else sec += s.duration;
  }
  return { n, duration: fps > 0 ? frames / fps : sec };
}

function applySegments(raw, force) {
  const segments = Array.isArray(raw) ? raw : [];
  state.segs = segments;
  pruneSelection();
  const fp = fingerprint(segments);
  const gc = $("gridCount");
  if (gc) gc.textContent = `${segments.length} 段`;
  if (!force && fp === state.segFp) {
    renderSegInfo();
    renderSelCount();
    return;
  }
  state.segFp = fp;
  renderTimeline();
  renderGrid();
  renderRestoreList();
  renderSegInfo();
  renderSelCount();
  const noSeg = !segments.length;
  for (const id of ["btnTagRestore", "btnTagSkip", "btnPreview", "btnKeep", "btnKeepAll", "btnDropAll", "btnKeepInvert"]) {
    const b = $(id);
    if (b) b.disabled = noSeg;
  }
}

async function refreshSegments(force) {
  if (!state.jobId) return;
  try {
    const data = await api(`/api/jobs/${state.jobId}/segments`);
    applySegments(data && data.segments, force);
  } catch (e) {
    applySegments([], true);
    const gc = $("gridCount");
    if (gc && state.job && state.job.total_segments) {
      gc.textContent = `${state.job.total_segments} 段（載入失敗）`;
    }
    throw e;
  }
}

function renderTimeline() {
  const el = $("timeline");
  el.innerHTML = "";
  for (const s of state.segs) {
    const d = document.createElement("div");
    const kept = isKept(s);
    d.className = `tl-seg ${s.tag}${kept ? "" : " dropped"}${s.index === state.selected ? " on" : ""}${state.sel.has(s.index) ? " sel" : ""}`;
    d.dataset.index = String(s.index);
    d.style.flex = `${Math.max(s.duration, 0.2)} 0 0`;
    d.title = `#${pad4(s.index)} ${s.t0_tc}–${s.t1_tc} ${s.tag} · ${kept ? "保留" : "捨去"}（右鍵切換，Ctrl/Shift 多選）`;
    el.appendChild(d);
  }
}

function renderGrid() {
  const el = $("grid");
  el.innerHTML = "";
  if (!state.segs.length) {
    const n = state.job && state.job.total_segments;
    const failed = state.job && state.job.analyze_status === "failed";
    if (n) {
      el.innerHTML = `<div class="empty">任務標示有 ${n} 段，但分段清單沒載入。請按 Ctrl+F5 重新整理，或再點一次既有任務卡片。</div>`;
    } else if (failed) {
      el.innerHTML = '<div class="empty">分析失敗，所以沒有可選的段。請看上方錯誤，修好後再按「分析」。</div>';
    } else {
      el.innerHTML = '<div class="empty">還沒有分段。請先按上方「分析」，完成後即可點縮圖選擇、預覽、標修復。</div>';
    }
    return;
  }
  const frag = document.createDocumentFragment();
  for (const s of state.segs) {
    const card = document.createElement("div");
    const kept = isKept(s);
    card.className = `card ${s.tag}${kept ? "" : " dropped"}${s.index === state.selected ? " on" : ""}${state.sel.has(s.index) ? " sel" : ""}`;
    card.dataset.index = String(s.index);
    if (s.has_thumb) {
      const img = document.createElement("img");
      img.loading = "lazy";
      img.alt = `#${pad4(s.index)}`;
      img.src = `/api/jobs/${state.jobId}/thumbs/${pad4(s.index)}.jpg`;
      card.appendChild(img);
    } else {
      const ph = document.createElement("div");
      ph.className = "thumb-ph";
      ph.textContent = s.t0_tc;
      card.appendChild(ph);
    }
    const cap = document.createElement("div");
    cap.className = "cap";
    let tag = "跳過";
    if (s.tag === "restore") {
      const m = methodShort(s);
      tag = s.status === "done" ? `已${m}` : m;
    }
    cap.innerHTML = `<b>#${pad4(s.index)}</b><span>${s.t0_tc} · ${tag}</span>`;
    card.appendChild(cap);
    const kb = document.createElement("button");
    kb.type = "button";
    kb.className = `keep-btn${kept ? "" : " off"}`;
    kb.textContent = kept ? "保留" : "捨去";
    kb.title = "點一下切換保留／捨去（D）";
    card.appendChild(kb);
    frag.appendChild(card);
  }
  el.appendChild(frag);
}

function renderRestoreList() {
  const items = state.segs.filter((s) => s.tag === "restore");
  $("restoreCount").textContent = String(items.length);
  const ul = $("restoreItems");
  ul.innerHTML = "";
  if (!items.length) {
    ul.innerHTML = '<li class="hint">尚未選取任何段</li>';
    return;
  }
  for (const s of items) {
    const li = document.createElement("li");
    if (s.index === state.selected) li.classList.add("on");
    const kind = methodShort(s);
    let st = s.status === "done" ? "已完成" : s.status === "failed" ? "失敗" : "佇列中";
    if (!isKept(s)) {
      li.classList.add("dropped");
      st = s.status === "done" ? "已完成（已捨去）" : "已捨去，不修";
    }
    li.innerHTML = `<div>#${String(s.index).padStart(4, "0")}　${kind}　${st}</div>
      <div class="tc">${s.t0_tc} – ${s.t1_tc}</div>`;
    li.onclick = () => selectSeg(s.index, false);
    ul.appendChild(li);
  }
}

function currentSeg() {
  return state.segs.find((s) => s.index === state.selected) || state.segs[0] || null;
}

function renderSegInfo() {
  const s = currentSeg();
  renderKeepSummary();
  if (!s) {
    $("segInfo").textContent = "尚未選段";
    $("btnKeep").textContent = "捨去 (D)";
    return;
  }
  const methodLabel = s.tag === "restore" ? methodShort(s) : "跳過";
  const kept = isKept(s);
  $("segInfo").innerHTML =
    `<b>第 ${s.index} 段</b><br>${s.t0_tc} → ${s.t1_tc}（${s.duration.toFixed(2)} 秒）<br>` +
    `標籤：${methodLabel}　狀態：${s.status}<br>成品：${kept ? "保留" : "<span class=\"bad-text\">捨去</span>"}`;
  $("btnKeep").textContent = kept ? "捨去 (D)" : "保留 (D)";
}

function renderKeepSummary() {
  const el = $("keepSummary");
  if (!el) return;
  if (!state.segs.length) {
    el.textContent = "";
    el.classList.remove("warn");
    return;
  }
  const { n, duration } = keptStats();
  const total = state.job ? jobmodFmt(state.job.duration) : "";
  if (!n) {
    el.textContent = `保留 0/${state.segs.length} 段：沒有保留任何段，無法輸出`;
  } else {
    el.textContent =
      `保留 ${n}/${state.segs.length} 段　成品長度 ${jobmodFmt(duration)}` +
      (n < state.segs.length && total ? `（原片 ${total}）` : "");
  }
  el.classList.toggle("warn", !n);
}

async function setKeep(index, keep) {
  if (!state.jobId || index == null) return;
  try {
    await api(`/api/jobs/${state.jobId}/segments/${index}/keep`, {
      method: "PATCH",
      body: JSON.stringify({ keep }),
    });
    await refreshSegments(true);
    await refreshJob();
  } catch (e) {
    alert(e.message);
  }
}

function toggleKeep(index) {
  const s = state.segs.find((x) => x.index === index);
  if (!s) return;
  setKeep(index, !isKept(s));
}

// D on a selection: any kept -> drop all, otherwise keep all. One request.
async function toggleKeepMany(indices) {
  if (!state.jobId || !indices.length) return;
  if (indices.length === 1) {
    toggleKeep(indices[0]);
    return;
  }
  const byIdx = new Map(state.segs.map((s) => [s.index, s]));
  const anyKept = indices.some((i) => isKept(byIdx.get(i)));
  try {
    await api(`/api/jobs/${state.jobId}/keep`, {
      method: "POST",
      body: JSON.stringify({ action: anyKept ? "drop" : "keep", indices }),
    });
    await refreshSegments(true);
    await refreshJob();
  } catch (e) {
    alert(e.message);
  }
}

async function bulkKeep(action) {
  if (!state.jobId || !state.segs.length) return;
  const mixed = state.segs.some((s) => isKept(s)) && state.segs.some((s) => !isKept(s));
  if (mixed && action !== "invert") {
    const label = action === "keep" ? "全部保留" : "全部捨去";
    if (!confirm(`${label}？目前的保留／捨去選擇會被覆蓋。`)) return;
  }
  try {
    await api(`/api/jobs/${state.jobId}/keep`, {
      method: "POST",
      body: JSON.stringify({ action }),
    });
    await refreshSegments(true);
    await refreshJob();
  } catch (e) {
    alert(e.message);
  }
}

function selectSeg(index, play) {
  if (!state.segs.length) return;
  state.selected = index;
  state.sel = new Set([index]);
  state.anchor = index;
  renderSelection();
  loadPreview(index, !!play);
}

function segPos(index) {
  return state.segs.findIndex((s) => s.index === index);
}

function rangeIndices(a, b) {
  let i = segPos(a);
  let j = segPos(b);
  if (i < 0) i = j;
  if (i > j) [i, j] = [j, i];
  return state.segs.slice(i, j + 1).map((s) => s.index);
}

// Indices the next F/S/D applies to, in segment order. Never empty.
function selectedIndices() {
  const out = state.segs.filter((s) => state.sel.has(s.index)).map((s) => s.index);
  if (out.length) return out;
  const cur = currentSeg();
  return cur ? [cur.index] : [];
}

function pruneSelection() {
  const have = new Set(state.segs.map((s) => s.index));
  for (const i of [...state.sel]) if (!have.has(i)) state.sel.delete(i);
  if (!state.sel.size && state.segs.length) {
    const cur = currentSeg();
    state.sel.add(cur.index);
    state.anchor = cur.index;
  }
}

// File-explorer style: plain = single, Ctrl/Cmd = toggle, Shift = range from anchor,
// Ctrl+Shift = add range. Current (preview / segInfo) follows the clicked segment.
function clickSeg(index, ev) {
  if (!state.segs.length) return;
  const ctrl = ev.ctrlKey || ev.metaKey;
  if (!ctrl && !ev.shiftKey) {
    selectSeg(index, false);
    return;
  }
  if (ev.shiftKey) {
    const range = rangeIndices(state.anchor, index);
    if (!ctrl) state.sel = new Set(range);
    else for (const i of range) state.sel.add(i);
  } else if (state.sel.has(index)) {
    if (state.sel.size > 1) state.sel.delete(index);
    state.anchor = index;
  } else {
    state.sel.add(index);
    state.anchor = index;
  }
  state.selected = index;
  renderSelection();
  loadPreview(index, false);
}

function moveCurrent(step, extend) {
  const pos = segPos(state.selected);
  const next = state.segs[Math.min(state.segs.length - 1, Math.max(0, (pos < 0 ? 0 : pos) + step))];
  if (!next) return;
  if (!extend) {
    selectSeg(next.index, false);
    return;
  }
  state.selected = next.index;
  state.sel = new Set(rangeIndices(state.anchor, next.index));
  renderSelection();
  loadPreview(next.index, false);
}

function selectAll() {
  state.sel = new Set(state.segs.map((s) => s.index));
  renderSelection();
}

function collapseSelection() {
  const cur = currentSeg();
  if (!cur) return;
  state.selected = cur.index;
  state.sel = new Set([cur.index]);
  state.anchor = cur.index;
  renderSelection();
}

// Class-only update so large grids keep their <img> nodes and scroll position.
function renderSelection() {
  for (const root of [$("grid"), $("timeline")]) {
    for (const el of root.querySelectorAll("[data-index]")) {
      const i = Number(el.dataset.index);
      el.classList.toggle("on", i === state.selected);
      el.classList.toggle("sel", state.sel.has(i));
    }
  }
  scrollGridTo(state.selected);
  renderRestoreList();
  renderSegInfo();
  renderSelCount();
}

// Scroll only the grid pane, never the page.
function scrollGridTo(index) {
  const grid = $("grid");
  const card = grid.querySelector(`[data-index="${index}"]`);
  if (!card) return;
  const top = card.offsetTop - grid.offsetTop;
  if (top < grid.scrollTop) grid.scrollTop = top - 10;
  else if (top + card.offsetHeight > grid.scrollTop + grid.clientHeight) {
    grid.scrollTop = top + card.offsetHeight - grid.clientHeight + 10;
  }
}

function renderSelCount() {
  const n = state.segs.length ? selectedIndices().length : 0;
  for (const el of document.querySelectorAll(".sel-count")) {
    el.textContent = n ? `已選 ${n} 段` : "";
    el.classList.toggle("multi", n > 1);
  }
}

function loadPreview(index, play) {
  if (!state.jobId || index == null) return;
  const player = $("player");
  const status = $("previewStatus");
  const url = `/api/jobs/${state.jobId}/segments/${index}/media`;
  player.muted = true;
  const already = player.dataset.index === String(index) && player.getAttribute("src");
  if (already) {
    status.textContent = "";
    if (play) player.play().catch(() => {});
    else player.pause();
    return;
  }
  status.textContent = play ? "載入預覽…" : "";
  player.onerror = () => {
    status.textContent = "預覽失敗。請確認原片路徑仍可讀取。";
  };
  player.oncanplay = () => {
    status.textContent = "";
    if (play) player.play().catch(() => {});
    else player.pause();
  };
  player.dataset.index = String(index);
  player.pause();
  player.src = url;
}

async function setTag(tag) {
  if (!state.jobId) return;
  if (!state.segs.length) {
    alert("還沒有分段，請先按「分析」。");
    return;
  }
  const indices = selectedIndices();
  if (!indices.length) {
    alert("請先在縮圖或時間軸點選一段。");
    return;
  }
  try {
    const body = { tag, indices };
    if (tag === "restore") {
      const ms = selectedMethods();
      if (!ms.length) {
        alert("請至少勾一種修復方式。");
        return;
      }
      body.methods = ms;
    }
    await api(`/api/jobs/${state.jobId}/tags`, {
      method: "POST",
      body: JSON.stringify(body),
    });
    await refreshSegments(true);
    await refreshJob();
  } catch (e) {
    alert(e.message);
  }
}

function startPoll() {
  stopPoll();
  state.poll = setInterval(async () => {
    if (!state.jobId) return;
    try {
      const prevCount = state.job && state.job.total_segments;
      await refreshJob();
      if (!state.job) return;
      const running = !!state.job.running;
      const segsChanged = (state.job.total_segments || 0) !== prevCount;
      if (running || segsChanged || state.job.analyze_status === "done") {
        await refreshSegments();
      }
      if (!running) await refreshJobs();
    } catch {
      /* ignore transient */
    }
  }, 1000);
}

function stopPoll() {
  if (state.poll) clearInterval(state.poll);
  state.poll = null;
}

$("btnBrowse").onclick = async () => {
  try {
    const { path } = await api("/api/browse", { method: "POST", body: "{}" });
    if (path) $("sourcePath").value = path;
  } catch (e) {
    alert(e.message);
  }
};

$("btnCreate").onclick = async () => {
  const source_path = $("sourcePath").value.trim();
  if (!source_path) {
    alert("請先選擇 MKV");
    return;
  }
  try {
    const job = await api("/api/jobs", {
      method: "POST",
      body: JSON.stringify({ source_path }),
    });
    await refreshJobs();
    $("jobSelect").value = job.job_id;
    await loadJob(job.job_id);
  } catch (e) {
    alert(e.message);
  }
};

$("btnOpen").onclick = async () => {
  const id = $("jobSelect").value;
  if (!id) return;
  await loadJob(id);
};

async function postAction(path, startsTask = false) {
  try {
    await api(path, { method: "POST", body: "{}" });
    // A task that fails again with the same message should still alert.
    if (startsTask) state.lastErr = null;
    await refreshJob();
  } catch (e) {
    alert(e.message);
  }
}

$("btnAnalyze").onclick = () => postAction(`/api/jobs/${state.jobId}/analyze`, true);
$("btnRestore").onclick = () => {
  const tagged = state.segs.filter((s) => s.tag === "restore");
  if (!tagged.length) {
    alert("尚未選取任何修復段（預設全部跳過）。請先用 F 標 3–4 段。");
    return;
  }
  if (!tagged.some((s) => isKept(s))) {
    alert("已選的修復段都被捨去了，捨去的段不會修復。請先用 D 改回保留。");
    return;
  }
  postAction(`/api/jobs/${state.jobId}/restore`, true);
};
$("btnAssemble").onclick = () => {
  if (!state.segs.some((s) => isKept(s))) {
    alert("沒有保留任何段，無法輸出。請至少保留一段（D 切換保留／捨去）。");
    return;
  }
  const pending = state.segs.filter((s) => s.tag === "restore" && s.status !== "done" && isKept(s));
  if (pending.length) {
    alert(`還有 ${pending.length} 段修復未完成，請先開始修復或改回跳過。`);
    return;
  }
  postAction(`/api/jobs/${state.jobId}/assemble`, true);
};
$("btnRelink").onclick = () => relinkSource();
$("btnStop").onclick = () => postAction(`/api/jobs/${state.jobId}/stop`);
$("btnClear").onclick = async () => {
  if (!confirm("清除已選修復段的輸出並重跑？未選的段不會動。")) return;
  await postAction(`/api/jobs/${state.jobId}/clear-restore`);
  await refreshSegments();
};

$("btnTagRestore").onclick = () => setTag("restore");
$("btnTagSkip").onclick = () => setTag("skip");
$("btnKeep").onclick = () => toggleKeepMany(selectedIndices());
$("btnKeepAll").onclick = () => bulkKeep("keep");
$("btnDropAll").onclick = () => bulkKeep("drop");
$("btnKeepInvert").onclick = () => bulkKeep("invert");
$("btnPreview").onclick = () => {
  const s = currentSeg();
  if (!s) {
    alert("還沒有分段，請先按「分析」。");
    return;
  }
  loadPreview(s.index, true);
};

$("grid").addEventListener("click", (ev) => {
  const card = ev.target.closest("[data-index]");
  if (!card) return;
  if (ev.target.closest(".keep-btn")) {
    toggleKeep(Number(card.dataset.index));
    return;
  }
  clickSeg(Number(card.dataset.index), ev);
});
$("timeline").addEventListener("click", (ev) => {
  const bit = ev.target.closest("[data-index]");
  if (!bit) return;
  clickSeg(Number(bit.dataset.index), ev);
});
$("timeline").addEventListener("contextmenu", (ev) => {
  const bit = ev.target.closest("[data-index]");
  if (!bit) return;
  ev.preventDefault();
  const i = Number(bit.dataset.index);
  toggleKeepMany(state.sel.has(i) ? selectedIndices() : [i]);
});
// Shift/Ctrl-click must not start a browser text selection.
for (const id of ["grid", "timeline"]) {
  $(id).addEventListener("mousedown", (ev) => {
    if (ev.shiftKey || ev.ctrlKey || ev.metaKey) ev.preventDefault();
  });
}

let paramTimer = null;
function onParam() {
  $("fidelityVal").textContent = Number($("fidelity").value).toFixed(2);
  $("visibilityVal").textContent = Number($("visibility").value).toFixed(2);
  clearTimeout(paramTimer);
  paramTimer = setTimeout(async () => {
    if (!state.jobId) return;
    await api(`/api/jobs/${state.jobId}/params`, {
      method: "POST",
      body: JSON.stringify({
        fidelity: Number($("fidelity").value),
        visibility: Number($("visibility").value),
        restore_methods: selectedMethods(),
        deblock_strength: $("deblockStrength").value,
        denoise_strength: $("denoiseStrength").value,
        realesrgan_strength: $("realesrganStrength").value,
      }),
    });
  }, 300);
}
function selectedMethods() {
  const m = [];
  if ($("mCodeformer") && $("mCodeformer").checked) m.push("codeformer");
  if ($("mDeblock") && $("mDeblock").checked) m.push("deblock");
  if ($("mDeblur") && $("mDeblur").checked) m.push("deblur");
  if ($("mDenoise") && $("mDenoise").checked) m.push("denoise");
  if ($("mRealesrgan") && $("mRealesrgan").checked) m.push("realesrgan");
  return m;
}

function methodShort(s) {
  const ms = Array.isArray(s.methods)
    ? s.methods
    : String(s.method || "codeformer").split("+").filter(Boolean);
  const names = { codeformer: "修臉", deblock: "去塊", deblur: "去糊", denoise: "降噪", realesrgan: "AI強化" };
  return ms.map((x) => names[x] || x).join("+") || "修臉";
}

function syncMethodUi() {
  const ms = selectedMethods();
  const cf = ms.includes("codeformer");
  const blockish = ms.includes("deblock") || ms.includes("deblur");
  const denoise = ms.includes("denoise");
  const re = ms.includes("realesrgan");
  $("fidelityRow").classList.toggle("hidden", !cf);
  $("visibilityRow").classList.toggle("hidden", !cf);
  $("deblockRow").classList.toggle("hidden", !blockish);
  $("denoiseRow").classList.toggle("hidden", !denoise);
  $("realesrganRow").classList.toggle("hidden", !re);
}

$("fidelity").oninput = onParam;
$("visibility").oninput = onParam;
$("mCodeformer").onchange = () => {
  syncMethodUi();
  onParam();
};
$("mDeblock").onchange = () => {
  syncMethodUi();
  onParam();
};
$("mDeblur").onchange = () => {
  syncMethodUi();
  onParam();
};
$("mDenoise").onchange = () => {
  syncMethodUi();
  onParam();
};
$("mRealesrgan").onchange = () => {
  syncMethodUi();
  onParam();
};
$("deblockStrength").onchange = onParam;
$("denoiseStrength").onchange = onParam;
$("realesrganStrength").onchange = onParam;

window.addEventListener("keydown", (ev) => {
  const t = ev.target;
  if (t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.tagName === "SELECT")) return;
  if (!state.jobId || !state.segs.length) return;
  const mod = ev.ctrlKey || ev.metaKey;
  if ((ev.key === "a" || ev.key === "A") && mod && !ev.altKey) {
    ev.preventDefault();
    selectAll();
    return;
  }
  if (ev.key === "Escape") {
    collapseSelection();
    return;
  }
  if (mod || ev.altKey) {
    if (ev.key !== "ArrowLeft" && ev.key !== "ArrowRight") return;
  }
  if (ev.key === "f" || ev.key === "F") {
    ev.preventDefault();
    setTag("restore");
  } else if (ev.key === "s" || ev.key === "S") {
    ev.preventDefault();
    setTag("skip");
  } else if (ev.key === "d" || ev.key === "D") {
    ev.preventDefault();
    toggleKeepMany(selectedIndices());
  } else if (ev.key === " ") {
    ev.preventDefault();
    const p = $("player");
    if (!p.src || p.dataset.index !== String(state.selected)) loadPreview(state.selected, true);
    else if (p.paused) p.play();
    else p.pause();
  } else if (ev.key === "ArrowLeft") {
    ev.preventDefault();
    moveCurrent(-1, ev.shiftKey);
  } else if (ev.key === "ArrowRight") {
    ev.preventDefault();
    moveCurrent(1, ev.shiftKey);
  }
});

(async function boot() {
  await refreshHealth();
  let jobs = [];
  try {
    jobs = (await refreshJobs()) || [];
  } catch {
    /* ignore */
  }
  const preferred =
    jobs.find((j) => (j.total_segments || 0) > 0) || jobs[0];
  if (preferred && !state.jobId) {
    $("jobSelect").value = preferred.job_id;
    try {
      await loadJob(preferred.job_id);
    } catch (e) {
      console.warn(e);
    }
  }
})();
setInterval(refreshHealth, 15000);
