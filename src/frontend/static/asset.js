const ASSET_ID = window.location.pathname.split("/")[2];
const PAGE_SIZE = 24;

let currentView = "gallery"; // "gallery" | "search"
let currentPage = 1;
let lastItems = []; // whatever's currently rendered in the grid, for modal lookup
let sourcesById = {};
let pollTimer = null;

function formatTimestamp(ms) {
  const totalSec = Math.floor(ms / 1000);
  const h = String(Math.floor(totalSec / 3600)).padStart(2, "0");
  const m = String(Math.floor((totalSec % 3600) / 60)).padStart(2, "0");
  const s = String(totalSec % 60).padStart(2, "0");
  return `${h}:${m}:${s}`;
}

function escapeHtml(s) {
  const div = document.createElement("div");
  div.textContent = s;
  return div.innerHTML;
}

const IN_PROGRESS_STATUSES = ["queued", "extracting", "deduping", "embedding"];

// --- asset detail + source status polling ---

async function loadAssetDetail() {
  const resp = await fetch(`/assets/${ASSET_ID}`);
  if (!resp.ok) {
    document.getElementById("asset-name").textContent = "Asset not found";
    stopPolling();
    return;
  }
  const asset = await resp.json();
  document.title = `${asset.name} -- frames-extractor`;
  document.getElementById("asset-name").textContent = asset.name;
  renderSources(asset.sources);

  // Frames appear as soon as a source's stage2 output exists (dedup done),
  // well before it reaches "ready" -- refresh the gallery on every poll
  // tick too, not just the source-status list, so new frames actually show
  // up without a manual reload. Never clobbers an active search view.
  if (currentView === "gallery") {
    loadGallery(currentPage);
  }

  const anyInProgress = asset.sources.some((s) => IN_PROGRESS_STATUSES.includes(s.status));
  if (anyInProgress) {
    startPolling();
  } else {
    stopPolling();
  }
}

function renderSources(sources) {
  sourcesById = {};
  for (const s of sources) sourcesById[s.source_id] = s;

  const list = document.getElementById("source-list");
  const empty = document.getElementById("source-empty");
  list.innerHTML = "";

  if (sources.length === 0) {
    empty.style.display = "block";
    return;
  }
  empty.style.display = "none";

  for (const s of sources) {
    const li = document.createElement("li");
    li.className = "source-row";
    const countText = s.candidate_count != null ? ` -- ${s.candidate_count} frame(s)` : "";
    const retryButton =
      s.status === "error"
        ? `<button class="secondary retry-btn" data-source-id="${s.source_id}">Retry</button>`
        : "";
    li.innerHTML = `
      <div class="source-row-top">
        <div>${escapeHtml(s.filename)}${countText}</div>
        <span class="status-word">${s.status}</span>
      </div>
      ${renderStepper(s)}
      ${s.error_message ? `<div class="error-detail">${escapeHtml(s.error_message)}</div>` : ""}
      ${retryButton}
    `;
    list.appendChild(li);
  }
}

// Delegated once on the list itself, not per-button -- renderSources()
// rebuilds the list's innerHTML every poll tick, which would otherwise
// drop any directly-attached listeners.
document.getElementById("source-list").addEventListener("click", async (e) => {
  const btn = e.target.closest(".retry-btn");
  if (!btn) return;
  const sourceId = btn.dataset.sourceId;
  btn.disabled = true;
  btn.textContent = "Retrying...";
  const resp = await fetch(`/assets/${ASSET_ID}/sources/${sourceId}/retry`, { method: "POST" });
  if (!resp.ok) {
    const body = await resp.json().catch(() => ({}));
    alert("Retry failed: " + (body.detail || resp.status));
  }
  await loadAssetDetail();
});

// Which of the 3 known pipeline steps a source is on, purely from its
// existing status string -- no backend changes, this is exactly the
// information the API already returns.
const STEP_LABELS = ["Extract", "Dedup", "Embed"];

function renderStepper(source) {
  const { status, candidate_count } = source;
  let stepState; // array of "done" | "active" | "pending" | "failed", one per STEP_LABELS entry
  let readyState; // "done" | "pending" | "failed" | "active" for the final "Ready" segment

  if (status === "ready") {
    stepState = ["done", "done", "done"];
    readyState = "done";
  } else if (status === "error") {
    // Best inference available without backend changes: candidate_count is
    // only set once dedup finishes (right as embedding starts) -- so its
    // presence means the failure happened during/after embedding, not
    // during extract/dedup. Can't distinguish extract-vs-dedup failure from
    // existing fields alone, so an early failure is attributed to Extract.
    const failedIdx = candidate_count != null ? 2 : 0;
    stepState = STEP_LABELS.map((_, i) => (i < failedIdx ? "done" : i === failedIdx ? "failed" : "pending"));
    readyState = "pending";
  } else {
    const activeIdx = { queued: -1, extracting: 0, deduping: 1, embedding: 2 }[status] ?? -1;
    stepState = STEP_LABELS.map((_, i) => (i < activeIdx ? "done" : i === activeIdx ? "active" : "pending"));
    readyState = "pending";
  }

  const stepHtml = STEP_LABELS.map(
    (label, i) => `
      <div class="step ${stepState[i]}">
        <span class="step-dot"></span>
        <span class="step-label">${label}</span>
      </div>`
  ).join("");

  return `
    <div class="stepper">
      ${stepHtml}
      <div class="step ${readyState}">
        <span class="step-dot"></span>
        <span class="step-label">Ready</span>
      </div>
    </div>
  `;
}

function startPolling() {
  if (pollTimer) return;
  pollTimer = setInterval(loadAssetDetail, 3000);
}

function stopPolling() {
  if (pollTimer) {
    clearInterval(pollTimer);
    pollTimer = null;
  }
}

// --- upload ---

document.getElementById("upload-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const fileInput = document.getElementById("video-file");
  if (!fileInput.files.length) return;

  const submitBtn = document.getElementById("upload-submit");
  submitBtn.disabled = true;
  submitBtn.textContent = "Uploading...";

  const formData = new FormData();
  formData.append("file", fileInput.files[0]);

  const downscale = document.getElementById("opt-downscale").value;
  const areaRatio = document.getElementById("opt-area-ratio").value;
  const autoMask = document.getElementById("opt-auto-mask").checked;
  const hamming = document.getElementById("opt-hamming").value;
  const window_ = document.getElementById("opt-window").value;
  const crossSourceDedup = document.getElementById("opt-cross-source-dedup").checked;

  if (downscale) formData.append("downscale_factor", downscale);
  if (areaRatio) formData.append("min_event_area_ratio", areaRatio);
  if (autoMask) formData.append("auto_mask", "true");
  if (hamming) formData.append("dedup_hamming_threshold", hamming);
  if (window_) formData.append("dedup_window_size", window_);
  if (crossSourceDedup) formData.append("cross_source_dedup", "true");

  try {
    const resp = await fetch(`/assets/${ASSET_ID}/upload`, { method: "POST", body: formData });
    if (!resp.ok) {
      alert("Upload failed: " + resp.status);
      return;
    }
    document.getElementById("upload-form").reset();
    await loadAssetDetail();
  } finally {
    submitBtn.disabled = false;
    submitBtn.textContent = "Upload";
  }
});

// --- gallery / search shared grid rendering ---

// Keyed by "source_id::filename" -- stable across re-renders of the same
// underlying frame (e.g. a background poll tick re-fetching an unchanged
// gallery page), unlike a plain array index.
let selectedKeys = new Set();

function frameKey(item) {
  return `${item.source_id}::${item.image_url.split("/").pop()}`;
}

function updateSelectionUI() {
  const count = selectedKeys.size;
  const downloadBtn = document.getElementById("download-selected");
  downloadBtn.textContent = `Download selected (${count})`;
  downloadBtn.disabled = count === 0;

  const selectAll = document.getElementById("select-all");
  selectAll.checked = lastItems.length > 0 && count === lastItems.length;
  selectAll.indeterminate = count > 0 && count < lastItems.length;
}

function renderGrid(items, { showScore }) {
  // Selection survives a re-render as long as the frame is still present
  // (e.g. the same gallery page reloading mid-poll) -- only frames that
  // dropped out of the new item list are dropped from the selection.
  const newKeys = new Set(items.map(frameKey));
  selectedKeys = new Set([...selectedKeys].filter((k) => newKeys.has(k)));
  lastItems = items;

  const grid = document.getElementById("gallery");
  const empty = document.getElementById("gallery-empty");
  grid.innerHTML = "";

  if (items.length === 0) {
    empty.style.display = "block";
    updateSelectionUI();
    return;
  }
  empty.style.display = "none";

  items.forEach((item, idx) => {
    const key = frameKey(item);
    const div = document.createElement("div");
    div.className = "thumb" + (selectedKeys.has(key) ? " selected" : "");
    div.style.animationDelay = `${Math.min(idx, 24) * 20}ms`; // capped stagger -- late pages don't crawl in
    const scoreHtml =
      showScore && item.similarity_score != null
        ? `<span class="score">${item.similarity_score.toFixed(3)}</span>`
        : "";
    div.innerHTML = `
      <input type="checkbox" class="thumb-select" data-key="${key}" ${selectedKeys.has(key) ? "checked" : ""}>
      <img src="${item.image_url}" loading="lazy">
      <span class="timestamp">${formatTimestamp(item.timestamp_ms)}</span>
      ${scoreHtml}
    `;
    div.addEventListener("click", () => openModal(idx));
    const checkbox = div.querySelector(".thumb-select");
    checkbox.addEventListener("click", (e) => e.stopPropagation()); // don't also open the modal
    checkbox.addEventListener("change", () => {
      if (checkbox.checked) selectedKeys.add(key);
      else selectedKeys.delete(key);
      div.classList.toggle("selected", checkbox.checked);
      updateSelectionUI();
    });
    grid.appendChild(div);
  });

  updateSelectionUI();
}

document.getElementById("select-all").addEventListener("change", (e) => {
  selectedKeys = e.target.checked ? new Set(lastItems.map(frameKey)) : new Set();
  document.querySelectorAll(".thumb-select").forEach((cb) => {
    const checked = selectedKeys.has(cb.dataset.key);
    cb.checked = checked;
    cb.closest(".thumb").classList.toggle("selected", checked);
  });
  updateSelectionUI();
});

document.getElementById("download-selected").addEventListener("click", async () => {
  if (selectedKeys.size === 0) return;
  const btn = document.getElementById("download-selected");
  btn.disabled = true;
  btn.textContent = "Preparing zip...";

  try {
    const frames = [...selectedKeys].map((key) => {
      const [source_id, filename] = key.split("::");
      return { source_id, filename };
    });
    const resp = await fetch(`/assets/${ASSET_ID}/frames/download`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ frames }),
    });
    if (!resp.ok) {
      const body = await resp.json().catch(() => ({}));
      alert("Download failed: " + (body.detail || resp.status));
      return;
    }
    const blob = await resp.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `frames_${ASSET_ID}.zip`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  } finally {
    updateSelectionUI(); // resets the label/disabled state from the real selection count
  }
});

// --- gallery (paginated) ---

async function loadGallery(page) {
  currentView = "gallery";
  currentPage = page;
  document.getElementById("view-title").textContent = "Gallery";
  document.getElementById("search-clear").style.display = "none";

  const resp = await fetch(`/assets/${ASSET_ID}/frames?page=${page}&page_size=${PAGE_SIZE}`);
  if (!resp.ok) return;
  const body = await resp.json();

  renderGrid(body.frames, { showScore: false });
  renderPagination(body.page, body.page_size, body.total_frames);
}

function renderPagination(page, pageSize, total) {
  const el = document.getElementById("pagination");
  if (currentView !== "gallery") {
    el.innerHTML = "";
    return;
  }
  const totalPages = Math.max(1, Math.ceil(total / pageSize));
  el.innerHTML = `
    <button class="secondary" id="prev-page" ${page <= 1 ? "disabled" : ""}>&larr; Prev</button>
    <span>Page ${page} of ${totalPages} (${total} frames)</span>
    <button class="secondary" id="next-page" ${page >= totalPages ? "disabled" : ""}>Next &rarr;</button>
  `;
  document.getElementById("prev-page").addEventListener("click", () => loadGallery(page - 1));
  document.getElementById("next-page").addEventListener("click", () => loadGallery(page + 1));
}

// --- search ---

let searchInFlight = false;

// Placeholder cards shown the instant search is submitted, before the
// request resolves -- gives immediate visual feedback that the click
// registered, instead of the UI sitting motionless until results pop in
// all at once (the exact "did my click even work?" complaint this fixes).
function renderSkeleton(count) {
  const grid = document.getElementById("gallery");
  document.getElementById("gallery-empty").style.display = "none";
  grid.innerHTML = "";
  const n = Math.min(Math.max(count, 8), 24);
  for (let i = 0; i < n; i++) {
    const div = document.createElement("div");
    div.className = "thumb skeleton";
    grid.appendChild(div);
  }
}

async function runSearch() {
  // Guards against the reported double/triple-click pile-up -- once a
  // search is in flight, further clicks (or repeated Enter presses) are
  // no-ops until it resolves, rather than firing overlapping requests
  // that later land out of order.
  if (searchInFlight) return;

  const query = document.getElementById("search-query").value.trim();
  if (!query) return;
  const topK = parseInt(document.getElementById("search-topk").value, 10) || 20;

  searchInFlight = true;
  const submitBtn = document.getElementById("search-submit");
  submitBtn.disabled = true;
  submitBtn.innerHTML = '<span class="spinner"></span>Searching...';

  currentView = "search";
  document.getElementById("view-title").textContent = `Searching for "${query}"...`;
  document.getElementById("search-clear").style.display = "inline-block";
  document.getElementById("pagination").innerHTML = "";
  renderSkeleton(topK);

  try {
    const resp = await fetch(`/assets/${ASSET_ID}/search`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, top_k: topK }),
    });
    if (!resp.ok) {
      const body = await resp.json().catch(() => ({}));
      alert("Search failed: " + (body.detail || resp.status));
      document.getElementById("view-title").textContent = "Gallery";
      currentView = "gallery";
      document.getElementById("search-clear").style.display = "none";
      await loadGallery(currentPage);
      return;
    }
    const results = await resp.json();
    document.getElementById("view-title").textContent = `Search results for "${query}"`;
    renderGrid(results, { showScore: true });
  } finally {
    searchInFlight = false;
    submitBtn.disabled = false;
    submitBtn.textContent = "Search";
  }
}

document.getElementById("search-submit").addEventListener("click", runSearch);
document.getElementById("search-query").addEventListener("keydown", (e) => {
  if (e.key === "Enter") runSearch();
});
document.getElementById("search-clear").addEventListener("click", () => {
  if (searchInFlight) return;
  document.getElementById("search-query").value = "";
  loadGallery(1);
});

// --- frame detail modal ---
// Navigable within whatever's currently in lastItems (the current gallery
// page or search-result set) -- doesn't reach across pages, matching how
// lastItems itself is scoped by renderGrid().

let modalIndex = -1;

function renderModalContent(idx) {
  const item = lastItems[idx];
  document.getElementById("modal-image").src = item.image_url;
  const source = sourcesById[item.source_id];
  const scoreLine =
    item.similarity_score != null
      ? `<div>Similarity score: <span class="mono">${item.similarity_score.toFixed(4)}</span></div>`
      : "";
  document.getElementById("modal-meta").innerHTML = `
    <div>Source: ${escapeHtml(source ? source.filename : item.source_id)}</div>
    <div>Timestamp: <span class="mono">${formatTimestamp(item.timestamp_ms)}</span></div>
    ${scoreLine}
  `;
  document.getElementById("modal-counter").textContent = `${idx + 1} / ${lastItems.length}`;
  document.getElementById("modal-prev").disabled = idx <= 0;
  document.getElementById("modal-next").disabled = idx >= lastItems.length - 1;
}

function openModal(idx) {
  if (!lastItems[idx]) return;
  modalIndex = idx;
  renderModalContent(modalIndex);
  document.getElementById("modal-overlay").classList.add("open");
}

function navigateModal(delta) {
  const next = modalIndex + delta;
  if (!lastItems[next]) return;
  modalIndex = next;
  renderModalContent(modalIndex);
}

function closeModal() {
  document.getElementById("modal-overlay").classList.remove("open");
  modalIndex = -1;
}

document.getElementById("modal-close").addEventListener("click", closeModal);
document.getElementById("modal-prev").addEventListener("click", () => navigateModal(-1));
document.getElementById("modal-next").addEventListener("click", () => navigateModal(1));
document.getElementById("modal-overlay").addEventListener("click", (e) => {
  if (e.target.id === "modal-overlay") closeModal();
});

// Arrow-key / Escape navigation, active only while the modal is open --
// this listener is global but a no-op whenever "open" isn't set.
document.addEventListener("keydown", (e) => {
  if (!document.getElementById("modal-overlay").classList.contains("open")) return;
  if (e.key === "ArrowLeft") {
    e.preventDefault();
    navigateModal(-1);
  } else if (e.key === "ArrowRight") {
    e.preventDefault();
    navigateModal(1);
  } else if (e.key === "Escape") {
    closeModal();
  }
});

// --- init ---

loadAssetDetail();
loadGallery(1);
