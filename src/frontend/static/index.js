function escapeHtml(s) {
  const div = document.createElement("div");
  div.textContent = s;
  return div.innerHTML;
}

// A simple outline camera glyph for assets with no frames yet (still
// extracting/deduping, or a genuinely empty source) -- not decorative,
// stands in for a real thumbnail until one exists.
const CAMERA_PLACEHOLDER_SVG = `
  <svg viewBox="0 0 24 24" width="28" height="28" fill="none" stroke="currentColor" stroke-width="1.5">
    <path d="M4 8h3l1.5-2h7L17 8h3a1 1 0 0 1 1 1v10a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1V9a1 1 0 0 1 1-1z"/>
    <circle cx="12" cy="14" r="3.5"/>
  </svg>
`;

async function fetchThumbnail(assetId) {
  try {
    const resp = await fetch(`/assets/${assetId}/frames?page=1&page_size=1`);
    if (!resp.ok) return null;
    const body = await resp.json();
    return body.frames[0]?.image_url ?? null;
  } catch {
    return null;
  }
}

async function loadAssets() {
  const resp = await fetch("/assets");
  const assets = await resp.json();

  const grid = document.getElementById("asset-list");
  const empty = document.getElementById("asset-empty");
  grid.innerHTML = "";

  if (assets.length === 0) {
    empty.style.display = "block";
    return;
  }
  empty.style.display = "none";

  // Representative-frame lookups run in parallel -- N assets shouldn't mean
  // N sequential round-trips before the page settles.
  const thumbnails = await Promise.all(assets.map((a) => fetchThumbnail(a.asset_id)));

  assets.forEach((asset, idx) => {
    const imageUrl = thumbnails[idx];
    const card = document.createElement("a");
    card.className = "asset-card";
    card.href = `/assets/${asset.asset_id}/ui`;
    card.style.animationDelay = `${Math.min(idx, 24) * 25}ms`;

    const mediaHtml = imageUrl
      ? `<img src="${imageUrl}" loading="lazy" alt="Preview frame from ${escapeHtml(asset.name)}">`
      : `<div class="asset-card-placeholder">${CAMERA_PLACEHOLDER_SVG}</div>`;

    card.innerHTML = `
      <div class="asset-card-media">${mediaHtml}</div>
      <div class="asset-card-body">
        <div class="asset-card-name">${escapeHtml(asset.name)}</div>
        <div class="asset-meta mono">${asset.source_count} source(s) &middot; ${asset.frame_count} frame(s)</div>
      </div>
    `;
    grid.appendChild(card);
  });
}

document.getElementById("create-asset-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const name = document.getElementById("asset-name").value.trim();
  if (!name) return;

  const resp = await fetch("/assets", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name }),
  });
  if (!resp.ok) {
    alert("Failed to create asset: " + resp.status);
    return;
  }
  const asset = await resp.json();
  window.location.href = `/assets/${asset.asset_id}/ui`;
});

loadAssets();
