"use strict";

const DISPLAY_SIZE = 512;

const paintCanvas = document.getElementById("paint");
const ghostCanvas = document.getElementById("ghost");
const pctx = paintCanvas.getContext("2d");
const gctx = ghostCanvas.getContext("2d");
const previewCanvas = document.getElementById("preview");
const vctx = previewCanvas.getContext("2d");
vctx.imageSmoothingEnabled = false;
pctx.imageSmoothingEnabled = false;

const statusEl = document.getElementById("status");
const radiusEl = document.getElementById("radius");
const softEl = document.getElementById("soft");
const colorEl = document.getElementById("color");
const alphaEl = document.getElementById("alpha");
const policyEl = document.getElementById("policy");
const copiesEl = document.getElementById("copies");

let tileW = 256;
let tileH = 256;
let drawing = false;
let strokePts = [];
let tileImage = null;

function setStatus(msg, kind = "ok") {
  statusEl.textContent = msg;
  statusEl.className = "status " + kind;
}

function params() {
  const hex = colorEl.value;
  const r = parseInt(hex.slice(1, 3), 16) / 255;
  const g = parseInt(hex.slice(3, 5), 16) / 255;
  const b = parseInt(hex.slice(5, 7), 16) / 255;
  return {
    radius: parseFloat(radiusEl.value),
    soft: parseFloat(softEl.value),
    color: [r, g, b, parseFloat(alphaEl.value)],
    policy: policyEl.value,
  };
}

function refreshLabels() {
  document.getElementById("radiusVal").textContent = radiusEl.value + " px";
  document.getElementById("softVal").textContent = parseFloat(softEl.value).toFixed(2);
  document.getElementById("alphaVal").textContent = parseFloat(alphaEl.value).toFixed(2);
  document.getElementById("copiesVal").textContent = copiesEl.value + " × " + copiesEl.value;
}

// ---------------------------------------------------------------------------
// Tile-local coordinates.  Everything sent to the kernel lives in
// [0, W) x [0, H); wrapping happens server-side on the torus.
// ---------------------------------------------------------------------------
function eventToTile(ev) {
  const rect = paintCanvas.getBoundingClientRect();
  const x = ((ev.clientX - rect.left) / rect.width) * tileW;
  const y = ((ev.clientY - rect.top) / rect.height) * tileH;
  return [((x % tileW) + tileW) % tileW, ((y % tileH) + tileH) % tileH];
}

// ---------------------------------------------------------------------------
// Ghost cursor: every periodic copy of the brush that intersects the unit is
// drawn, including copies from brushes larger than the tile.
// ---------------------------------------------------------------------------
function drawGhost(ev) {
  gctx.clearRect(0, 0, DISPLAY_SIZE, DISPLAY_SIZE);
  if (!ev) return;
  const [tx, ty] = eventToTile(ev);
  const p = params();
  const sx = DISPLAY_SIZE / tileW;
  const sy = DISPLAY_SIZE / tileH;
  const rr = p.radius;

  const nxLo = Math.floor((tx - rr) / tileW) - 1;
  const nxHi = Math.floor((tx + rr) / tileW) + 1;
  const nyLo = Math.floor((ty - rr) / tileH) - 1;
  const nyHi = Math.floor((ty + rr) / tileH) + 1;

  gctx.strokeStyle = "rgba(255, 255, 255, 0.9)";
  gctx.fillStyle = "rgba(255,255,255,0.12)";
  gctx.lineWidth = 1;
  for (let ny = nyLo; ny <= nyHi; ny++) {
    for (let nx = nxLo; nx <= nxHi; nx++) {
      const cx = (tx + nx * tileW) * sx;
      const cy = (ty + ny * tileH) * sy;
      gctx.beginPath();
      gctx.arc(cx, cy, rr * sx, 0, Math.PI * 2);
      gctx.fill();
      gctx.stroke();
    }
  }
}

function drawSeamGuides() {
  gctx.clearRect(0, 0, DISPLAY_SIZE, DISPLAY_SIZE);
  gctx.strokeStyle = "rgba(40, 40, 40, 0.55)";
  gctx.setLineDash([6, 6]);
  gctx.lineWidth = 1;
  gctx.strokeRect(0.5, 0.5, DISPLAY_SIZE - 1, DISPLAY_SIZE - 1);
  gctx.beginPath();
  gctx.moveTo(DISPLAY_SIZE / 2, 0);
  gctx.lineTo(DISPLAY_SIZE / 2, DISPLAY_SIZE);
  gctx.moveTo(0, DISPLAY_SIZE / 2);
  gctx.lineTo(DISPLAY_SIZE, DISPLAY_SIZE / 2);
  gctx.stroke();
  gctx.setLineDash([]);
}

// ---------------------------------------------------------------------------
// Renders come straight from the kernel; the client never fabricates pixels.
// ---------------------------------------------------------------------------
async function refreshTile() {
  const img = new Image();
  img.onload = () => {
    tileImage = img;
    pctx.clearRect(0, 0, DISPLAY_SIZE, DISPLAY_SIZE);
    pctx.drawImage(img, 0, 0, DISPLAY_SIZE, DISPLAY_SIZE);
    drawSeamGuides();
  };
  img.src = "/api/tile.png?ts=" + Date.now();
}

async function refreshPreview() {
  const copies = parseInt(copiesEl.value, 10);
  const img = new Image();
  img.onload = () => {
    vctx.clearRect(0, 0, previewCanvas.width, previewCanvas.height);
    // The server already tiles the kernel render; draw it 1:1 so cross-tile
    // seams can be inspected directly.
    const side = previewCanvas.width;
    vctx.drawImage(img, 0, 0, side, side);
  };
  img.src = `/api/preview.png?copies=${copies}&ts=${Date.now()}`;
}

async function api(path, payload) {
  const resp = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload || {}),
  });
  const data = await resp.json();
  if (!data.ok) throw new Error(data.error || "kernel error");
  return data;
}

// ---------------------------------------------------------------------------
// Pointer strokes.  Raw tile-local points are buffered; the kernel unwraps
// the wrapped jumps and resamples by arc length, so no erroneous straight
// line can cross the unit when the pointer jumps between edges.
// ---------------------------------------------------------------------------
paintCanvas.addEventListener("pointerdown", (ev) => {
  ev.preventDefault();
  paintCanvas.setPointerCapture(ev.pointerId);
  drawing = true;
  strokePts = [eventToTile(ev)];
});

paintCanvas.addEventListener("pointermove", (ev) => {
  drawGhost(ev);
  if (!drawing) return;
  strokePts.push(eventToTile(ev));
});

paintCanvas.addEventListener("pointerleave", () => {
  if (!drawing) gctx.clearRect(0, 0, DISPLAY_SIZE, DISPLAY_SIZE);
});

paintCanvas.addEventListener("pointerup", async () => {
  if (!drawing) return;
  drawing = false;
  const pts = strokePts;
  strokePts = [];
  if (pts.length === 0) return;
  setStatus("内核正在按周期覆盖规则落笔画…", "busy");
  try {
    await api("/api/stroke", { points: pts, ...params() });
    await refreshTile();
    await refreshPreview();
    setStatus("笔画已按周期关系写入，接缝预览已更新。");
  } catch (err) {
    // Failure feedback back into the seamless-pattern window.
    setStatus("笔画被内核拒绝：" + err.message, "err");
  }
});

// ---------------------------------------------------------------------------
// Controls
// ---------------------------------------------------------------------------
[radiusEl, softEl, alphaEl, copiesEl].forEach((el) =>
  el.addEventListener("input", () => {
    refreshLabels();
  })
);

document.getElementById("undo").addEventListener("click", async () => {
  try {
    const r = await api("/api/undo", {});
    await refreshTile();
    await refreshPreview();
    setStatus(r.undone ? "已撤销最后一笔并按历史重绘。" : "没有可撤销的笔画。");
  } catch (err) {
    setStatus("撤销失败：" + err.message, "err");
  }
});

document.getElementById("clear").addEventListener("click", async () => {
  if (!confirm("清空当前纹理单元与全部笔画历史？")) return;
  try {
    await api("/api/clear", {});
    await refreshTile();
    await refreshPreview();
    setStatus("单元已清空，可重新沿边缘落笔。");
  } catch (err) {
    setStatus("清空失败：" + err.message, "err");
  }
});

document.getElementById("newTile").addEventListener("click", async () => {
  const w = parseInt(document.getElementById("newW").value, 10);
  const h = parseInt(document.getElementById("newH").value, 10);
  try {
    const r = await api("/api/new", { width: w, height: h });
    tileW = r.width;
    tileH = r.height;
    await refreshTile();
    await refreshPreview();
    setStatus(`已新建 ${tileW}×${tileH} 单元。`);
  } catch (err) {
    setStatus("新建单元失败：" + err.message, "err");
  }
});

document.getElementById("save").addEventListener("click", async () => {
  try {
    setStatus("正在保存原始单元与笔画历史…", "busy");
    const r = await api("/api/save", {});
    document.getElementById("savedInfo").textContent =
      Object.entries(r.paths).map(([k, v]) => `${k}: ${v}`).join("\n");
    setStatus("已保存：原始单元、笔画历史 JSON、3×3 周期预览。");
  } catch (err) {
    setStatus("保存失败：" + err.message, "err");
  }
});

copiesEl.addEventListener("change", refreshPreview);

async function init() {
  refreshLabels();
  try {
    const resp = await fetch("/api/state");
    const st = await resp.json();
    tileW = st.width;
    tileH = st.height;
    setStatus(`内核就绪（${tileW}×${tileH}，numpy 加速：${st.numpy ? "是" : "否"}）。`);
  } catch (err) {
    setStatus("无法连接内核：" + err.message, "err");
  }
  await refreshTile();
  await refreshPreview();
  drawSeamGuides();
}

init();
