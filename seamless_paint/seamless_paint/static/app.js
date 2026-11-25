"use strict";

const VIEW_PX = 512;

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
const angleEl = document.getElementById("angle");
const aspectEl = document.getElementById("aspect");

let tileW = 256;
let tileH = 256;
let lattice = { ax: 256, ay: 0, bx: 0, by: 256 };
let view = { x0: 0, y0: 0, x1: 256, y1: 256, scale_x: 2, scale_y: 2 };
let drawing = false;
let strokePts = [];

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
    angle_deg: parseFloat(angleEl.value),
    aspect: parseFloat(aspectEl.value),
  };
}

function refreshLabels() {
  document.getElementById("radiusVal").textContent = radiusEl.value + " px";
  document.getElementById("softVal").textContent = parseFloat(softEl.value).toFixed(2);
  document.getElementById("alphaVal").textContent = parseFloat(alphaEl.value).toFixed(2);
  document.getElementById("angleVal").textContent = angleEl.value + "°";
  document.getElementById("aspectVal").textContent = parseFloat(aspectEl.value).toFixed(2);
  document.getElementById("copiesVal").textContent = copiesEl.value + " × " + copiesEl.value;
}

// ---------------------------------------------------------------------------
// Coordinate transforms.  Screen -> world (view rect) -> lattice cell.
// The kernel performs the same B / B^{-1} transforms authoritatively; these
// client copies only position the cursor/grid overlays.
// ---------------------------------------------------------------------------
function eventToWorld(ev) {
  const rect = paintCanvas.getBoundingClientRect();
  const fx = (ev.clientX - rect.left) / rect.width;
  const fy = (ev.clientY - rect.top) / rect.height;
  return [
    view.x0 + fx * (view.x1 - view.x0),
    view.y0 + fy * (view.y1 - view.y0),
  ];
}

function worldToScreen(x, y) {
  return [
    ((x - view.x0) / (view.x1 - view.x0)) * VIEW_PX,
    ((y - view.y0) / (view.y1 - view.y0)) * VIEW_PX,
  ];
}

function worldToCell(x, y) {
  const d = lattice.ax * lattice.by - lattice.ay * lattice.bx;
  return [
    (lattice.by * x - lattice.bx * y) / d,
    (-lattice.ay * x + lattice.ax * y) / d,
  ];
}

function cellToWorld(s, t) {
  return [lattice.ax * s + lattice.bx * t,
          lattice.ay * s + lattice.by * t];
}

// Conservative support half-extents in cell coords from the outer radius.
function cellExtents(rOuter) {
  const d = lattice.ax * lattice.by - lattice.ay * lattice.bx;
  const m00 = lattice.by / d, m01 = -lattice.bx / d;
  const m10 = -lattice.ay / d, m11 = lattice.ax / d;
  return [
    rOuter * Math.hypot(m00, m01),
    rOuter * Math.hypot(m10, m11),
  ];
}

// Unique lattice identities of brush copies visible in the world viewport.
function visibleCopyIds(wx, wy, p) {
  const rx = p.radius;
  const ry = p.radius * p.aspect;
  const rOuter = Math.max(rx, ry);
  const [es, et] = cellExtents(rOuter);
  const [sc, tc] = worldToCell(wx, wy);
  const corners = [
    worldToCell(view.x0, view.y0), worldToCell(view.x1, view.y0),
    worldToCell(view.x0, view.y1), worldToCell(view.x1, view.y1),
  ];
  const sMin = Math.min(...corners.map(q => q[0])) - es;
  const sMax = Math.max(...corners.map(q => q[0])) + es;
  const tMin = Math.min(...corners.map(q => q[1])) - et;
  const tMax = Math.max(...corners.map(q => q[1])) + et;
  const iLo = Math.ceil(sMin - sc), iHi = Math.floor(sMax - sc);
  const jLo = Math.ceil(tMin - tc), jHi = Math.floor(tMax - tc);
  const ids = [];
  const seen = new Set();
  for (let j = jLo; j <= jHi; j++) {
    for (let i = iLo; i <= iHi; i++) {
      const key = i + "," + j;
      if (seen.has(key)) continue;
      seen.add(key);
      ids.push([i, j]);
    }
  }
  return ids;
}

// ---------------------------------------------------------------------------
// Lattice guide grid + periodic brush ghost (one outline per cell identity).
// ---------------------------------------------------------------------------
function drawLatticeGrid() {
  const corners = [
    worldToCell(view.x0, view.y0), worldToCell(view.x1, view.y0),
    worldToCell(view.x0, view.y1), worldToCell(view.x1, view.y1),
  ];
  const sLo = Math.floor(Math.min(...corners.map(q => q[0]))) - 1;
  const sHi = Math.ceil(Math.max(...corners.map(q => q[0]))) + 1;
  const tLo = Math.floor(Math.min(...corners.map(q => q[1]))) - 1;
  const tHi = Math.ceil(Math.max(...corners.map(q => q[1]))) + 1;

  gctx.lineWidth = 1;
  gctx.strokeStyle = "rgba(70, 90, 130, 0.55)";
  for (let j = tLo; j <= tHi; j++) {
    for (let i = sLo; i <= sHi; i++) {
      const pts = [[i, j], [i + 1, j], [i + 1, j + 1], [i, j + 1]]
        .map(([s, t]) => worldToScreen(...cellToWorld(s, t)));
      const base = i === 0 && j === 0;
      gctx.beginPath();
      gctx.moveTo(pts[0][0], pts[0][1]);
      for (let k = 1; k < 4; k++) gctx.lineTo(pts[k][0], pts[k][1]);
      gctx.closePath();
      gctx.strokeStyle = base ? "rgba(255, 214, 92, 0.95)" : "rgba(120, 150, 210, 0.4)";
      gctx.lineWidth = base ? 2 : 1;
      if (base) gctx.setLineDash([7, 5]);
      gctx.stroke();
      gctx.setLineDash([]);
    }
  }
}

function drawBrushShapeAtScreen(p, sx, sy) {
  const rx = p.radius;
  const ry = p.radius * p.aspect;
  const theta = (p.angle_deg || 0) * Math.PI / 180;
  gctx.save();
  gctx.translate(sx, sy);
  // World->screen view scale (possibly non-uniform for non-square ortho),
  // then world rotation and local ellipse radii.
  gctx.scale(view.scale_x, view.scale_y);
  gctx.rotate(theta);
  gctx.scale(rx, ry);
  gctx.beginPath();
  gctx.arc(0, 0, 1, 0, Math.PI * 2);
  gctx.fillStyle = "rgba(255,255,255,0.14)";
  gctx.fill();
  gctx.lineWidth = 1.2 / Math.max(view.scale_x, view.scale_y);
  gctx.strokeStyle = "rgba(20,20,30,0.85)";
  gctx.stroke();
  gctx.restore();
}

function drawOverlay(ev) {
  gctx.clearRect(0, 0, VIEW_PX, VIEW_PX);
  drawLatticeGrid();
  if (!ev) return;
  const [wx, wy] = eventToWorld(ev);
  const p = params();
  const ids = visibleCopyIds(wx, wy, p);
  gctx.lineWidth = 1;
  for (const [i, j] of ids) {
    const cx = wx + i * lattice.ax + j * lattice.bx;
    const cy = wy + i * lattice.ay + j * lattice.by;
    const [sx, sy] = worldToScreen(cx, cy);
    if (sx < -p.radius * 2 || sx > VIEW_PX + p.radius * 2 ||
        sy < -p.radius * 2 || sy > VIEW_PX + p.radius * 2) continue;
    drawBrushShapeAtScreen(p, sx, sy);
  }
}

// ---------------------------------------------------------------------------
// Kernel-rendered images only; the client never fabricates pattern pixels.
// ---------------------------------------------------------------------------
function loadImage(url) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = reject;
    img.src = url;
  });
}

async function refreshPaint() {
  try {
    const img = await loadImage("/api/world.png?w=512&h=512&ts=" + Date.now());
    pctx.clearRect(0, 0, VIEW_PX, VIEW_PX);
    pctx.drawImage(img, 0, 0, VIEW_PX, VIEW_PX);
    gctx.clearRect(0, 0, VIEW_PX, VIEW_PX);
    drawLatticeGrid();
  } catch (err) {
    setStatus("世界视图刷新失败：" + err.message, "err");
  }
}

async function refreshPreview() {
  const copies = parseInt(copiesEl.value, 10);
  try {
    const img = await loadImage(
      `/api/lattice_preview.png?copies=${copies}&max=420&ts=${Date.now()}`);
    vctx.clearRect(0, 0, previewCanvas.width, previewCanvas.height);
    vctx.fillStyle = "#ffffff";
    vctx.fillRect(0, 0, previewCanvas.width, previewCanvas.height);
    const side = previewCanvas.width;
    const scale = Math.min(side / img.width, side / img.height);
    const w = img.width * scale;
    const h = img.height * scale;
    vctx.imageSmoothingEnabled = false;
    vctx.drawImage(img, (side - w) / 2, (side - h) / 2, w, h);
  } catch (err) {
    setStatus("平铺预览刷新失败：" + err.message, "err");
  }
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

function applyState(st) {
  tileW = st.width;
  tileH = st.height;
  lattice = {
    ax: st.lattice.a[0], ay: st.lattice.a[1],
    bx: st.lattice.b[0], by: st.lattice.b[1],
  };
  view = st.view;
  document.getElementById("ax").value = Math.round(lattice.ax * 100) / 100;
  document.getElementById("ay").value = Math.round(lattice.ay * 100) / 100;
  document.getElementById("bx").value = Math.round(lattice.bx * 100) / 100;
  document.getElementById("by").value = Math.round(lattice.by * 100) / 100;
  document.getElementById("newW").value = tileW;
  document.getElementById("newH").value = tileH;
  document.getElementById("basisInfo").textContent =
    `a=(${lattice.ax}, ${lattice.ay})  b=(${lattice.bx}, ${lattice.by})\n` +
    `格元面积=${st.lattice.cell_area.toFixed(1)} px²  ` +
    `a→b 夹角=${st.lattice.angle_deg.toFixed(2)}°  ` +
    `${st.lattice.orthogonal ? "正交" : "非正交斜格"}`;
}

// ---------------------------------------------------------------------------
// Pointer strokes (world coordinates; continuous across visible cell edges).
// ---------------------------------------------------------------------------
paintCanvas.addEventListener("pointerdown", (ev) => {
  ev.preventDefault();
  paintCanvas.setPointerCapture(ev.pointerId);
  drawing = true;
  strokePts = [eventToWorld(ev)];
});

paintCanvas.addEventListener("pointermove", (ev) => {
  drawOverlay(ev);
  if (!drawing) return;
  strokePts.push(eventToWorld(ev));
});

paintCanvas.addEventListener("pointerleave", () => {
  if (!drawing) drawOverlay(null);
});

paintCanvas.addEventListener("pointerup", async () => {
  if (!drawing) return;
  drawing = false;
  const pts = strokePts;
  strokePts = [];
  if (pts.length === 0) return;
  setStatus("内核正在按晶格周期覆盖规则落笔…", "busy");
  try {
    await api("/api/stroke", { points: pts, ...params() });
    await refreshPaint();
    await refreshPreview();
    setStatus("笔画已写入晶格单元，真实斜格平铺与跨边接缝已更新。");
  } catch (err) {
    setStatus("笔画被内核拒绝：" + err.message, "err");
  }
});

// ---------------------------------------------------------------------------
// Standard controls
// ---------------------------------------------------------------------------
[radiusEl, softEl, alphaEl, copiesEl, angleEl, aspectEl].forEach((el) =>
  el.addEventListener("input", refreshLabels));

document.getElementById("undo").addEventListener("click", async () => {
  try {
    const r = await api("/api/undo", {});
    await refreshPaint();
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
    await refreshPaint();
    await refreshPreview();
    setStatus("单元已清空，可重新沿格边/格点落笔。");
  } catch (err) {
    setStatus("清空失败：" + err.message, "err");
  }
});

document.getElementById("newTile").addEventListener("click", async () => {
  const w = parseInt(document.getElementById("newW").value, 10);
  const h = parseInt(document.getElementById("newH").value, 10);
  try {
    const st = await api("/api/new", { width: w, height: h });
    applyState(st);
    await refreshPaint();
    await refreshPreview();
    setStatus(`已新建 ${tileW}×${tileH} 正交单元。`);
  } catch (err) {
    setStatus("新建单元失败：" + err.message, "err");
  }
});

document.getElementById("save").addEventListener("click", async () => {
  try {
    setStatus("正在保存单元、含重复基向量的历史与平铺预览…", "busy");
    const r = await api("/api/save", {});
    document.getElementById("savedInfo").textContent =
      Object.entries(r.paths).map(([k, v]) => `${k}: ${v}`).join("\n");
    setStatus("已保存：单元 PNG、记录重复基的历史 JSON、坐标预览与真实斜格平铺。");
  } catch (err) {
    setStatus("保存失败：" + err.message, "err");
  }
});

copiesEl.addEventListener("change", refreshPreview);

// ---------------------------------------------------------------------------
// Lattice basis controls
// ---------------------------------------------------------------------------
function readBasisInputs() {
  return {
    ax: parseFloat(document.getElementById("ax").value),
    ay: parseFloat(document.getElementById("ay").value),
    bx: parseFloat(document.getElementById("bx").value),
    by: parseFloat(document.getElementById("by").value),
  };
}

function selectedMode() {
  return document.querySelector('input[name="bmode"]:checked').value;
}

const PRESETS = {
  ortho: () => ({ ax: tileW, ay: 0, bx: 0, by: tileH }),
  diamond60: () => ({ ax: tileW, ay: 0, bx: tileW / 2, by: Math.round(tileW * 0.8660254) }),
  halfdrop: () => ({ ax: tileW, ay: 0, bx: tileW / 2, by: tileH }),
  acute15: () => {
    const L = Math.max(tileW, tileH);
    return { ax: tileW, ay: 0, bx: Math.round(L * Math.cos(Math.PI / 12) * 10) / 10,
             by: Math.round(L * Math.sin(Math.PI / 12) * 10) / 10 };
  },
};

document.getElementById("basisPreset").addEventListener("change", async (ev) => {
  const b = PRESETS[ev.target.value]();
  document.getElementById("ax").value = b.ax;
  document.getElementById("ay").value = b.ay;
  document.getElementById("bx").value = b.bx;
  document.getElementById("by").value = b.by;
  await applyBasis();
});

async function applyBasis() {
  const b = readBasisInputs();
  const mode = selectedMode();
  try {
    setStatus("正在更换重复基向量…", "busy");
    const st = await api("/api/lattice", { ...b, mode });
    applyState(st);
    await refreshPaint();
    await refreshPreview();
    setStatus(mode === "deform"
      ? "已变形现有内容：参数格画作保留，世界笔迹与笔尖度量随 T=B′B⁻¹ 线性形变。"
      : "已按新基重新包裹：物理笔迹与笔尖保持不变，周期归属重新折叠。");
  } catch (err) {
    setStatus("基向量被内核拒绝：" + err.message, "err");
  }
}

document.getElementById("applyBasis").addEventListener("click", applyBasis);

document.getElementById("compareBasis").addEventListener("click", async () => {
  const b = readBasisInputs();
  const wrap = document.getElementById("compareWrap");
  wrap.hidden = false;
  try {
    setStatus("正在生成两种换基影响的非写入对比预览…", "busy");
    for (const mode of ["rewrap", "deform"]) {
      const r = await api("/api/basis/preview", { ...b, mode });
      document.getElementById(mode === "rewrap" ? "cmpRewrap" : "cmpDeform").src =
        r.world_png;
    }
    setStatus("对比图已生成：左为保留物理笔迹重新包裹，右为变形现有内容。会话未被预览修改。");
  } catch (err) {
    setStatus("对比预览失败（可能是退化晶格）：" + err.message, "err");
  }
});

// ---------------------------------------------------------------------------
async function init() {
  refreshLabels();
  try {
    const resp = await fetch("/api/state");
    const st = await resp.json();
    applyState(st);
    setStatus(`内核就绪（${tileW}×${tileH}，numpy 加速：${st.numpy ? "是" : "否（纯 Python）"}）。默认正交模式与初版兼容。`);
  } catch (err) {
    setStatus("无法连接内核：" + err.message, "err");
  }
  await refreshPaint();
  await refreshPreview();
}

init();
