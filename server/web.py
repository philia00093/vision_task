"""网页监控台的单文件页面（HTML + JS，无任何外部依赖）。

页面完全靠调 API 工作：GET /frame.jpg 拿画面、GET /state 拿框、POST /detect 自己送帧。"""

from __future__ import annotations


PAGE_HTML = """<!doctype html>
<html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>识别服务 · 监控台</title>
<style>
  :root { color-scheme: dark; }
  body { margin:0; background:#14171c; color:#dfe4ea;
         font:14px/1.5 system-ui,"Microsoft YaHei",sans-serif; }
  header { display:flex; align-items:center; gap:14px; padding:10px 16px;
           background:#1c2128; border-bottom:1px solid #2b3138; flex-wrap:wrap; }
  h1 { font-size:16px; margin:0 10px 0 0; font-weight:600; }
  .tag { padding:2px 8px; border-radius:10px; background:#2b3138; font-size:12px; }
  .ok { background:#1f6f3f; } .bad { background:#7a2b2b; }
  main { display:flex; gap:14px; padding:14px; align-items:flex-start; flex-wrap:wrap; }
  canvas { background:#000; border:1px solid #2b3138; border-radius:6px;
           max-width:100%; height:auto; }
  aside { min-width:260px; max-width:340px; background:#1c2128; border:1px solid #2b3138;
          border-radius:6px; padding:12px 14px; }
  aside h2 { font-size:13px; margin:0 0 8px; color:#8b949e; font-weight:600;
             text-transform:uppercase; letter-spacing:.04em; }
  label { display:flex; justify-content:space-between; align-items:center; gap:8px;
          margin:6px 0; }
  select, input[type=number] { background:#0d1117; color:#dfe4ea; border:1px solid #30363d;
          border-radius:4px; padding:3px 6px; width:110px; }
  code { background:#0d1117; padding:1px 5px; border-radius:4px; color:#a5d6ff; }
  ul { padding-left:18px; margin:6px 0; }
  .muted { color:#8b949e; }
</style></head>
<body>
<header>
  <h1>识别服务 · 监控台</h1>
  <span id="link" class="tag">连接中…</span>
  <span class="tag" id="srcTag">画面源：—</span>
  <span class="tag" id="stat">—</span>
</header>
<main>
  <canvas id="view" width="1280" height="960"></canvas>
  <aside>
    <h2>显示</h2>
    <label>画面源
      <select id="mode">
        <option value="remote">树莓派画面（服务端最新帧）</option>
        <option value="local">本机摄像头（浏览器直采）</option>
      </select>
    </label>
    <label>拉流频率 <input type="number" id="hz" value="10" min="1" max="30"> 次/秒</label>
    <label>框线粗细 <input type="number" id="lw" value="2" min="1" max="6"></label>
    <label>显示标签 <input type="checkbox" id="showLabel" checked></label>
    <hr style="border-color:#2b3138">
    <h2>接口（这就是网页在调的 API）</h2>
    <ul class="muted">
      <li><code>GET /frame.jpg</code> 最近一帧画面</li>
      <li><code>GET /state</code> 最近一次检测结果</li>
      <li><code>POST /detect</code> 发一帧 JPEG，回框</li>
      <li><code>GET /health</code> 连通性与类别名</li>
    </ul>
    <p class="muted" style="margin:8px 0 0">树莓派只要照常 <code>--infer-backend http</code> 跑，
       它 POST 上来的帧会自动变成这个页面的画面；本机摄像头模式则直接由浏览器调
       <code>POST /detect</code>。</p>
  </aside>
</main>
<video id="cam" playsinline style="display:none"></video>
<script>
const canvas = document.getElementById('view');
const ctx = canvas.getContext('2d');
const video = document.getElementById('cam');
const modeSel = document.getElementById('mode');
const hzInput = document.getElementById('hz');
const lwInput = document.getElementById('lw');
const labelToggle = document.getElementById('showLabel');
const linkTag = document.getElementById('link');
const srcTag = document.getElementById('srcTag');
const statTag = document.getElementById('stat');
const NAMES = [];        // 由 /health 填

let boxes = [], imgSize = [0, 0], serverMs = 0, roundTrip = 0, frames = 0, fpsTs = 0, fps = 0;

function draw() {
  const [iw, ih] = imgSize[0] ? imgSize : [canvas.width, canvas.height];
  if (canvas.width !== iw || canvas.height !== ih) { canvas.width = iw; canvas.height = ih; }
  const sx = canvas.width / iw, sy = canvas.height / ih;
  ctx.lineWidth = Number(lwInput.value) || 2;
  ctx.font = Math.max(12, Math.round(canvas.height / 55)) + 'px system-ui';
  for (const b of boxes) {
    const [x1, y1, x2, y2, cls, conf] = b;
    ctx.strokeStyle = '#32c48d';
    ctx.strokeRect(x1 * sx, y1 * sy, (x2 - x1) * sx, (y2 - y1) * sy);
    if (labelToggle.checked) {
      const text = (NAMES[cls] || cls) + ' ' + Number(conf).toFixed(2);
      const w = ctx.measureText(text).width + 8;
      const h = parseInt(ctx.font) + 6;
      const ty = y1 * sy - h >= 0 ? y1 * sy - h : y2 * sy;
      ctx.fillStyle = 'rgba(0,0,0,.65)';
      ctx.fillRect(x1 * sx, ty, w, h);
      ctx.fillStyle = '#32c48d';
      ctx.fillText(text, x1 * sx + 4, ty + h - 5);
    }
  }
  frames++;
  const now = performance.now();
  if (now - fpsTs > 1000) { fps = frames * 1000 / (now - fpsTs); frames = 0; fpsTs = now; }
  statTag.textContent = '框 ' + boxes.length + ' | 服务端 ' + serverMs.toFixed(0) + 'ms'
      + ' | 往返 ' + roundTrip.toFixed(0) + 'ms | ' + fps.toFixed(1) + ' fps';
}

async function pullRemote() {
  // 网页这边就是"调 API"：/state 拿框，/frame.jpg 拿画面
  try {
    const t0 = performance.now();
    const state = await (await fetch('/state', {cache: 'no-store'})).json();
    const blob = await (await fetch('/frame.jpg?t=' + Date.now(), {cache: 'no-store'})).blob();
    if (blob.size > 0) {
      const bitmap = await createImageBitmap(blob);
      imgSize = [bitmap.width, bitmap.height];
      ctx.drawImage(bitmap, 0, 0, canvas.width, canvas.height);
      bitmap.close();
    }
    boxes = state.boxes || []; serverMs = state.ms || 0; imgSize = state.size || imgSize;
    roundTrip = performance.now() - t0;
    linkTag.textContent = '已连接'; linkTag.className = 'tag ok';
    const age = (state.age == null) ? '—' : state.age.toFixed(1) + 's 前';
    srcTag.textContent = '画面源：树莓派 ' + imgSize[0] + '×' + imgSize[1]
        + '（第 ' + (state.count || 0) + ' 帧，上一帧 ' + age + '）';
    draw();
  } catch (e) {
    linkTag.textContent = '断开：' + e.message; linkTag.className = 'tag bad';
  }
}

async function pullLocal() {
  // 本机摄像头：浏览器直接 POST /detect（绕开树莓派，用来单独测服务）
  try {
    const bitmap = await createImageBitmap(video);
    imgSize = [bitmap.width, bitmap.height];
    ctx.drawImage(bitmap, 0, 0, canvas.width, canvas.height);
    const c = document.createElement('canvas');
    c.width = bitmap.width; c.height = bitmap.height;
    c.getContext('2d').drawImage(bitmap, 0, 0);
    bitmap.close();
    const blob = await new Promise(r => c.toBlob(r, 'image/jpeg', 0.8));
    const t0 = performance.now();
    const res = await (await fetch('/detect', {method: 'POST', body: blob,
        headers: {'Content-Type': 'image/jpeg'}})).json();
    roundTrip = performance.now() - t0;
    boxes = res.boxes || []; serverMs = res.ms || 0; imgSize = res.size || imgSize;
    linkTag.textContent = '已连接'; linkTag.className = 'tag ok';
    srcTag.textContent = '画面源：本机摄像头（' + imgSize[0] + '×' + imgSize[1] + '）';
    draw();
  } catch (e) {
    linkTag.textContent = '出错：' + e.message; linkTag.className = 'tag bad';
  }
}

async function startLocalCamera(enable) {
  if (enable && !video.srcObject) {
    const stream = await navigator.mediaDevices.getUserMedia({video: {width: 1280, height: 720}});
    video.srcObject = stream;
    await video.play();
  } else if (!enable && video.srcObject) {
    video.srcObject.getTracks().forEach(t => t.stop());
    video.srcObject = null;
  }
}

let timer = null;
function schedule() {
  clearInterval(timer);
  const period = 1000 / Math.max(1, Number(hzInput.value) || 10);
  timer = setInterval(modeSel.value === 'remote' ? pullRemote : pullLocal, period);
}
async function applyMode() {
  await startLocalCamera(modeSel.value === 'local');
  boxes = []; schedule();
}
modeSel.addEventListener('change', applyMode);
hzInput.addEventListener('change', schedule);
lwInput.addEventListener('input', draw);
labelToggle.addEventListener('change', draw);

(async () => {
  try {
    const health = await (await fetch('/health')).json();
    NAMES.splice(0, NAMES.length, ...(health.names || []));
    document.querySelector('h1').textContent =
        '识别服务 · 监控台  ' + health.model + '  imgsz=' + health.imgsz + '  device=' + health.device;
  } catch (e) { /* /health 失败也没关系，后面拉流会报错 */ }
  applyMode();
})();
</script></body></html>
"""
