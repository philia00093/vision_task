#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""延时 + 视频流监控台（默认 2000 端口）—— 从识别服务那边捞帧和统计，不参与推理。

为什么要单独一个端口
--------------------
`server/run_server.py` 的 8080 是**给树莓派用的**：它按请求-响应工作，页面只是顺便看看。
这里 2000 是**给这台电脑自己看的**：把上游（8080）的最新帧和识别结果拉过来，
持续统计「推理耗时 / 帧龄 / 收到帧率 / 拉取往返」，画成实时曲线。

它**不碰模型、不占 GPU**，所以开着它不会影响识别速度。

用法
----
    python server/monitor.py                              # 监听 0.0.0.0:2000，上游 127.0.0.1:8080
    python server/monitor.py --port 2000 --upstream http://127.0.0.1:8080
    python server/monitor.py --upstream http://10.62.143.158:8080 --poll-hz 20

浏览器打开 http://<本机IP>:2000/ ，同一局域网的手机也能看。

页面上的四个数
--------------
* **推理**：上游报的 YOLO 推理耗时（`/state` 里的 `ms`）
* **帧龄**：上游最近一帧是多久以前收到的（树莓派那边是不是在发）
* **帧率**：按上游 `count` 增长算出的**实际收到帧率**（不是你请求的帧率）
* **拉取**：本监控台 GET 一次 `/frame.jpg` + `/state` 的往返（本机→本机，应该 <5ms）

历史曲线保留最近 240 个采样点（约 2 分钟 @2Hz）。
"""

from __future__ import annotations

import argparse
import json
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# ---------------------------------------------------------------- 采样
class Sampler:
    """后台按固定频率拉上游 /state 和 /frame.jpg，保存最近一帧和一段历史。"""

    def __init__(self, upstream: str, poll_hz: float, history: int = 240):
        self.upstream = upstream.rstrip("/")
        self.period = 1.0 / max(poll_hz, 0.5)
        self.lock = threading.Lock()
        self.jpeg: bytes | None = None
        self.state: dict = {}
        self.fetch_ms = 0.0
        self.upstream_ok = False
        self.error = ""
        self.last_count: int | None = None
        self.last_count_time = 0.0
        self.fps = 0.0
        self.history: list[dict] = []
        self.history_max = history
        self.samples = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="sampler", daemon=True)
        self._thread.start()

    def _get(self, path: str, timeout: float = 3.0) -> bytes:
        request = urllib.request.Request(self.upstream + path,
                                         headers={"Cache-Control": "no-store"})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read()

    def _loop(self) -> None:
        while not self._stop.is_set():
            started = time.time()
            try:
                state = json.loads(self._get("/state").decode("utf-8"))
                jpeg = self._get("/frame.jpg")
                elapsed = (time.time() - started) * 1000.0
                with self.lock:
                    self.state = state
                    self.jpeg = jpeg if jpeg else self.jpeg
                    self.fetch_ms = elapsed
                    self.upstream_ok = True
                    self.error = ""
                    self._update_fps(state)
                    self._push_history(state)
            except Exception as error:
                with self.lock:
                    self.upstream_ok = False
                    self.error = f"{type(error).__name__}: {error}"
            self._stop.wait(self.period)

    def _update_fps(self, state: dict) -> None:
        count = state.get("count")
        now = time.time()
        if isinstance(count, int):
            if self.last_count is not None and now > self.last_count_time:
                delta = count - self.last_count
                if delta >= 0:
                    rate = delta / (now - self.last_count_time)
                    # 指数平滑，避免一跳一跳
                    self.fps = rate if self.fps == 0.0 else self.fps * 0.6 + rate * 0.4
            self.last_count, self.last_count_time = count, now

    def _push_history(self, state: dict) -> None:
        self.samples += 1
        self.history.append({
            "t": time.time(),
            "infer": float(state.get("ms") or 0.0),
            "age": float(state.get("age") or 0.0),
            "fps": self.fps,
            "fetch": self.fetch_ms,
        })
        if len(self.history) > self.history_max:
            del self.history[: len(self.history) - self.history_max]

    def snapshot(self) -> dict:
        with self.lock:
            return {"state": dict(self.state), "fetch_ms": self.fetch_ms,
                    "ok": self.upstream_ok, "error": self.error, "fps": self.fps,
                    "history": list(self.history), "samples": self.samples,
                    "jpeg": self.jpeg}

    def frame(self):
        with self.lock:
            return self.jpeg

    def forward(self, path: str, body: bytes | None = None,
                content_type: str = "application/json") -> tuple[int, bytes]:
        """把请求转发给上游（保存图片用）。返回 (状态码, 响应体)。"""
        request = urllib.request.Request(self.upstream + path, data=body, method="POST",
                                        headers={"Content-Type": content_type})
        try:
            with urllib.request.urlopen(request, timeout=5) as response:
                return response.status, response.read()
        except urllib.error.HTTPError as error:
            return error.code, error.read()
        except Exception as error:
            return 502, json.dumps({"ok": False, "error": str(error)}).encode("utf-8")

    def close(self) -> None:
        self._stop.set()


# ---------------------------------------------------------------- 页面
PAGE = r"""<!doctype html>
<html lang="zh"><head><meta charset="utf-8">
<title>延时监控 · vision_task</title>
<style>
 :root{color-scheme:dark}
 body{margin:0;background:#0f1216;color:#dfe4ea;font:14px/1.5 system-ui,"Microsoft YaHei",sans-serif}
 header{display:flex;gap:14px;align-items:center;padding:8px 14px;background:#171b21;
        border-bottom:1px solid #262c34;flex-wrap:wrap}
 h1{font-size:15px;margin:0 8px 0 0}
 .tag{padding:2px 8px;border-radius:10px;background:#262c34;font-size:12px}
 .ok{background:#1f6f3f}.bad{background:#7a2b2b}
 main{display:flex;gap:14px;padding:12px;align-items:flex-start;flex-wrap:wrap}
 #wrap{position:relative;flex:0 0 auto;background:#000;border:1px solid #262c34;border-radius:6px;overflow:hidden}
 canvas{display:block;max-width:min(72vw,1100px);height:auto}
 aside{flex:1 1 320px;min-width:300px;max-width:420px;background:#171b21;border:1px solid #262c34;
       border-radius:6px;padding:12px 14px}
 .row{display:flex;justify-content:space-between;margin:6px 0}
 .big{font-size:22px;font-variant-numeric:tabular-nums}
 .muted{color:#8b949e}
 .bar{height:6px;background:#262c34;border-radius:3px;overflow:hidden;margin-top:4px}
 .bar>i{display:block;height:100%;background:#3fb950;width:0%}
 h2{font-size:12px;color:#8b949e;margin:16px 0 6px;letter-spacing:.05em;text-transform:uppercase}
</style></head><body>
<header>
  <h1>延时监控</h1>
  <span class="tag" id="link">连接中…</span>
  <span class="tag muted" id="src">—</span>
  <span class="tag muted" id="peak">—</span>
</header>
<main>
  <div id="wrap"><canvas id="view" width="1280" height="960"></canvas></div>
  <aside>
    <h2>实时</h2>
    <div class="row"><span>推理耗时</span><span class="big" id="infer">–</span></div>
    <div class="row"><span>帧龄（上游多久前收到帧）</span><span class="big" id="age">–</span></div>
    <div class="row"><span>实际收到帧率</span><span class="big" id="fps">–</span></div>
    <div class="row"><span>拉取往返（本机）</span><span class="big" id="fetch">–</span></div>
    <div class="row muted"><span>上游帧序号</span><span id="count">–</span></div>
    <div class="row muted"><span>检出目标</span><span id="det">–</span></div>

    <h2>保存</h2>
    <div class="row">
      <input id="label" type="text" placeholder="标签（可空）"
             style="flex:1;background:#0d1117;color:#dfe4ea;border:1px solid #30363d;
                    border-radius:4px;padding:4px 6px">
      <button id="save" style="margin-left:8px;background:#238636;color:#fff;border:0;
              border-radius:4px;padding:5px 12px;cursor:pointer">保存当前帧</button>
    </div>
    <p class="muted" id="saveMsg" style="margin:6px 0 0"></p>

    <h2>最近 2 分钟</h2>
    <div class="row"><span class="muted">推理</span><span class="muted" id="inferStat">–</span></div>
    <div class="bar"><i id="inferBar" style="background:#3fb950"></i></div>
    <div class="row"><span class="muted">帧龄</span><span class="muted" id="ageStat">–</span></div>
    <div class="bar"><i id="ageBar" style="background:#d29922"></i></div>
    <div class="row"><span class="muted">帧率</span><span class="muted" id="fpsStat">–</span></div>
    <div class="bar"><i id="fpsBar" style="background:#58a6ff"></i></div>
    <canvas id="chart" width="380" height="90" style="width:100%;margin-top:10px"></canvas>
    <p class="muted" id="note" style="margin:8px 0 0"></p>
  </aside>
</main>
<script>
const cv = document.getElementById('view'), ctx = cv.getContext('2d');
const chart = document.getElementById('chart'), cctx = chart.getContext('2d');
const el = id => document.getElementById(id);
let imgSize = [0,0];

function drawChart(hist){
  const w = chart.width, h = chart.height;
  cctx.clearRect(0,0,w,h);
  cctx.strokeStyle = '#262c34'; cctx.lineWidth = 1;
  cctx.beginPath(); cctx.moveTo(0,h-0.5); cctx.lineTo(w,h-0.5); cctx.stroke();
  if (hist.length < 2) return;
  const series = [
    {key:'infer', color:'#3fb950', scale: Math.max(...hist.map(d=>d.infer), 1)},
    {key:'age',   color:'#d29922', scale: Math.max(...hist.map(d=>d.age), 0.5)},
    {key:'fps',   color:'#58a6ff', scale: Math.max(...hist.map(d=>d.fps), 1)},
  ];
  for (const s of series){
    cctx.strokeStyle = s.color; cctx.beginPath();
    hist.forEach((d,i)=>{
      const x = i/(hist.length-1)*w;
      const y = h - Math.min(d[s.key]/s.scale, 1)*(h-6) - 3;
      i ? cctx.lineTo(x,y) : cctx.moveTo(x,y);
    });
    cctx.stroke();
  }
}

function fmt(v, unit){ return (v===null||v===undefined||isNaN(v)) ? '–' : v.toFixed(1)+unit; }
function stat(arr){ 
  if(!arr.length) return {avg:0,max:0};
  return {avg: arr.reduce((a,b)=>a+b,0)/arr.length, max: Math.max(...arr)};
}

async function tick(){
  try {
    const t0 = performance.now();
    const stats = await (await fetch('/stats', {cache:'no-store'})).json();
    const rtt = performance.now() - t0;
    const st = stats.state || {};
    if (st.size && st.size[0]) imgSize = st.size;

    // 画面
    const blob = await (await fetch('/frame.jpg?t=' + Date.now(), {cache:'no-store'})).blob();
    if (blob.size > 0){
      const bmp = await createImageBitmap(blob);
      if (cv.width !== bmp.width || cv.height !== bmp.height){
        cv.width = bmp.width; cv.height = bmp.height;
      }
      ctx.drawImage(bmp, 0, 0);
      const sx = cv.width / bmp.width, sy = cv.height / bmp.height;
      ctx.lineWidth = 2; ctx.font = '16px system-ui';
      for (const b of (st.boxes||[])){
        const [x1,y1,x2,y2,cls,conf] = b;
        ctx.strokeStyle = '#3fb950';
        ctx.strokeRect(x1*sx, y1*sy, (x2-x1)*sx, (y2-y1)*sy);
        const label = '(' + cls + ') ' + Number(conf).toFixed(2);
        const w = ctx.measureText(label).width + 8;
        ctx.fillStyle='rgba(0,0,0,.6)';
        ctx.fillRect(x1*sx, y1*sy-20, w, 20);
        ctx.fillStyle='#3fb950';
        ctx.fillText(label, x1*sx+4, y1*sy-5);
      }
      bmp.close();
    }

    // 数字
    el('infer').textContent = fmt(st.ms, ' ms');
    el('age').textContent   = fmt(st.age, ' s');
    el('fps').textContent   = fmt(stats.fps, ' fps');
    el('fetch').textContent = fmt(stats.fetch_ms, ' ms');
    el('count').textContent = (st.count ?? '–') + '（浏览器往返 ' + rtt.toFixed(1) + 'ms）';
    const dets = (st.boxes||[]).length;
    el('det').textContent = dets + ' 个';

    const hist = stats.history || [];
    const inferS = stat(hist.map(d=>d.infer));
    const ageS   = stat(hist.map(d=>d.age));
    const fpsS   = stat(hist.map(d=>d.fps));
    el('inferStat').textContent = '均 ' + inferS.avg.toFixed(1) + ' / 峰 ' + inferS.max.toFixed(1) + ' ms';
    el('ageStat').textContent   = '均 ' + ageS.avg.toFixed(2) + ' / 峰 ' + ageS.max.toFixed(2) + ' s';
    el('fpsStat').textContent   = '均 ' + fpsS.avg.toFixed(1) + ' / 峰 ' + fpsS.max.toFixed(1) + ' fps';
    el('inferBar').style.width = Math.min(inferS.max/100*100, 100) + '%';
    el('ageBar').style.width   = Math.min(ageS.max/2*100, 100) + '%';
    el('fpsBar').style.width   = Math.min(fpsS.max/30*100, 100) + '%';
    drawChart(hist);

    el('link').textContent = stats.ok ? '已连接' : ('断开：' + (stats.error||''));
    el('link').className = 'tag ' + (stats.ok ? 'ok' : 'bad');
    el('src').textContent = '上游 ' + (st.size ? st.size[0]+'×'+st.size[1] : '—') +
                            '，第 ' + (st.count ?? 0) + ' 帧';
    el('peak').textContent = '采样 ' + (stats.samples ?? 0) + ' 次';
  } catch (e) {
    el('link').textContent = '出错：' + e.message;
    el('link').className = 'tag bad';
  }
}
document.getElementById('save').addEventListener('click', async () => {
  const label = encodeURIComponent(document.getElementById('label').value.trim());
  const msg = document.getElementById('saveMsg');
  msg.textContent = '保存中…';
  try {
    const res = await (await fetch('/save' + (label ? ('?label=' + label) : ''),
                                   {method: 'POST'})).json();
    msg.textContent = res.ok ? ('已保存：' + res.path + '（' + res.bytes + ' 字节，无画框）')
                             : ('失败：' + (res.error || '未知错误'));
  } catch (e) {
    msg.textContent = '失败：' + e.message;
  }
});

setInterval(tick, 120);
tick();
</script></body></html>
"""


# ---------------------------------------------------------------- HTTP
class Handler(BaseHTTPRequestHandler):
    sampler: Sampler = None

    def log_message(self, fmt, *args):
        pass

    def _send(self, body: bytes, content_type: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store, max-age=0")
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):                                      # noqa: N802
        path = self.path.split("?", 1)[0].rstrip("/") or "/"
        if path == "/save":
            query = self.path.split("?", 1)
            suffix = "?" + query[1] if len(query) > 1 else ""
            status, body = self.sampler.forward("/save" + suffix)
            self._send(body, "application/json; charset=utf-8", status)
            return
        self._send(b"not found", "text/plain; charset=utf-8", 404)

    def do_GET(self):                                       # noqa: N802
        path = self.path.split("?", 1)[0].rstrip("/") or "/"
        sampler = self.sampler
        if path == "/":
            self._send(PAGE.encode("utf-8"), "text/html; charset=utf-8")
            return
        if path == "/frame.jpg":
            jpeg = sampler.frame()
            if not jpeg:
                self._send(b"no frame yet", "text/plain; charset=utf-8", 503)
                return
            self._send(jpeg, "image/jpeg")
            return
        if path == "/save":
            query = self.path.split("?", 1)
            suffix = "?" + query[1] if len(query) > 1 else ""
            status, body = self.sampler.forward("/save" + suffix)
            self._send(body, "application/json; charset=utf-8", status)
            return
        if path == "/stats":
            snap = sampler.snapshot()
            snap.pop("jpeg", None)
            body = json.dumps(snap, ensure_ascii=False).encode("utf-8")
            self._send(body, "application/json; charset=utf-8")
            return
        if path == "/health":
            snap = sampler.snapshot()
            self._send(json.dumps({"ok": snap["ok"], "upstream": sampler.upstream,
                                   "error": snap["error"]}).encode(),
                       "application/json; charset=utf-8")
            return
        self._send(b"not found", "text/plain; charset=utf-8", 404)


def main() -> int:
    parser = argparse.ArgumentParser(description="延时 + 视频流监控台（从识别服务捞帧）")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--upstream", default="http://127.0.0.1:8080",
                        help="识别服务地址（server/run_server.py 那个）")
    parser.add_argument("--poll-hz", type=float, default=10.0, help="从上游拉帧/状态的频率")
    args = parser.parse_args()

    Handler.sampler = Sampler(args.upstream, args.poll_hz)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[monitor] 监控台 http://{args.host}:{args.port}/   （上游 {args.upstream}）")
    print("[monitor] 页面上的数：推理耗时 / 帧龄 / 实际帧率 / 拉取往返")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[monitor] 已停止")
    finally:
        server.server_close()
        Handler.sampler.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
