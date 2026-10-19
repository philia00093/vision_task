#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""伪装成「电脑端识别服务」：不加载任何模型，回伪造的检测框。

用途
----
`server/run_server.py` 需要训练好的 `best.pt` 才能启动；在模型还没训出来的时候，
用这个假服务顶上，就能把 **树莓派 → 网络 → 下行帧 → 屏上画框** 整条链路先跑通。

它和真服务**接口完全一致**（同一个路由、同一套二进制帧），所以树莓派那边一个字都不用改：

    # 电脑上
    python3 sim/mock_server.py --host 0.0.0.0 --port 8080

    # 树莓派上（就是原来那条命令）
    python3 edge/run_edge.py --infer-backend http \
        --server-url http://<电脑IP>:8080 --wire binary --fb /dev/fb0

伪造的内容
----------
* **3 个零件框**在画面里画圆移动（类别 base/support/compass，置信度 0.55~0.95 抖动）
* **手部框**每 4 秒出现 1.5 秒（洋红线，验证手部槽位）
* **步数** 1→2→3→4 每 3 秒循环，**3 个布尔**跟着变（验证状态栏 `srv stepN b101`）
* **服务端耗时** `cost_us` 随机 8~25 ms（验证状态栏显示）

参数
----
    --boxes N        伪造几个零件框（<=4，默认 3）
    --fps N          伪造框的刷新节奏（只是时间基准，不影响帧率）
    --no-hand        不发手部框
    --no-step        步数恒 0、布尔恒 0（模拟"电脑端什么都没算"）
    --jitter-ms N    人为给每次推理加 N 毫秒延迟（模拟显卡慢）
    --drop-ratio P   按概率直接丢包（不回包，模拟网络抖动；客户端会走超时）

自检：`python3 sim/mock_server.py --self-test` 会自己发一帧、检查回来的帧能不能解析。
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import protocol as wire  # noqa: E402


# ---------------------------------------------------------------- 伪造数据
class Faker:
    """按时间生成一整套"看起来像真的"识别结果。"""

    def __init__(self, boxes=3, hand=True, step=True, jitter_ms=0.0, drop_ratio=0.0,
                 seed=0):
        self.n_boxes = max(0, min(int(boxes), wire.MAX_DET))
        self.hand = bool(hand)
        self.step = bool(step)
        self.jitter = max(0.0, float(jitter_ms))
        self.drop_ratio = max(0.0, min(float(drop_ratio), 1.0))
        self.rng = random.Random(seed)
        self.t0 = time.time()
        self.served = 0
        self.dropped = 0
        self.last_jpeg = None
        self.last_size = (0, 0)
        self.last_boxes: list = []
        self.last_result: dict = {}

    def boxes_for(self, now: float, width: int, height: int) -> list[list]:
        """3 个框沿圆周运动：位置/尺寸/置信度都随时间变化，肉眼能看出"在动"。"""
        out = []
        names = ("base", "support", "compass", "top")
        for index in range(self.n_boxes):
            phase = now * 0.9 + index * (2 * math.pi / max(self.n_boxes, 1))
            cx = width * (0.5 + 0.32 * math.cos(phase))
            cy = height * (0.55 + 0.22 * math.sin(phase * 1.3))
            size = min(width, height) * (0.12 + 0.03 * math.sin(phase * 2.1))
            x1, y1 = cx - size / 2, cy - size / 2
            x2, y2 = cx + size / 2, cy + size / 2
            conf = 0.55 + 0.4 * (0.5 + 0.5 * math.sin(phase * 1.7))
            out.append([round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1),
                        index % len(wire.CLS_NAMES), round(conf, 3)])
        return out

    def hand_for(self, now: float, width: int, height: int):
        """每 4 秒露 1.5 秒的手。"""
        if not self.hand:
            return None
        cycle = (now % 4.0)
        if cycle > 1.5:
            return None
        x = width * (0.15 + 0.05 * math.sin(now * 3))
        return {"conf": 0.72, "box": (x, height * 0.62, x + width * 0.18, height * 0.95)}

    def step_for(self, now: float):
        if not self.step:
            return 0, (0, 0, 0)
        step = 1 + int(now / 3.0) % 4
        flags = (1 if int(now) % 2 == 0 else 0,
                 1 if step >= 2 else 0,
                 1 if step >= 3 else 0)
        return step, flags


FAKER: Faker | None = None
SERVER_SEQ = 0


# ---------------------------------------------------------------- HTTP 处理
class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        pass

    def _send_json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self._send_bytes(body, "application/json; charset=utf-8", status)

    def _send_bytes(self, body: bytes, content_type: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store, max-age=0")
        self.end_headers()
        self.wfile.write(body)

    # ------------------------------------------------------------ GET
    def do_GET(self):                                          # noqa: N802
        path = self.path.split("?", 1)[0].rstrip("/") or "/"
        faker = FAKER
        if path == "/":
            self._send_bytes(b"mock inference server (no model). routes: /health /state "
                             b"/frame.jpg POST /detect POST /stage",
                             "text/plain; charset=utf-8")
            return
        if path == "/health":
            self._send_json({"ok": True, "model": "MOCK (no weights)", "mock": True,
                             "names": list(wire.CLS_NAMES),
                             "imgsz": 0, "device": "none"})
            return
        if path == "/state":
            self._send_json({"ok": True, "boxes": faker.last_boxes,
                             "ms": faker.last_result.get("cost_ms", 0.0),
                             "size": list(faker.last_size),
                             "count": faker.served,
                             "step": faker.last_result.get("step", 0),
                             "flags": list(faker.last_result.get("flags", (0, 0, 0))),
                             "age": 0.0})
            return
        if path == "/frame.jpg":
            if faker.last_jpeg is None:
                self._send_json({"ok": False, "error": "还没收到任何帧"}, 404)
                return
            self._send_bytes(faker.last_jpeg, "image/jpeg")
            return
        self._send_json({"ok": False,
                         "error": "未知路径。可用：/ 、/health 、/state 、/frame.jpg 、"
                                  "POST /detect 、POST /stage"}, 404)

    # ------------------------------------------------------------ POST
    def do_POST(self):                                         # noqa: N802
        path = self.path.split("?", 1)[0].rstrip("/")
        if path == "/stage":
            self._handle_stage()
            return
        if path != "/detect":
            self._send_json({"ok": False, "error": "未知路径"}, 404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        raw = self.rfile.read(length) if length > 0 else b""
        if not raw:
            self._send_json({"ok": False, "error": "请求体是空的"}, 400)
            return
        faker = FAKER
        if faker.drop_ratio and random.random() < faker.drop_ratio:
            faker.dropped += 1
            # 直接断开，模拟"网络丢了这把包"（客户端会走超时/重试）
            self.close_connection = True
            return
        if faker.jitter:
            time.sleep(faker.jitter / 1000.0)
        if wire.looks_like_frame(raw):
            self._handle_binary(raw)
            return
        self._handle_json(raw)

    # ------------------------------------------------------------ 二进制帧
    def _handle_binary(self, raw: bytes) -> None:
        global SERVER_SEQ
        faker = FAKER
        try:
            image = wire.parse_image_frame(raw)
        except wire.FrameError as error:
            self._send_json({"ok": False, "error": f"帧不合法：{error}"}, 400)
            return
        now = time.time() - faker.t0
        width, height = image["width"] or 1280, image["height"] or 960
        boxes = faker.boxes_for(now, width, height)
        hand = faker.hand_for(now, width, height)
        step, flags = faker.step_for(now)
        cost_us = int(random.uniform(8, 25) * 1000)
        try:
            body = wire.encode_result_frame(image["seq"], boxes, step=step, flags=flags,
                                            hand=hand, seq=SERVER_SEQ, cost_us=cost_us)
        except wire.FrameError as error:
            self._send_json({"ok": False, "error": f"回包组不起来：{error}"}, 500)
            return
        SERVER_SEQ = (SERVER_SEQ + 1) & 0xFFFFFFFF
        faker.served += 1
        faker.last_jpeg = image["jpeg"]
        faker.last_size = (width, height)
        faker.last_boxes = boxes
        faker.last_result = {"step": step, "flags": flags,
                             "cost_ms": cost_us / 1000.0, "hand": hand is not None}
        self._send_bytes(body, "application/octet-stream")

    # ------------------------------------------------------------ 老 JSON 路
    def _handle_json(self, raw: bytes) -> None:
        faker = FAKER
        faker.served += 1
        faker.last_jpeg = raw
        now = time.time() - faker.t0
        boxes = faker.boxes_for(now, 1280, 960)
        faker.last_boxes = boxes
        self._send_json({"ok": True, "boxes": boxes, "ms": round(random.uniform(8, 25), 2),
                         "size": [1280, 960], "mock": True})

    # ------------------------------------------------------------ 预留字段
    def _handle_stage(self) -> None:
        """POST /stage {"step":3,"flags":[1,0,1],"hand":null} —— 和真服务一致。"""
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
        except Exception as error:
            self._send_json({"ok": False, "error": f"请求体不是 JSON：{error}"}, 400)
            return
        faker = FAKER
        if "hand" in payload and payload["hand"] is None:
            faker.hand = False
        elif "hand" in payload:
            faker.hand = True
        if "step" in payload:
            faker.step = bool(payload["step"])
        self._send_json({"ok": True, "mock": True,
                         "hand": faker.hand, "step": faker.step})


# ---------------------------------------------------------------- 自检
def self_test(port: int = 18080) -> int:
    """起一个临时实例，自己发一帧、检查回来的结果帧能解析。"""
    import threading
    import urllib.request

    global FAKER
    FAKER = Faker(boxes=3, hand=True, step=True)
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    failures = []

    def check(name, ok):
        print(f"  {'ok  ' if ok else 'FAIL'} {name}")
        if not ok:
            failures.append(name)

    try:
        health = json.loads(urllib.request.urlopen(
            f"http://127.0.0.1:{port}/health", timeout=3).read())
        check("/health 返回类别名", health.get("names") == list(wire.CLS_NAMES))

        jpeg = bytes(range(256)) * 40
        frame = wire.encode_image_frame(jpeg, 1280, 960, seq=7, quality=80)
        request = urllib.request.Request(f"http://127.0.0.1:{port}/detect", data=frame,
                                         method="POST",
                                         headers={"Content-Type": "application/octet-stream"})
        raw = urllib.request.urlopen(request, timeout=3).read()
        check("下行是二进制结果帧", wire.looks_like_frame(raw))
        result = wire.parse_result_frame(raw)
        check("下行恒 160 字节", len(raw) == 160)
        check("ref_seq 回显上行序号", result["ref_seq"] == 7)
        check("带 3 个伪造框", result["det_count"] == 3)
        check("框在画面范围内", all(0 <= d["xyxy"][0] <= 1280 for d in result["detections"]))
        check("step 在 0~4", result["step"] in (0, 1, 2, 3, 4))

        # 坏帧必须被拒
        broken = bytearray(frame)
        broken[-1] ^= 0xFF
        request = urllib.request.Request(f"http://127.0.0.1:{port}/detect", data=bytes(broken),
                                         method="POST",
                                         headers={"Content-Type": "application/octet-stream"})
        try:
            urllib.request.urlopen(request, timeout=3)
            check("坏 CRC 被拒", False)
        except urllib.error.HTTPError as error:
            check(f"坏 CRC 被拒（HTTP {error.code}）", error.code == 400)
    finally:
        server.shutdown()
    print("\n" + ("全部通过" if not failures else f"失败：{failures}"))
    return 0 if not failures else 1


def main() -> int:
    global FAKER
    parser = argparse.ArgumentParser(description="伪装成识别服务（不需要模型）")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--boxes", type=int, default=3, help="伪造几个零件框（0~4）")
    parser.add_argument("--fps", type=int, default=10, help="伪造框的运动节奏基准")
    parser.add_argument("--no-hand", action="store_true", help="不发手部框")
    parser.add_argument("--no-step", action="store_true", help="步数/布尔恒 0")
    parser.add_argument("--jitter-ms", type=float, default=0.0, help="人为加推理延迟")
    parser.add_argument("--drop-ratio", type=float, default=0.0,
                        help="按概率直接不回包（0~1，模拟丢包）")
    parser.add_argument("--self-test", action="store_true", help="跑一遍自检后退出")
    args = parser.parse_args()

    if args.self_test:
        return self_test()

    FAKER = Faker(boxes=args.boxes, hand=not args.no_hand, step=not args.no_step,
                  jitter_ms=args.jitter_ms, drop_ratio=args.drop_ratio)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[mock] 伪装识别服务已启动 {args.host}:{args.port}"
          f"（{args.boxes} 个框，手部={'开' if not args.no_hand else '关'}，"
          f"步数={'开' if not args.no_step else '关'}）")
    print(f"[mock] 树莓派这样连： python3 edge/run_edge.py --infer-backend http "
          f"--server-url http://<本机IP>:{args.port} --wire binary")
    print("[mock] Ctrl+C 退出")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[mock] 已停止")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
