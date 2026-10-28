"""推理后端：一张图 → 一串检测框。三种实现。

· YoloRunner：本机 ultralytics(torch) / 本机 ONNX(cv2.dnn)
· RemoteRunner：把帧发给电脑上的识别服务（异步，不阻塞主循环）

RemoteRunner 有两种帧格式（`--wire`）：
· binary：`protocol.py` 的二进制帧（下行恒定 160 字节，默认）
· json  ：老的 JSON（`{"boxes":[[x1,y1,x2,y2,cls,conf]]}`，调试/兼容用）"""

from __future__ import annotations

import numpy as np
import cv2
import sys
import time
import json
import socket
import threading
import http.client
import urllib.parse
from collections import deque
from pathlib import Path

from edge.constants import DEFAULT_INFER_BACKEND, DEFAULT_JPEG_QUALITY, DEFAULT_PART_NAMES, DEFAULT_SERVER_TIMEOUT, DEFAULT_SERVER_URL, DEFAULT_WIRE, DEFAULT_YOLO_CONF, DEFAULT_YOLO_IMGSZ, DEFAULT_YOLO_IOU

try:                                                     # 帧格式模块在项目根目录
    import protocol as wire
except ImportError:                                      # 从别处起 Python 时补一下路径
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import protocol as wire


class RemoteRunner:
    """把帧发给电脑上的识别服务，取回检测框（`--infer-backend http`）。

    ========================== 为什么改成"环形缓冲 + 滑动窗口" ==========================

    原来的实现是"只留最新一帧"：后台线程发一帧、等一帧，主循环把新帧覆盖上去。
    一帧在飞的时候下一帧根本没发出去，吞吐 = 1 / RTT。实测 Wi-Fi RTT 60~180ms，
    于是只有 4~6 fps——**不是带宽不够，是流水线空的**。

    现在改成标准的**停等窗口（滑动窗口）**：

        主循环 submit() ──▶ 环形缓冲 deque(maxlen=window*2)
                                   │  满了就丢最旧的（统计 dropped）
                                   ▼
                    window 条并发连接，各自从缓冲取帧发送
                                   │
                                   ▼
              服务端回 160 字节结果帧，尾部带 ack_seq（确认帧）
                                   │
                                   ▼
                        acked++，这一帧出窗口

    吞吐 ≈ window / RTT。window=6、RTT=180ms 时约 33 fps，正好够 30 帧。

    ============================== 握手（HELLO） ==============================
    启动时 GET /health，服务端回 `window`（允许几帧在飞）和 `ack: true`。
    客户端按这个值开连接数——服务端说 6 就开 6 条，说 1 就退回原来的停等方式。

    ============================== 帧格式 ==============================
    · binary：上行图像帧（protocol.encode_image_frame）
              下行 160 字节结果帧，**尾部 16 字节里前 10 字节是 ACK**：
              ack_seq(u32) + rcv_nxt(u32) + win(u16)
    · json  ：老的裸 JPEG + JSON（服务端返回里没有 ACK，退化为停等）
    """

    def __init__(self, url, token="", quality=80, timeout=1.0, stale_after=1.5,
                 wire_format=DEFAULT_WIRE, window=6):
        self.url = url.rstrip("/")
        self.token = token
        self.quality = int(quality)
        self.timeout = float(timeout)
        self.stale_after = float(stale_after)
        self.wire_format = "json" if str(wire_format).lower() == "json" else "binary"
        self.window_max = max(1, int(window))     # 客户端自己的上限（服务端再确认一次）
        self.window = 1                           # 握手后确定
        self.names: dict[int, str] = {}
        self.error = ""
        self.last_cost_ms = 0.0
        self.round_trip_ms = 0.0                  # 平均往返（滑动平均）
        self.sent = 0                             # 发出去的帧
        self.acked = 0                            # 收到确认的帧
        self.dropped = 0                          # 缓冲满被覆盖、没发出去的帧
        self.errors = 0                           # 请求失败次数
        self.stale_results = 0                    # 迟到的旧结果（乱序回来，已丢弃）
        self.result: dict = {}
        self._seq = 0
        self.last_submitted_seq = 0        # 最近一次 submit() 的帧号（显示端用它跟结果对齐）
        self._lock = threading.Lock()
        self._ring: deque = deque(maxlen=1)
        self._cond = threading.Condition()
        self._boxes: list = []
        self._boxes_time = 0.0
        self._last_ref = -1
        self._rtt_sum = 0.0
        self._rtt_n = 0
        self._stop = threading.Event()
        self._workers: list[threading.Thread] = []
        self._started = False
        self._last_start_try = 0.0

    # ------------------------------------------------------------ 握手 / 启动
    def start(self) -> bool:
        """握手：GET /health → 拿类别名 + 窗口大小；然后开 window 条发送线程。"""
        try:
            payload = self._request_once("GET", "/health")
            info = json.loads(payload.decode("utf-8"))
        except Exception as error:
            self.error = f"连不上识别服务 {self.url}（{error}）"
            return False
        names = info.get("names") or []
        self.names = {index: str(name) for index, name in enumerate(names)}
        server_window = int(info.get("window") or 1)
        self.window = max(1, min(server_window, self.window_max))
        self._ring = deque(maxlen=max(2, self.window * 2))
        self._started = True
        for index in range(self.window):
            thread = threading.Thread(target=self._worker, args=(index,),
                                      name=f"infer-http-{index}", daemon=True)
            thread.start()
            self._workers.append(thread)
        return True

    @property
    def ready(self) -> bool:
        """能不能干活 = 发送线程开起来了，**不看 error**。

        error 是"上一次请求失败了"的瞬时状态。以前这里是 `_started and not error`，
        于是网络抖一下 → error 置位 → ready 变 False → 上层 PartTracker.detect()
        直接 return → 再没有请求 → error 永远清不掉 → 系统永久卡死（实测：卡在
        "收到过帧但之后一帧不发"，屏上还正常刷新，只是永远没有框）。
        现在 error 只用于显示，连不上就一直在后台重试。
        """
        return self._started

    def ensure_started(self) -> bool:
        """没握手成功就每隔几秒重试一次（服务端起晚了也能自己接上）。"""
        if self._started:
            return True
        now = time.time()
        if now - self._last_start_try < 2.0:
            return False
        self._last_start_try = now
        return self.start()

    def close(self) -> None:
        self._stop.set()
        with self._cond:
            self._cond.notify_all()
        for thread in self._workers:
            thread.join(timeout=1.0)
        self._workers.clear()

    # ------------------------------------------------------------ HTTP（长连接）
    def _connect(self):
        parsed = urllib.parse.urlsplit(self.url)
        host = parsed.hostname or "127.0.0.1"
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        if parsed.scheme == "https":
            connection = http.client.HTTPSConnection(host, port, timeout=self.timeout)
        else:
            connection = http.client.HTTPConnection(host, port, timeout=self.timeout)
        connection.connect()
        try:
            connection.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except Exception:
            pass
        return connection

    def _call(self, connection, method, path, body=None, content_type="image/jpeg"):
        """在给定连接上发一次请求；失败抛异常，调用方负责重连。"""
        headers = {}
        if self.token:
            headers["X-Token"] = self.token
        if body is not None:
            headers["Content-Type"] = content_type
        connection.request(method, path, body=body, headers=headers)
        response = connection.getresponse()
        data = response.read()
        if response.status != 200:
            raise RuntimeError(f"HTTP {response.status}: "
                               f"{data[:160].decode('utf-8', 'replace')}")
        return data

    def _request_once(self, method, path) -> bytes:
        """一次性连接（只用于 /health 握手）。"""
        connection = self._connect()
        try:
            return self._call(connection, method, path)
        finally:
            try:
                connection.close()
            except Exception:
                pass

    # ------------------------------------------------------------ 属性
    @property
    def step(self) -> int:
        with self._lock:
            return int(self.result.get("step", 0) or 0)

    @property
    def flags(self) -> tuple:
        with self._lock:
            return tuple(self.result.get("flags", (0, 0, 0)))

    @property
    def hand_box(self):
        """电脑端 MediaPipe 给的手框（尾部 6 字节压缩传回来的），没有就是 None。"""
        with self._lock:
            box = (self.result or {}).get("hand_box")
        return box

    @property
    def hand(self):
        with self._lock:
            return self.result.get("hand")

    @property
    def inflight(self) -> int:
        with self._lock:
            return max(0, self.sent - self.acked)

    def describe_stats(self) -> str:
        return (f"窗 {self.window} | 发 {self.sent} 收 {self.acked} "
                f"丢 {self.dropped} 错 {self.errors} | 往返 {self.round_trip_ms:.0f}ms")

    # ------------------------------------------------------------ 主循环接口
    def submit(self, frame) -> None:
        """把一帧塞进环形缓冲。

        两道闸门，目的是**别让队列越积越深**（队列一深，往返时间就从 80ms 涨到 900ms）：
          1. 在飞帧数已经到窗口 → 直接丢，不排队；
          2. 环形缓冲满了 → 丢最旧的。
        丢掉的帧计入 dropped，界面上能看出"这台网络吃不下 30 帧"。
        """
        with self._cond:
            if self.sent - self.acked >= self.window:      # 窗口已满：不排队，直接丢
                self.dropped += 1
                return
            if len(self._ring) >= self._ring.maxlen:
                try:
                    self._ring.popleft()
                    self.dropped += 1
                except IndexError:
                    pass
            self._seq = (self._seq + 1) & 0xFFFFFFFF
            self.last_submitted_seq = self._seq
            self.sent += 1                     # 真正入队才算发出
            self._ring.append((self._seq, frame.copy()))
            self._cond.notify()

    def detect(self, frame) -> list[tuple]:
        """提交一帧，立刻返回"最近一次"的结果（不阻塞主循环）。"""
        if not self._started and not self.ensure_started():
            return []
        self.submit(frame)
        with self._lock:
            boxes = list(self._boxes)
            fresh = (time.time() - self._boxes_time) <= self.stale_after
            self.result["inflight"] = max(0, self.sent - self.acked)
        return boxes if fresh else []

    # ------------------------------------------------------------ 发送线程
    def _worker(self, index: int) -> None:
        connection = None
        while not self._stop.is_set():
            with self._cond:
                while not self._ring and not self._stop.is_set():
                    self._cond.wait(0.05)
                if self._stop.is_set():
                    return
                try:
                    seq, frame = self._ring.popleft()
                except IndexError:
                    continue
            try:
                if connection is None:
                    connection = self._connect()
                ok, buffer = cv2.imencode(".jpg", frame,
                                          [int(cv2.IMWRITE_JPEG_QUALITY), self.quality])
                if not ok:
                    continue
                jpeg = buffer.tobytes()
                start = time.time()
                if self.wire_format == "binary":
                    boxes, result = self._post_binary(connection, frame, jpeg, seq)
                else:
                    boxes, result = self._post_json(connection, jpeg, seq)
                elapsed = (time.time() - start) * 1000.0
                with self._lock:
                    self.acked += 1
                    self._rtt_sum += elapsed
                    self._rtt_n += 1
                    self.round_trip_ms = self._rtt_sum / self._rtt_n
                    # ---- 顺序保护：只收"比手上更新的"结果 ----
                    # 窗口里有 4 条连接并发，结果回来的顺序不保证。以前 self.result 是
                    # 无条件赋值，于是 500 号结果回来后又会被 480 号覆盖 → 画面跳回旧帧，
                    # 看起来就是"帧是乱的"。这里统一按 ref_seq 单调递增来收。
                    ref = (result or {}).get("ref_seq", seq)
                    if result is not None and ref >= self._last_ref:
                        self.result = result
                        self._last_ref = ref
                        self.last_cost_ms = float(result.get("cost_ms", 0.0) or 0.0)
                        if boxes is not None:
                            self._boxes = boxes
                            self._boxes_time = time.time()
                    else:
                        self.stale_results += 1        # 迟到的旧结果：确认收下，但不改显示
                    self.error = ""
            except Exception as error:
                self.errors += 1
                self.error = f"请求失败（{error}）"
                try:
                    if connection is not None:
                        connection.close()
                except Exception:
                    pass
                connection = None
                # ---- 关键：把窗口位置还回去 ----
                # sent 在 submit() 入队时就 +1，只有收到回包（acked）才让窗口前进。
                # 请求失败时这一帧永远不会回包，如果不在这里退掉，窗口会被"永远在飞"
                # 的帧占满：sent - acked 恒等于 window → submit() 全部丢弃 →
                # 服务端重启一次，客户端就永久卡死（实测：在飞 16/16、之后一帧不发）。
                with self._cond:
                    self.sent = max(self.acked, self.sent - 1)
                    stale = len(self._ring)          # 断线期间攒下的帧已经过期，别补发
                    if stale:
                        self._ring.clear()
                        self.dropped += stale
                        self.sent = max(self.acked, self.sent - stale)
                time.sleep(0.05)
        try:
            if connection is not None:
                connection.close()
        except Exception:
            pass

    # ------------------------------------------------------------ 两种帧格式
    def _post_binary(self, connection, frame, jpeg: bytes, seq: int):
        height, width = frame.shape[:2]
        body = wire.encode_image_frame(jpeg, width, height, seq=seq,
                                       quality=self.quality,
                                       stamp_ms=int(time.time() * 1000) & 0xFFFFFFFF)
        raw = self._call(connection, "POST", "/detect", body, "application/octet-stream")
        if not wire.looks_like_frame(raw):            # 旧服务端走 JSON 分支
            payload = json.loads(raw.decode("utf-8"))
            if not payload.get("ok"):
                raise RuntimeError(str(payload.get("error", "服务端返回失败")))
            return [tuple(item) for item in payload.get("boxes", [])], None
        result = wire.parse_result_frame(raw)
        boxes = []
        for det in result["detections"]:
            x1, y1, x2, y2 = det["xyxy"]
            boxes.append((float(x1), float(y1), float(x2), float(y2),
                          int(det["cls"]), float(det["conf"])))
        return boxes, result

    def _post_json(self, connection, jpeg: bytes, seq: int):
        raw = self._call(connection, "POST", "/detect", jpeg, "image/jpeg")
        payload = json.loads(raw.decode("utf-8"))
        if not payload.get("ok"):
            raise RuntimeError(str(payload.get("error", "服务端返回失败")))
        self.last_cost_ms = float(payload.get("ms", 0.0))
        return [tuple(item) for item in payload.get("boxes", [])], None

class YoloRunner:
    """把"一张图"跑成"一串检测框"，两种后端可选：

    ultralytics : 直接用 best.pt（开发机上方便，依赖 torch）
    onnx        : ONNX 模型 + cv2.dnn（**树莓派上用这个**：不用装 torch，
                  OpenCV 本来就装了，arm64 上还能再挂 NCNN/OpenVINO 后端）

    树莓派 4B 没有 CUDA：yolo26n 用 torch 跑一帧要 1~3 秒，而 ONNX + cv2.dnn
    只要几百毫秒。所以"能不能跑通"的关键不在代码，而在**别在 Pi 上装 torch**。
    导出 ONNX（开发机上跑一次）：
        yolo export model=vision_yolo_train/runs/wersion_parts/weights/best.pt \
             format=onnx imgsz=640 simplify=True
    """

    def __init__(self, weights, backend=DEFAULT_INFER_BACKEND, imgsz=DEFAULT_YOLO_IMGSZ,
                 conf=DEFAULT_YOLO_CONF, iou=DEFAULT_YOLO_IOU, device=None,
                 server_url=DEFAULT_SERVER_URL, server_token="",
                 jpeg_quality=DEFAULT_JPEG_QUALITY, server_timeout=DEFAULT_SERVER_TIMEOUT,
                 wire_format=DEFAULT_WIRE, window=6, result_hold=1.5):
        self.weights = str(weights)
        self.backend = backend
        self.result_hold = float(result_hold)
        self.imgsz = int(imgsz)
        self.conf, self.iou = float(conf), float(iou)
        self.device = device or None
        self.server_url = server_url
        self.server_token = server_token
        self.jpeg_quality = int(jpeg_quality)
        self.server_timeout = float(server_timeout)
        self.wire_format = wire_format
        self.window = max(1, int(window))       # 滑动窗口（服务端 /health 会再确认一次）
        self.model = None
        self.net = None
        self.remote: RemoteRunner | None = None
        self.names: dict[int, str] = {}
        self.error = ""
        self._load()

    @property
    def ready(self) -> bool:
        if self.remote is not None:
            return self.remote.ready
        return self.model is not None or self.net is not None

    def _load(self):
        if self.backend == "http":
            self.remote = RemoteRunner(self.server_url, token=self.server_token,
                                       quality=self.jpeg_quality,
                                       timeout=self.server_timeout,
                                       wire_format=self.wire_format,
                                       window=self.window,
                                       stale_after=self.result_hold)
            if self.remote.start():
                self.names = dict(self.remote.names)
            else:
                self.error = self.remote.error
        elif self.backend == "onnx":
            self._load_onnx()
        else:
            self._load_ultralytics()

    def _load_ultralytics(self):
        try:
            from ultralytics import YOLO
        except Exception as error:               # 没装 torch/ultralytics
            self.error = f"没装 ultralytics/torch（{error}）"
            return
        if not Path(self.weights).is_file():
            self.error = f"找不到模型文件 {self.weights}"
            return
        try:
            self.model = YOLO(self.weights)
        except Exception as error:
            self.error = f"模型加载失败（{error}）"
            return
        names = getattr(self.model, "names", None)
        if isinstance(names, dict):
            self.names = {int(k): str(v) for k, v in names.items()}
        elif isinstance(names, (list, tuple)):
            self.names = {index: str(v) for index, v in enumerate(names)}

    def _load_onnx(self):
        if not Path(self.weights).is_file():
            self.error = f"找不到 ONNX 模型 {self.weights}（先在开发机上导出）"
            return
        try:
            self.net = cv2.dnn.readNetFromONNX(str(self.weights))
        except Exception as error:
            self.error = f"ONNX 读不进来（{error}）"
            return
        # 纯 CPU。装了 opencv 的 NCNN/OpenVINO 后端时，这里换一行就能再加速。
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self.names = {index: name for index, name in enumerate(DEFAULT_PART_NAMES)}

    def detect(self, frame) -> list[tuple]:
        """返回 [(x1, y1, x2, y2, cls_id, conf), ...]，坐标是 frame 自己的像素坐标。

        http 后端：把帧丢给后台线程，立刻返回"最近一次的结果"（不阻塞主循环）。
        """
        if self.remote is not None:
            return self.remote.detect(frame)
        if self.net is not None:
            return self._detect_onnx(frame)
        if self.model is not None:
            return self._detect_ultralytics(frame)
        return []

    def close(self) -> None:
        if self.remote is not None:
            self.remote.close()

    def _detect_ultralytics(self, frame) -> list[tuple]:
        kwargs = dict(conf=self.conf, iou=self.iou, imgsz=self.imgsz, verbose=False)
        if self.device:
            kwargs["device"] = self.device
        results = self.model.predict(frame, **kwargs)
        boxes = results[0].boxes if results else None
        if boxes is None or len(boxes) == 0:
            return []
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        clss = boxes.cls.cpu().numpy().astype(int)
        return [(float(b[0]), float(b[1]), float(b[2]), float(b[3]), int(c), float(p))
                for b, c, p in zip(xyxy, clss, confs)]

    def _detect_onnx(self, frame) -> list[tuple]:
        """ONNX 推理：自己 letterbox、自己 NMS，不依赖 ultralytics/torch。"""
        height, width = frame.shape[:2]
        size = self.imgsz
        scale = min(size / float(width), size / float(height))
        new_w, new_h = int(round(width * scale)), int(round(height * scale))
        resized = cv2.resize(frame, (new_w, new_h))
        canvas = np.full((size, size, 3), 114, np.uint8)        # letterbox 灰边
        dx, dy = (size - new_w) // 2, (size - new_h) // 2
        canvas[dy:dy + new_h, dx:dx + new_w] = resized
        blob = cv2.dnn.blobFromImage(canvas, 1 / 255.0, (size, size),
                                     swapRB=True, crop=False)
        self.net.setInput(blob)
        pred = np.squeeze(self.net.forward())
        if pred.ndim != 2:
            raise RuntimeError(f"不认识的 ONNX 输出形状 {pred.shape}")
        if pred.shape[0] < pred.shape[1]:                       # (4+nc, N) → (N, 4+nc)
            pred = pred.T
        expected = 4 + len(self.names)
        if pred.shape[1] != expected:
            raise RuntimeError(
                f"ONNX 输出列数 {pred.shape[1]}，期望 4+{len(self.names)}={expected}。"
                "这个导出多半是 end-to-end（NMS 已内置）格式，需要按实际布局改解析："
                f"实际形状 {pred.shape}")
        cx, cy, bw, bh = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
        scores = pred[:, 4:]
        cls_ids = scores.argmax(1)
        best = scores[np.arange(len(scores)), cls_ids]
        keep = best >= self.conf
        if not np.any(keep):
            return []
        rects = np.stack([cx - bw / 2, cy - bh / 2, bw, bh], 1)[keep]
        scores_kept = best[keep]
        classes_kept = cls_ids[keep]
        indices = cv2.dnn.NMSBoxes(rects.tolist(), scores_kept.tolist(), self.conf, self.iou)
        out = []
        for index in np.array(indices).flatten():
            x, y, w, h = rects[index]
            out.append(((x - dx) / scale, (y - dy) / scale,
                        (x + w - dx) / scale, (y + h - dy) / scale,
                        int(classes_kept[index]), float(scores_kept[index])))
        return out
