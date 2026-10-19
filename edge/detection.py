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
import threading
import urllib.request
from pathlib import Path

from edge.constants import DEFAULT_INFER_BACKEND, DEFAULT_JPEG_QUALITY, DEFAULT_PART_NAMES, DEFAULT_SERVER_TIMEOUT, DEFAULT_SERVER_URL, DEFAULT_WIRE, DEFAULT_YOLO_CONF, DEFAULT_YOLO_IMGSZ, DEFAULT_YOLO_IOU

try:                                                     # 帧格式模块在项目根目录
    import protocol as wire
except ImportError:                                      # 从别处起 Python 时补一下路径
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import protocol as wire


class RemoteRunner:
    """把帧发给电脑上的识别服务，取回检测框（`--infer-backend http`）。

    为什么要异步：如果主循环"发一帧 → 等结果"，一个来回 50~100ms 就把 30fps 拖到 10fps。
    这里后台线程只负责**最新一帧**（旧帧直接丢），主循环 submit 完立刻返回，
    拿到的永远是"最近一次的结果"；位置差由卡尔曼按墙钟时间外推补上，
    所以显示端照样是相机帧率、框也不抖。

    只用标准库（urllib），树莓派上不用装任何东西。

    帧格式（`--wire`）：
        binary  上行 = `protocol.encode_image_frame`，下行 = 160 字节结果帧
                （里面还带 step / 3 个布尔 / 手部框 / 服务端耗时）
        json    上行 = 裸 JPEG，下行 = `{"boxes":[[x1,y1,x2,y2,cls,conf]]}`
    两种都能用同一个服务端：服务端看请求体的头两个字节就知道该走哪条。
    """

    def __init__(self, url, token="", quality=80, timeout=1.0, stale_after=0.6,
                 wire_format=DEFAULT_WIRE):
        self.url = url.rstrip("/")
        self.token = token
        self.quality = int(quality)
        self.timeout = float(timeout)
        self.stale_after = float(stale_after)
        self.wire_format = "json" if str(wire_format).lower() == "json" else "binary"
        self.names: dict[int, str] = {}
        self.error = ""
        self.last_cost_ms = 0.0            # 服务端报的推理耗时
        self.round_trip_ms = 0.0           # 本地测的整轮往返
        self.dropped = 0                   # 因为太慢被丢掉的帧数
        self.result: dict = {}             # 最近一次下行帧（binary 才有：step/flags/hand/cost）
        self._seq = 0                      # 上行帧序号，服务端会原样回在 ref_seq 里
        self._lock = threading.Lock()
        self._pending = None
        self._boxes: list = []
        self._boxes_time = 0.0
        self._stop = threading.Event()
        self._thread = None

    # ------------------------------------------------------------ 启动
    def start(self) -> bool:
        """探一次 /health：确认连通 + 拿类别名。连不上不抛异常，交给调用方降级。"""
        try:
            payload = self._request("GET", "/health")
        except Exception as error:
            self.error = f"连不上识别服务 {self.url}（{error}）"
            return False
        names = payload.get("names") or []
        self.names = {index: str(name) for index, name in enumerate(names)}
        self._thread = threading.Thread(target=self._loop, name="infer-http", daemon=True)
        self._thread.start()
        return True

    @property
    def ready(self) -> bool:
        return self._thread is not None and not self.error

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    # ------------------------------------------------------------ HTTP
    def _request(self, method: str, path: str, body: bytes | None = None) -> dict:
        """发请求并当 JSON 解析（只用于 GET /health）。"""
        raw = self._request_bytes(method, path, body)
        return json.loads(raw.decode("utf-8"))

    def _request_bytes(self, method: str, path: str, body: bytes | None = None,
                       content_type: str = "image/jpeg") -> bytes:
        """发请求，原样返回响应体字节（二进制帧不走 JSON）。"""
        request = urllib.request.Request(self.url + path, data=body, method=method)
        if self.token:
            request.add_header("X-Token", self.token)
        if body is not None:
            request.add_header("Content-Type", content_type)
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return response.read()

    @property
    def step(self) -> int:
        """电脑端报的装配步数（0=没报；binary 帧才有）。"""
        with self._lock:
            return int(self.result.get("step", 0) or 0)

    @property
    def flags(self) -> tuple:
        """电脑端报的 3 个布尔（没报就是全 0）。"""
        with self._lock:
            return tuple(self.result.get("flags", (0, 0, 0)))

    @property
    def hand(self):
        """电脑端报的手部框（预留位；没报就是 None）。"""
        with self._lock:
            return self.result.get("hand")

    def _store(self, boxes, result=None) -> None:
        with self._lock:
            self._boxes = boxes
            self._boxes_time = time.time()
            if result is not None:
                self.result = result

    def submit(self, frame) -> None:
        """主循环调用：把最新一帧交给后台线程（覆盖上一帧，不排队）。"""
        with self._lock:
            if self._pending is not None:
                self.dropped += 1          # 上一帧还没发出去就被替换了
            self._pending = frame.copy()   # 拷一份：主循环还要继续往这帧上画东西

    def detect(self, frame) -> list[tuple]:
        """主循环调用：提交这一帧，并立刻返回"最近一次"的结果（不阻塞）。"""
        self.submit(frame)
        with self._lock:
            boxes = list(self._boxes)
        if time.time() - self._boxes_time > self.stale_after:
            return []                      # 结果太旧（网络卡了）就当没检出，别画错位置
        return boxes

    # ------------------------------------------------------------ 后台线程
    def _loop(self) -> None:
        while not self._stop.is_set():
            with self._lock:
                frame = self._pending
                self._pending = None
            if frame is None:
                time.sleep(0.004)
                continue
            ok, buffer = cv2.imencode(".jpg", frame,
                                      [int(cv2.IMWRITE_JPEG_QUALITY), self.quality])
            if not ok:
                continue
            start = time.time()
            try:
                if self.wire_format == "binary":
                    boxes, result = self._post_binary(frame, buffer.tobytes())
                else:
                    boxes, result = self._post_json(buffer.tobytes())
            except Exception as error:
                self.error = f"请求失败（{error}）"
                time.sleep(0.2)            # 网络不好时别把 CPU 打满
                continue
            self.error = ""
            self.round_trip_ms = (time.time() - start) * 1000.0
            self._store(boxes, result)

    def _post_json(self, jpeg: bytes):
        """老格式：裸 JPEG 上行，JSON 下行。"""
        payload = self._request("POST", "/detect", jpeg)
        if not payload.get("ok"):
            raise RuntimeError(str(payload.get("error", "服务端返回失败")))
        self.last_cost_ms = float(payload.get("ms", 0.0))
        boxes = [tuple(item) for item in payload.get("boxes", [])]
        return boxes, None

    def _post_binary(self, frame, jpeg: bytes):
        """二进制帧：整帧上行，160 字节结果帧下行。"""
        self._seq = (self._seq + 1) & 0xFFFFFFFF
        height, width = frame.shape[:2]
        body = wire.encode_image_frame(jpeg, width, height, seq=self._seq,
                                       quality=self.quality,
                                       stamp_ms=int(time.time() * 1000) & 0xFFFFFFFF)
        raw = self._request_bytes("POST", "/detect", body, "application/octet-stream")
        if not wire.looks_like_frame(raw):          # 服务端走的是 JSON 分支（旧服务端）
            payload = json.loads(raw.decode("utf-8"))
            if not payload.get("ok"):
                raise RuntimeError(str(payload.get("error", "服务端返回失败")))
            self.last_cost_ms = float(payload.get("ms", 0.0))
            return [tuple(item) for item in payload.get("boxes", [])], None

        result = wire.parse_result_frame(raw)       # FrameError 也是异常，上面统一兜住
        if result["ref_seq"] and result["ref_seq"] != self._seq:
            # 迟到的旧结果：位置沿用上一批，别让旧框盖掉新框（多线程/网络抖动时会遇到）
            with self._lock:
                return list(self._boxes), result
        self.last_cost_ms = float(result.get("cost_ms", 0.0))
        boxes = []
        for det in result["detections"]:
            x1, y1, x2, y2 = det["xyxy"]
            boxes.append((float(x1), float(y1), float(x2), float(y2),
                          int(det["cls"]), float(det["conf"])))
        return boxes, result

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
                 wire_format=DEFAULT_WIRE):
        self.weights = str(weights)
        self.backend = backend
        self.imgsz = int(imgsz)
        self.conf, self.iou = float(conf), float(iou)
        self.device = device or None
        self.server_url = server_url
        self.server_token = server_token
        self.jpeg_quality = int(jpeg_quality)
        self.server_timeout = float(server_timeout)
        self.wire_format = wire_format
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
                                       wire_format=self.wire_format)
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
