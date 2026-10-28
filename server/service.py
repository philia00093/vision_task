"""HTTP 服务：路由 + 启动入口。

GET  /          网页监控台
GET  /health    连通性 + 类别名
GET  /state     最近一次检测结果
GET  /frame.jpg 最近一帧画面
POST /detect    收一帧、回框。**两种帧格式同一个路由**：
                  请求体以 b"VP" 开头 → protocol.py 的二进制帧，回 160 字节结果帧
                  否则                → 当裸 JPEG 处理，回老的 JSON

所以老客户端（裸 JPEG）和新客户端（`--wire binary`）可以同时在线，不用改端口。"""

from __future__ import annotations

import numpy as np
import cv2
import urllib.parse
import sys
import time
import json
import argparse
import re
import socket
from types import SimpleNamespace
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

try:                                                     # 帧格式模块在项目根目录
    import protocol as wire
except ImportError:                                      # 从别处起 Python 时补一下路径
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import protocol as wire

from server.model import (DEFAULT_WEIGHTS, LAST, LAST_LOCK, STAGE, STAGE_ZONE, STATE,
                          infer, infer_stage, load_model, load_stage_model)
from server.pipeline import InferPipeline

# 保存"干净帧"的目录：存的是树莓派上传的原始 JPEG，**没有任何画框**
SNAPSHOT_DIR = Path(__file__).resolve().parent.parent / "output" / "snapshots"
from server.web import PAGE_HTML


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    pipeline: "InferPipeline | None" = None       # 推理流水线（main() 里建好挂上来）
    stage_conf: float = 0.35                       # 阶段模型的置信度阈值
    hand = None                                    # MediaPipe 手部关键点（有模型才建）
    hand_frame = 0                                 # 手部检测的帧计数（--hand-every）
    imgsz = 640
    conf = 0.35
    iou = 0.5
    device = "0"
    server_seq = 0                    # 下行帧自己的序号（客户端不靠它配对，靠 ref_seq）
    window = 6                        # 允许客户端同时在飞的帧数（滑动窗口）
    rcv_nxt = 0                       # 累积确认：目前收到的连续最高帧号

    # 静音默认的逐请求日志，只在出错时打印，免得刷屏
    def log_message(self, fmt, *args):
        pass

    def _send(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self._send_bytes(body, "application/json; charset=utf-8", status)

    def _send_bytes(self, body: bytes, content_type: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        # 监控台要每帧都拿新的，绝对不能让浏览器缓存
        self.send_header("Cache-Control", "no-store, max-age=0")
        self.end_headers()
        self.wfile.write(body)

    def _authorized(self) -> bool:
        token = STATE.get("token") or ""
        if not token:
            return True
        given = self.headers.get("X-Token", "")
        return given == token

    def do_GET(self):                                     # noqa: N802
        path = self.path.split("?", 1)[0].rstrip("/") or "/"
        if path == "/":                                   # 网页监控台
            self._send_bytes(PAGE_HTML.encode("utf-8"), "text/html; charset=utf-8")
            return
        if path == "/health":
            self._send({"ok": True, "model": Path(STATE["weights"]).name,
                        "names": [STATE["names"][i] for i in sorted(STATE["names"])],
                        "imgsz": self.imgsz, "device": self.device,
                        "window": self.window, "ack": True,
                        "max_frame": wire.RESULT_FRAME_SIZE})
            return
        if path == "/state":                              # 最近一次检测结果
            with LAST_LOCK:
                payload = {"ok": True, "boxes": LAST["boxes"], "ms": LAST["ms"],
                           "size": LAST["size"], "count": LAST["count"],
                           "step": LAST["step"], "flags": list(LAST["flags"]),
                           "hand": LAST["hand"],
                           "age": (round(time.time() - LAST["time"], 3)
                                   if LAST["time"] else None),
                           "queue": (Handler.pipeline.stats()
                                     if Handler.pipeline is not None else None)}
            self._send(payload)
            return
        if path == "/frame.jpg":                          # 最近一次收到的画面
            with LAST_LOCK:
                jpeg = LAST["jpeg"]
            if jpeg is None:
                self._send({"ok": False,
                            "error": "还没收到任何帧（树莓派还没开始 POST /detect）"}, 404)
                return
            self._send_bytes(jpeg, "image/jpeg")
            return
        if path == "/save":
            self._handle_save()
            return
        self._send({"ok": False,
                    "error": "未知路径。可用：/ 、/health 、/state 、/frame.jpg 、"
                             "/save 、POST /detect 、POST /save"}, 404)

    def do_POST(self):                                     # noqa: N802
        path = self.path.split("?", 1)[0].rstrip("/") or "/"
        if path == "/stage":                              # 填下行帧的预留字段（联调用）
            self._handle_stage()
            return
        if path == "/save":                               # 保存最近一帧的干净图
            self._handle_save()
            return
        if path != "/detect":
            self._send({"ok": False, "error": "未知路径，用 /health 或 POST /detect"}, 404)
            return
        if not self._authorized():
            self._send({"ok": False, "error": "口令不对"}, 403)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        if length <= 0:
            self._send({"ok": False, "error": "请求体是空的"}, 400)
            return
        raw = self.rfile.read(length)
        # 头两个字节说了算：b"VP" 是二进制帧，FF D8 是裸 JPEG。JPEG 绝不可能是 "VP"，
        # 所以两种客户端可以同时在线，服务端只多一次 2 字节比较。
        if wire.looks_like_frame(raw):
            self._handle_binary(raw)
            return
        image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            self._send({"ok": False, "error": "JPEG 解不开"}, 400)
            return
        job = self._run_pipeline(image)
        if job.error:
            self._send({"ok": False, "error": f"推理失败：{job.error}"}, 500)
            return
        self._remember(raw, image, job.boxes, round(job.cost_ms, 2))
        self._send({"ok": True, "boxes": job.boxes, "ms": round(job.cost_ms, 2),
                    "dropped": job.dropped, "queue_ms": round(job.wait_ms, 1),
                    "size": [image.shape[1], image.shape[0]]})

    # ------------------------------------------------------------ 推理流水线
    def _run_pipeline(self, image):
        """把解码好的图丢进有界 FIFO，等这一帧的结果（顺序、限流、丢旧帧都在里面）。"""
        pipeline = Handler.pipeline
        if pipeline is not None:
            return pipeline.run(image, imgsz=self.imgsz, conf=self.conf,
                                iou=self.iou, device=self.device)
        # 没挂流水线（单帧联调时）：直接推，返回同样结构的对象
        fallback = SimpleNamespace(boxes=[], cost_ms=0.0, wait_ms=0.0,
                                   dropped=False, error="")
        start = time.time()
        try:
            fallback.boxes = infer(image, self.imgsz, self.conf, self.iou, self.device)
        except Exception as error:
            fallback.error = f"{type(error).__name__}: {error}"
        fallback.cost_ms = (time.time() - start) * 1000.0
        return fallback

    # ------------------------------------------------------------ 二进制分支
    def _handle_binary(self, raw: bytes) -> None:
        """收一整帧图像帧 → 推理 → 回 160 字节结果帧（4 组四点 + 手部 + 步数 + 布尔 + CRC）。"""
        try:
            image_info = wire.parse_image_frame(raw)
        except wire.FrameError as error:                  # CRC/版本/长度不对：直接告诉对方
            self._send({"ok": False, "error": f"帧不合法：{error}"}, 400)
            return
        image = cv2.imdecode(np.frombuffer(image_info["jpeg"], np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            self._send({"ok": False, "error": "帧里的图像解不开"}, 400)
            return
        job = self._run_pipeline(image)
        if job.error:
            self._send({"ok": False, "error": f"推理失败：{job.error}"}, 500)
            return
        boxes, cost = job.boxes, job.cost_ms
        # 手部关键点（每 --hand-every 帧一次）：把 21 个点合成手框，当 cls=4 追加到检测里。
        # 桩：树莓派侧见到 cls=4 就当成"手"，不参与零件跟踪。
        hand_box = None
        if Handler.hand is not None:
            Handler.hand_frame += 1
            if Handler.hand_frame % max(1, Handler.hand_every) == 0:
                hand = Handler.hand.detect(image)
                if hand is not None:
                    hand_box = tuple(float(v) for v in hand["box"])
                    if len(boxes) < 4:          # 还有空槽就塞进检测槽（cls=4）
                        boxes = list(boxes) + [[*hand_box, 4, float(hand["conf"])]]
                    # 四个槽都被零件占了也没关系：手框走尾部 6 字节回传
        self._remember(image_info["jpeg"], image, boxes, round(cost, 2))
        # 阶段模型（如果有）：结果放进下行帧的预留槽，cls=阶段号、box=框
        stage = None
        if STAGE.get("model") is not None:
            stage = infer_stage(image, self.imgsz, self.stage_conf, self.device)
        # 下行帧里电脑端说了算的三个字段（现在默认 0/全0/无，可用 POST /stage 临时改）
        step, flags, hand = STATE["step"], STATE["flags"], STATE["hand"]
        if stage is not None:
            hand = {"cls": stage["cls"], "conf": stage["conf"], "box": stage["box"]}
        with LAST_LOCK:
            LAST.update(step=step, flags=flags, hand=hand)
        try:
            Handler.rcv_nxt = image_info["seq"]
            body = wire.encode_result_frame(image_info["seq"], boxes, step=step,
                                            flags=flags, hand=hand,
                                            cost_us=int(cost * 1000),
                                            seq=self.server_seq,
                                            ack_seq=image_info["seq"],
                                            rcv_nxt=image_info["seq"],
                                            win=self.window,
                                            hand_box=hand_box)
        except wire.FrameError as error:
            self._send({"ok": False, "error": f"回包组不起来：{error}"}, 500)
            return
        self.server_seq = (self.server_seq + 1) & 0xFFFFFFFF
        self._send_bytes(body, "application/octet-stream")

    # ------------------------------------------------------------ 保存干净帧
    def _handle_save(self) -> None:
        """把**最近一帧原始 JPEG**（树莓派上传的，没有画框）存到 output/snapshots/。

        触发方式：
            GET  /save?label=base          （浏览器/curl 直接点）
            POST /save  {"label": "base"}  （monitor 的按钮走这个）
        label 只进文件名，允许中文；最终文件名形如 20261024_130512_123_base.jpg。
        """
        with LAST_LOCK:
            jpeg = LAST["jpeg"]
            size = list(LAST["size"])
            stamp = LAST["time"]
        if not jpeg:
            self._send({"ok": False, "error": "还没收到任何帧，无法保存"}, 404)
            return

        label = ""
        subdir = ""
        query = self.path.split("?", 1)
        if len(query) > 1:
            params = urllib.parse.parse_qs(query[1])
            label = (params.get("label") or [""])[0]
            subdir = (params.get("dir") or [""])[0]
        if self.command == "POST":
            try:
                length = int(self.headers.get("Content-Length", "0") or 0)
                if length > 0:
                    payload = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
                    label = str(payload.get("label", label) or "")
            except Exception:
                pass
        # HTTP 里的路径按 latin-1 解出来，中文会变乱码；这里还原成 UTF-8。
        # 同时支持 percent-encoding（浏览器/encodeURIComponent 会编码）。
        try:
            label = urllib.parse.unquote(label)
        except Exception:
            pass
        try:
            label = label.encode("latin-1").decode("utf-8")
        except Exception:
            pass
        label = "".join(ch for ch in label if ch not in '\\/:*?"<>|').strip()[:32]

        # 训练样本按阶段分目录存：/save?label=step0&dir=steps/step0
        # （目录名只允许普通字符，拦住 ../ 之类）
        target_dir = SNAPSHOT_DIR
        if subdir:
            safe = "/".join(part for part in urllib.parse.unquote(subdir).replace("\\", "/").split("/")
                            if part and part not in (".", "..")
                            and all(ch.isalnum() or ch in "-_." for ch in part))
            if safe:
                target_dir = SNAPSHOT_DIR / safe
        target_dir.mkdir(parents=True, exist_ok=True)
        # 按文件头判断真实格式（二进制通路上来是 JPEG，JSON 通路可能收到 PNG）
        if jpeg[:8] == b"\x89PNG\r\n\x1a\n":
            suffix = ".png"
        elif jpeg[:2] == b"\xff\xd8":
            suffix = ".jpg"
        else:
            suffix = ".bin"
        if subdir and label:
            # 训练样本用连续编号：steps/step_0_1.jpg、step_0_2.jpg ……
            # 这样按名字排序就是采集顺序，也方便后面按前缀统计张数
            index = 1
            pattern = re.compile(rf"^{re.escape(label)}_(\d+)$")
            for existing in target_dir.glob(f"{label}_*{suffix}"):
                found = pattern.match(existing.stem)
                if found:
                    index = max(index, int(found.group(1)) + 1)
            name = f"{label}_{index}{suffix}"
        else:
            now = time.strftime("%Y%m%d_%H%M%S") + f"_{int((time.time() % 1) * 1000):03d}"
            name = f"{now}_{label}{suffix}" if label else f"{now}{suffix}"
        target = target_dir / name
        try:
            target.write_bytes(jpeg)
        except OSError as error:
            self._send({"ok": False, "error": f"写盘失败：{error}"}, 500)
            return
        print(f"[save] {target}  {len(jpeg)} 字节  {size}")
        self._send({"ok": True, "path": str(target), "bytes": len(jpeg),
                    "size": size, "label": label, "frame": LAST.get("count", 0)})

    # ------------------------------------------------------------ 预留字段联调
    def _handle_stage(self) -> None:
        """POST /stage {"step":3,"flags":[1,0,1],"hand":[x1,y1,x2,y2]}

        只是把下行帧里那几个"预留/待定"的字段塞进去，方便树莓派那边先看效果：
        step=0 表示未知；hand 给 4 个坐标就发手部框，给 null 就发"没手"。
        """
        if not self._authorized():
            self._send({"ok": False, "error": "口令不对"}, 403)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
        except Exception as error:
            self._send({"ok": False, "error": f"请求体不是 JSON：{error}"}, 400)
            return
        if "step" in payload:
            step = int(payload["step"])
            if step != 0 and not 1 <= step <= 4:
                self._send({"ok": False, "error": f"step={step} 只能是 0 或 1~4"}, 400)
                return
            STATE["step"] = step
        if "flags" in payload:
            flags = payload["flags"]
            if not isinstance(flags, (list, tuple)) or len(flags) != 3:
                self._send({"ok": False, "error": "flags 要是 3 个 0/1"}, 400)
                return
            STATE["flags"] = tuple(1 if value else 0 for value in flags)
        if "hand" in payload:
            box = payload["hand"]
            if box is None:
                STATE["hand"] = None
            elif isinstance(box, (list, tuple)) and len(box) == 4:
                STATE["hand"] = {"conf": float(payload.get("hand_conf", 0.9)),
                                 "box": tuple(float(v) for v in box)}
            else:
                self._send({"ok": False, "error": "hand 要是 [x1,y1,x2,y2] 或 null"}, 400)
                return
        self._send({"ok": True, "step": STATE["step"],
                    "flags": list(STATE["flags"]), "hand": STATE["hand"]})

    def _remember(self, jpeg: bytes, image, boxes, cost: float) -> None:
        """留最近一帧/结果给网页监控台（二进制和 JSON 两条路共用）。"""
        with LAST_LOCK:
            LAST.update(jpeg=jpeg, boxes=boxes, ms=cost,
                        size=[image.shape[1], image.shape[0]],
                        time=time.time(), count=LAST["count"] + 1)

def local_ipv4_list() -> list[str]:
    """列出本机在局域网里的 IPv4 地址，用来把真实网址打给用户看。

    "占位符 <本机局域网IP> 到底是什么"这个问题不该让人去猜：启动时直接打出来。
    UDP connect 不会真的发包，只是让系统选一张出口网卡，所以离线也能用。
    """
    found = set()
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            found.add(info[4][0])
    except Exception:
        pass
    for probe in (("8.8.8.8", 80), ("114.114.114.114", 53)):   # 主出口网卡
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.connect(probe)
            found.add(sock.getsockname()[0])
            sock.close()
        except Exception:
            continue
    return sorted(ip for ip in found if not ip.startswith("127."))

def main() -> int:
    parser = argparse.ArgumentParser(description="YOLO 识别服务端（给树莓派调）")
    parser.add_argument("--host", default="0.0.0.0", help="监听地址，局域网用 0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    parser.add_argument("--imgsz", type=int, default=640, help="推理尺寸（PC 上 GPU 快，可上 960）")
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--device", default="0", help="0=第一块 GPU，cpu=CPU")
    parser.add_argument("--token", default="", help="简单口令：客户端要带同样的 X-Token")
    parser.add_argument("--window", type=int, default=6,
                        help="滑动窗口：允许树莓派同时在飞的帧数（默认 6）")
    parser.add_argument("--hand-model", default="",
                        help="MediaPipe 手部关键点模型（hand_landmarker.task）；给了就每帧检手，"
                             "手框随检测结果一起回给树莓派（cls=4）")
    parser.add_argument("--hand-every", type=int, default=2,
                        help="每隔几帧检测一次手（默认 2，省算力）")
    parser.add_argument("--stage-weights", default="",
                        help="阶段模型权重（可选，类别 step_0..step_3/complete）；"
                             "给了就每帧多跑一次，结果走下行帧预留槽")
    parser.add_argument("--stage-confs", type=float, default=0.35,
                        help="阶段模型置信度阈值")
    parser.add_argument("--stage-zone", type=int, nargs=4, metavar=("X", "Y", "W", "H"),
                        default=None,
                        help="只采信中心落在这个矩形里的阶段框（装配区，原图坐标）")
    parser.add_argument("--workers", type=int, default=1,
                        help="推理 worker 数（GPU 上 1 个就够，多了反而抢锁）")
    parser.add_argument("--queue", type=int, default=8,
                        help="服务端排队深度：满了丢最旧的帧，别让画面越算越旧")
    parser.add_argument("--pace-fps", type=float, default=0.0,
                        help="强制匀速消费（0=来多少算多少）")
    args = parser.parse_args()

    if not Path(args.weights).is_file():
        print(f"[错误] 找不到权重 {args.weights}")
        return 2
    load_model(args.weights, args.device, args.imgsz)
    if args.hand_model:
        try:
            from server.hand_mp import HANDS, HandLandmarker
            Handler.hand = HandLandmarker(args.hand_model)
            print(f"[手部] 每 {max(1, args.hand_every)} 帧检一次，手框用 cls=4 回给树莓派")
        except Exception as error:
            print(f"[手部] 初始化失败（不影响主流程）：{error}")
            Handler.hand = None
    if args.stage_weights:
        STAGE_ZONE["rect"] = tuple(args.stage_zone) if args.stage_zone else None
        load_stage_model(args.stage_weights, args.device, args.imgsz)
        if STAGE_ZONE["rect"]:
            print(f"[阶段模型] 只采信装配区 {STAGE_ZONE['rect']} 内的框")

    Handler.imgsz, Handler.conf = args.imgsz, args.conf
    Handler.stage_conf = args.stage_confs
    Handler.hand_every = max(1, args.hand_every)
    Handler.iou, Handler.device = args.iou, args.device
    Handler.window = max(1, min(int(args.window), 32))
    STATE["token"] = args.token
    # 推理流水线：有界 FIFO + 定数 worker（服务端也要缓冲，不然一到就是十几帧全挤进来）
    Handler.pipeline = InferPipeline(infer, workers=args.workers,
                                    capacity=args.queue, pace_fps=args.pace_fps,
                                    wait_timeout=2.0)

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"\n[服务] 监听 {args.host}:{args.port}   imgsz={args.imgsz} conf={args.conf} "
          f"device={args.device}")
    print(f"       推理流水线：{args.workers} 个 worker，队列 {args.queue} 帧"
          + (f"，限速 {args.pace_fps:.0f} fps" if args.pace_fps > 0 else "，不限速"))
    if args.host in ("0.0.0.0", ""):
        hosts = local_ipv4_list() or ["<本机局域网IP>"]
    else:
        hosts = [args.host]
    print("       网页监控台（浏览器打开，直接看树莓派画面 + 识别框）：")
    for host in hosts:
        print(f"         http://{host}:{args.port}/")
    if args.host in ("0.0.0.0", ""):
        print(f"         http://127.0.0.1:{args.port}/    （就在这台电脑上打开时用这个）")
    print(f"       树莓派这样调： python test_run.py --infer-backend http "
          f"--server-url http://{hosts[0]}:{args.port}"
          + (f" --server-token {args.token}" if args.token else ""))
    print(f"       保存干净帧：GET /save?label=base  →  {SNAPSHOT_DIR}")
    print("       提示：手机/树莓派要能打开这个网址，Windows 防火墙第一次会弹窗，"
          "选「专用网络」允许")
    print("       Ctrl+C 退出\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[服务] 已停止")
    finally:
        server.server_close()
    return 0
