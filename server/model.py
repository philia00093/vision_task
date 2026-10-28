"""模型加载与推理（只有电脑端才有 torch/GPU）。

STATE 存类别名等元信息；LAST 留最近一帧和结果给网页监控台；MODEL_LOCK 把并发请求串行化（推理不是线程安全的）。"""

from __future__ import annotations

import numpy as np
import time
import threading
from pathlib import Path



# 项目根目录 = 本文件所在包（server/）的上一层。
# 拆包时这里必须多退一层，否则默认权重会去找 server/vision_yolo_train/... 找不到。
ROOT = Path(__file__).resolve().parent.parent

DEFAULT_WEIGHTS = ROOT / "vision_yolo_train" / "runs" / "wersion_parts" / "weights" / "best.pt"

MODEL = None

MODEL_LOCK = threading.Lock()      # 推理串行化：ultralytics 不保证并发安全

# ---- 阶段模型（可选）：类别是 step_0/step_1/step_2/step_3/complete ----
# 和零件模型各干各的：零件模型出 base/support/compass/top 给物料跟踪，
# 阶段模型只判"装配区现在是第几步"，结果走下行帧里那个预留槽回给树莓派。
STAGE = {"model": None, "names": {}, "weights": ""}
STAGE_LOCK = threading.Lock()
# 阶段模型结果框只在这个矩形里采信（由 --stage-zone 传入，原图坐标）
STAGE_ZONE = {"rect": None}


def load_stage_model(weights: str, device: str, imgsz: int) -> None:
    """加载阶段模型（可选）。失败只打印，不影响主流程。"""
    from ultralytics import YOLO
    try:
        model = YOLO(weights)
    except Exception as error:
        print(f"[阶段模型] 加载失败：{error}")
        return
    names = getattr(model, "names", {}) or {}
    if isinstance(names, (list, tuple)):
        names = {index: name for index, name in enumerate(names)}
    STAGE.update(model=model, names={int(k): str(v) for k, v in names.items()},
                 weights=str(weights))
    print(f"[阶段模型] {weights}")
    print(f"           类别 {list(STAGE['names'].values())}")
    try:
        model.predict(np.zeros((imgsz, imgsz, 3), np.uint8), imgsz=imgsz,
                      conf=0.25, verbose=False, device=device or None)
    except Exception:
        pass


def infer_stage(image: np.ndarray, imgsz: int, conf: float, device: str):
    """跑阶段模型，返回置信度最高的那个 {'cls','conf','box'}；没模型/没检出返回 None。"""
    model = STAGE.get("model")
    if model is None:
        return None
    kwargs = dict(conf=conf, imgsz=imgsz, verbose=False)
    if device:
        kwargs["device"] = device
    with STAGE_LOCK:
        results = model.predict(image, **kwargs)
    boxes = results[0].boxes if results else None
    if boxes is None or len(boxes) == 0:
        return None
    best = None
    for xyxy, cls, cf in zip(boxes.xyxy.cpu().numpy(),
                             boxes.cls.cpu().numpy().astype(int),
                             boxes.conf.cpu().numpy()):
        x1, y1, x2, y2 = (float(v) for v in xyxy)
        if STAGE_ZONE["rect"] is not None:
            zx, zy, zw, zh = STAGE_ZONE["rect"]
            cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
            if not (zx <= cx <= zx + zw and zy <= cy <= zy + zh):
                continue                     # 不在装配区里，不采信
        if best is None or float(cf) > best["conf"]:
            best = {"cls": int(cls), "conf": float(cf), "box": (x1, y1, x2, y2)}
    return best

STATE = {"names": {}, "weights": "", "imgsz": 0, "device": "", "token": "",
         # 下行结果帧里"电脑端说了算"的三个字段，留给以后在电脑上做的阶段识别/手部识别。
         # 现在默认是"没报"：step=0、布尔全 0、hand=None；可以用 POST /stage 临时改，方便联调。
         "step": 0, "flags": (0, 0, 0), "hand": None}

LAST = {"jpeg": None, "boxes": [], "ms": 0.0, "size": [0, 0], "time": 0.0, "count": 0,
        "step": 0, "flags": (0, 0, 0), "hand": None}

LAST_LOCK = threading.Lock()

def load_model(weights: str, device: str, imgsz: int) -> None:
    global MODEL
    from ultralytics import YOLO
    MODEL = YOLO(weights)
    names = getattr(MODEL, "names", {}) or {}
    if isinstance(names, (list, tuple)):
        names = {index: name for index, name in enumerate(names)}
    STATE.update(names={int(k): str(v) for k, v in names.items()},
                 weights=str(weights), device=device)
    print(f"[模型] {weights}")
    print(f"       类别 {list(STATE['names'].values())} | 设备 {device}")
    # **预热**：不预热的话树莓派发来的第一帧要付 CUDA 初始化的代价（实测 1.3 秒），
    # 客户端多半会超时；这里先用一张小图跑一遍，把首帧延迟压到正常水平。
    start = time.time()
    try:
        infer(np.zeros((imgsz, imgsz, 3), np.uint8), imgsz, 0.35, 0.5, device)
        print(f"[预热] 完成，耗时 {(time.time() - start) * 1000:.0f}ms")
    except Exception as error:
        print(f"[预热] 失败（不影响使用）：{error}")

def infer(image: np.ndarray, imgsz: int, conf: float, iou: float, device: str) -> list:
    """跑一次推理，返回 [[x1,y1,x2,y2,cls,conf], ...]。"""
    kwargs = dict(conf=conf, iou=iou, imgsz=imgsz, verbose=False)
    if device:
        kwargs["device"] = device
    with MODEL_LOCK:                      # 排队推理，避免多请求同时进模型
        results = MODEL.predict(image, **kwargs)
    boxes = results[0].boxes if results else None
    if boxes is None or len(boxes) == 0:
        return []
    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    clss = boxes.cls.cpu().numpy().astype(int)
    return [[float(b[0]), float(b[1]), float(b[2]), float(b[3]), int(c), float(p)]
            for b, c, p in zip(xyxy, clss, confs)]
