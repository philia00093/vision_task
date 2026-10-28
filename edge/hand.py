#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""手部检测 + 跟踪（肤色分割 + 卡尔曼，不用神经网络）。

为什么不用 YOLO / MediaPipe
---------------------------
· 网上找不到可用的「手」YOLO 权重（试过 HF 上的 naruto 手势模型：12 张只中 2 张，
  类别是 hare/dog 这种手势，不是手部检测）。
· MediaPipe Hands 在我们的画面里只中 1/12 —— 相机是从上往下看，手多是伸进来
  的半个手掌/几根手指，它假定的是"正对手掌"的场景。

为什么肤色够用
--------------
工位是固定的：料盘是黑的、台面是灰的、零件是铜的，只有**手是肤色**。
实测：有手时肤色掩码能盖住整条小臂，没手时占比 < 0.2%（见 output/skin_test.jpg）。
用 YCrCb 的经典区间 ∩ HSV 的橙红区间，两路同时满足才判成肤色，抗反光。

跟踪沿用 `KalmanTrack`（和零件同一套）：状态只滤中心点，丢检时速度衰减滑停，
所以手短暂被零件挡住也不会飞。

用法
----
    tracker = HandTracker()
    hands = tracker.update(view, now)      # -> [{box, conf, trail, track_id, mask_ratio}]
    ratio = tracker.zone_overlap(hand, zone_rect)   # 手在该区域里占多少（0~1）
"""

from __future__ import annotations

import time

import cv2
import numpy as np

from edge.tracking import KalmanTrack

DEFAULT_MIN_AREA = 0.004        # 手至少占画面这个比例才算（滤掉小臂一小块）
DEFAULT_MAX_AREA = 0.25         # 手最多占画面这个比例（木桌面/背景连成一片时会被整块框住）
DEFAULT_MAX_AGE = 8             # 丢检多少帧后删掉轨迹
DEFAULT_TRAIL = 16


def skin_mask(frame):
    """BGR → 肤色二值掩码（255=像皮肤）。"""
    if frame is None or frame.size == 0:
        return None
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(ycrcb, (0, 133, 77), (255, 175, 130))
    m2 = cv2.inRange(hsv, (0, 25, 50), (28, 190, 255))
    mask = cv2.bitwise_and(m1, m2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
    return mask


class HandTracker:
    """一只/两只手的检测 + 卡尔曼跟踪。"""

    def __init__(self, min_area_ratio: float = DEFAULT_MIN_AREA,
                 max_area_ratio: float = DEFAULT_MAX_AREA,
                 max_age: int = DEFAULT_MAX_AGE, trail_len: int = DEFAULT_TRAIL,
                 max_hands: int = 1, scale: float = 0.5):
        self.min_area = float(min_area_ratio)
        self.max_area = float(max_area_ratio)
        self.max_age = int(max_age)
        self.trail_len = int(trail_len)
        self.max_hands = int(max_hands)
        self.scale = float(scale)          # 掩码在缩小图上算，省 CPU
        self.tracks: dict[int, KalmanTrack] = {}
        self.trails: dict[int, list] = {}
        self.next_id = 1
        self.bounds = (1280, 960)
        self.last_cost_ms = 0.0
        self.last_mask = None

    # ------------------------------------------------------------ 检测
    def detect(self, frame):
        """返回 [((x1,y1,x2,y2), conf), ...]，最多 max_hands 个。"""
        if frame is None:
            return []
        small = frame if self.scale == 1.0 else cv2.resize(
            frame, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)
        mask = skin_mask(small)
        if mask is None:
            return []
        self.last_mask = mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        area_max = float(mask.shape[0] * mask.shape[1]) or 1.0
        boxes = []
        for contour in contours:
            area = cv2.contourArea(contour)
            ratio = area / area_max
            if ratio < self.min_area or ratio > self.max_area:
                continue                       # 太小=碎块，太大=桌面/背景
            x, y, w, h = cv2.boundingRect(contour)
            # 手的形状大概是"一团+一条手臂"，太细长的（比如一根手指/反光条）不要
            if w < 8 or h < 8:
                continue
            fill = area / float(w * h + 1e-6)
            if fill < 0.25:
                continue
            inv = 1.0 / self.scale
            box = (x * inv, y * inv, (x + w) * inv, (y + h) * inv)
            boxes.append((box, min(0.99, 0.4 + ratio * 3.0)))
        boxes.sort(key=lambda item: -(item[0][2] - item[0][0]) * (item[0][3] - item[0][1]))
        return boxes[:max(1, self.max_hands)]

    # ------------------------------------------------------------ 跟踪
    def update(self, frame, now: float | None = None):
        now = time.time() if now is None else now
        started = now
        detections = self.detect(frame)
        self.bounds = (frame.shape[1], frame.shape[0])

        # 预测 + 关联（按中心距离贪心）
        pairs = []
        for tid, track in self.tracks.items():
            track.width, track.height = self.bounds
            predicted = track.predict(now, damp_velocity=track.miss >= 4)
            for index, (box, _conf) in enumerate(detections):
                cx, cy = (box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5
                dist = ((predicted[0] - cx) ** 2 + (predicted[1] - cy) ** 2) ** 0.5
                pairs.append((dist, tid, index))
        pairs.sort()
        used_tracks, used_det = set(), set()
        for dist, tid, index in pairs:
            if tid in used_tracks or index in used_det:
                continue
            box, conf = detections[index]
            w = box[2] - box[0]
            if dist > max(120.0, w * 1.5):        # 手跳太远不关联
                continue
            self.tracks[tid].update(box, 0, "hand", conf, now)
            used_tracks.add(tid)
            used_det.add(index)
        # 新目标
        for index, (box, conf) in enumerate(detections):
            if index in used_det:
                continue
            track = KalmanTrack(self.next_id, box, 0, "hand", conf,
                                self.max_age, self.bounds)
            self.tracks[self.next_id] = track
            self.trails[self.next_id] = []
            self.next_id += 1
        # 老化
        for tid in list(self.tracks):
            track = self.tracks[tid]
            if track.miss > self.max_age:
                self.tracks.pop(tid, None)
                self.trails.pop(tid, None)
            else:
                track.width, track.height = self.bounds
                track.predict(now, damp_velocity=track.miss >= 4)

        hands = []
        for tid, track in self.tracks.items():
            # 只输出**这一帧真的检出过**的轨迹。之前把预测框也画出来，
            # 手一拿开就有好几个旧框慢慢飘着走，看着就是"好几个框乱飞"。
            if track.miss > 0:
                continue
            box = track.box
            trail = self.trails.setdefault(tid, [])
            trail.append(((box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5))
            if len(trail) > self.trail_len:
                del trail[:len(trail) - self.trail_len]
            hands.append({"box": box, "conf": track.conf, "track_id": tid,
                          "trail": list(trail), "predicted": track.miss > 0})
        self.last_cost_ms = (time.time() - started) * 1000.0
        return hands

    # ------------------------------------------------------------ 区域占比
    def zone_ratio(self, hand_box, rect) -> float:
        """手框有多少比例落在 rect 里（0~1）。手比料盘大时用"手在区内"算更合适。"""
        if hand_box is None or rect is None:
            return 0.0
        hx1, hy1, hx2, hy2 = hand_box
        rx, ry, rw, rh = rect
        ix1, iy1 = max(hx1, rx), max(hy1, ry)
        ix2, iy2 = min(hx2, rx + rw), min(hy2, ry + rh)
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
        inter = (ix2 - ix1) * (iy2 - iy1)
        area = max(1e-6, (hx2 - hx1) * (hy2 - hy1))
        return float(inter / area)


class HandFeed:
    """从电脑端拉 21 个手部关键点（`GET /hand`），5Hz 就够。

    为什么单独走一个接口：21 个点 = 84 字节，160 字节的下行帧放不下；
    而这个信息**不需要和图像帧严格对齐**（手是连续运动的），低频拉取完全够用，
    还不会影响主链路（拉取失败就当没有手）。
    """

    def __init__(self, server_url: str, hz: float = 5.0, timeout: float = 0.6):
        import threading
        self.url = server_url.rstrip("/") + "/hand"
        self.period = 1.0 / max(hz, 0.5)
        self.timeout = float(timeout)
        self.points: list = []
        self.conf = 0.0
        self.ok = False
        self.error = ""
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="hand-feed", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        import json
        import urllib.request
        while not self._stop.is_set():
            try:
                with urllib.request.urlopen(self.url, timeout=self.timeout) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                with self._lock:
                    self.points = payload.get("points") or []
                    self.conf = float(payload.get("conf") or 0.0)
                    self.ok = bool(payload.get("ok"))
                    self.error = ""
            except Exception as error:
                with self._lock:
                    self.ok = False
                    self.error = f"{type(error).__name__}"
            self._stop.wait(self.period)

    def latest(self):
        with self._lock:
            return list(self.points), self.conf

    def close(self) -> None:
        self._stop.set()


def draw_keypoints(frame, points, color=(255, 0, 255), radius: int = 4) -> None:
    """把 21 个关键点画成一串小圆点（+ 指节连线），不用外框。"""
    if not points:
        return
    import numpy as _np
    pts = [tuple(int(round(v)) for v in p[:2]) for p in points]
    # MediaPipe 的 21 点连接关系（手掌 5 指）
    chains = [(0, 1, 2, 3, 4), (0, 5, 6, 7, 8), (9, 10, 11, 12),
              (13, 14, 15, 16), (0, 17, 18, 19, 20), (5, 9, 13, 17)]
    for chain in chains:
        for a, b in zip(chain[:-1], chain[1:]):
            if a < len(pts) and b < len(pts):
                cv2.line(frame, pts[a], pts[b], color, 2, cv2.LINE_AA)
    for index, point in enumerate(pts):
        r = radius + 1 if index in (4, 8, 12, 16, 20) else radius
        cv2.circle(frame, point, r, color, -1, cv2.LINE_AA)
