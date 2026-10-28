#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""电脑端手部关键点：MediaPipe HandLandmarker（21 个关键点）。

为什么在电脑端跑
----------------
树莓派上装不了 mediapipe（系统 python 没有 pip，arm64 轮子也不全），
而这边有现成的 conda 环境和 GPU，单帧 10~20 ms，顺手就把手框算出来了。

为什么用 VIDEO 模式
------------------
MediaPipe 的 IMAGE 模式每帧重新检测，手只露一半（我们的相机从上方看）时经常检不到；
VIDEO 模式会**在帧之间跟踪**，一旦检到就靠上一帧的外推接住，实测更稳。

发什么回去
----------
21 个关键点没法塞进 160 字节的下行帧（一个槽只有 4 个点）。所以这里把 21 个点
**合成一个手框**（最小外接矩形）发回去，信息量等价，协议一个字都不用改。

另外把 21 个点存到 `HANDS["last"]`，`/state` 里能看到，监控台/调试都能用。
"""

from __future__ import annotations

import time

HANDS = {"ready": False, "last": None, "cost_ms": 0.0, "hits": 0}


class HandLandmarker:
    """包一层 MediaPipe，输入 BGR 帧，输出 21 点 + 手框。"""

    def __init__(self, model_path: str, max_hands: int = 2, running_mode: str = "video"):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        self.mp = mp
        self.vision = vision
        self.max_hands = int(max_hands)
        self.mode_name = running_mode
        mode = vision.RunningMode.VIDEO if running_mode == "video" else vision.RunningMode.IMAGE
        options = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model_path),
            num_hands=self.max_hands,
            running_mode=mode,
            min_hand_detection_confidence=0.3,
            min_hand_presence_confidence=0.3,
            min_tracking_confidence=0.3)
        self.detector = vision.HandLandmarker.create_from_options(options)
        self._ts = 0
        HANDS["ready"] = True
        print(f"[手部] MediaPipe HandLandmarker（21 点）| {model_path} | {running_mode}")

    def detect(self, image):
        """BGR 图 → {'box','points','conf'} 或 None；points 是 21 个 (x, y) 像素坐标。"""
        import cv2
        if image is None or image.size == 0:
            return None
        height, width = image.shape[:2]
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
        started = time.time()
        if self.mode_name == "video":
            self._ts += 33                     # 约 30fps 的时间戳，只要单调递增
            result = self.detector.detect_for_video(mp_image, self._ts)
        else:
            result = self.detector.detect(mp_image)
        cost = (time.time() - started) * 1000.0
        HANDS["cost_ms"] = cost
        if not result.hand_landmarks:
            HANDS["last"] = None
            return None
        hand = result.hand_landmarks[0]        # 只发最靠前（最可信）的那只
        xs = [min(max(p.x, 0.0), 1.0) * width for p in hand]
        ys = [min(max(p.y, 0.0), 1.0) * height for p in hand]
        points = [[round(x, 1), round(y, 1)] for x, y in zip(xs, ys)]
        box = (min(xs), min(ys), max(xs), max(ys))
        conf = 1.0
        if getattr(result, "handedness", None):
            try:
                conf = float(result.handedness[0][0].score)
            except Exception:
                conf = 1.0
        HANDS["last"] = {"points": points, "box": box, "conf": conf}
        HANDS["hits"] += 1
        return {"box": box, "points": points, "conf": conf}
