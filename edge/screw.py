#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""螺丝/支撑杆拧紧检测：数**转了多少钱圈**。

物理依据
--------
待装的支撑杆是**偏心**的（杆头不在旋转轴上）。把它拧进去时，杆头绕轴心画圆：

    杆头角度 θ 每转一圈变化 360°，累计 |Δθ| / 360 = 转了几圈

所以只要每帧求出"杆头相对轴心的角度"，把相邻帧的角度差 unwrap 累加，就是圈数。
这和"面积/灰度"不同，对光照不敏感，只跟几何位置有关。

三个必须处理的坑
----------------
1. **角度折叠**：相邻帧角度差超过 180° 会被当成"反着转"。这里设了 `max_step_deg`
   （默认 90°），超过就丢弃这一帧，不累积。
2. **丢帧/遮挡**：手、螺丝刀挡一下，ROI 里找不到杆头。连续丢就暂停累积，
   不硬补 —— 宁可少数一圈，也不编造。
3. **什么时候算"拧完"**：不是看圈数到没到，而是看**手停没停**。
   连续 `stop_seconds`（默认 0.8s）没有明显转动，就判定动作结束，
   再看累计圈数够不够 4.5 圈 —— 不够就报错。

用法
----
    按屏上「开始装配支撑杆」按钮 → start()
    每帧喂 ROI 图 → update()
    自动停下后读 state/turns 显示"合格 / 不合格（只转了 X 圈）"
"""

from __future__ import annotations

import math
import time

import cv2
import numpy as np

DEFAULT_MIN_TURNS = 4.5         # 至少 4 圈半
DEFAULT_TARGET_MAX = 5.0        # 显示用的"目标上限"
DEFAULT_STOP_SECONDS = 0.8      # 停手多久算拧完
DEFAULT_MAX_STEP_DEG = 90.0     # 单帧最大可信转角（超过视为丢帧/折叠）
DEFAULT_MIN_MOVE_DEG = 2.0      # 小于这个角度算"没动"

IDLE, RUNNING, OK, NG = "idle", "running", "ok", "ng"


def tip_angle(roi, center=None):
    """在一小块图里找"偏心标记点"，返回 (角度°, 置信度)。

    关键：角度要相对**旋转轴**算，而不是相对零件质心 —— 拧的时候轴不动、偏心点绕它转。
    所以 `center` 默认取 ROI 中心（ROI 是按零件框+余量裁的，中心≈轴心），
    然后找最亮的那块（杆头/偏心块），用它的质心相对轴心的角度：

        θ = atan2(marker_y - axis_y, marker_x - axis_x)

    置信度 = 亮块面积占比，太小说明这一帧没看清（被手/螺丝刀挡住）。
    """
    if roi is None or roi.size == 0:
        return None, 0.0
    height, width = roi.shape[:2]
    ax, ay = (width * 0.5, height * 0.5) if center is None else center
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, 0.0
    contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(contour))
    conf = area / float(width * height)
    if conf < 0.015:                       # 亮块太小 = 没看清，别当数
        return None, conf
    moments = cv2.moments(contour)
    if moments["m00"] <= 0:
        return None, conf
    mx, my = moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]
    radius = math.hypot(mx - ax, my - ay)
    if radius < 0.08 * min(width, height):  # 偏心点几乎在轴上：角度没意义
        return None, conf
    return math.degrees(math.atan2(my - ay, mx - ax)), conf


class ScrewTurns:
    """一次"拧进去"的圈数统计。"""

    def __init__(self, min_turns: float = DEFAULT_MIN_TURNS,
                 target_max: float = DEFAULT_TARGET_MAX,
                 stop_seconds: float = DEFAULT_STOP_SECONDS,
                 max_step_deg: float = DEFAULT_MAX_STEP_DEG):
        self.min_turns = float(min_turns)
        self.target_max = float(target_max)
        self.stop_seconds = float(stop_seconds)
        self.max_step_deg = float(max_step_deg)
        self.state = IDLE
        self.turns = 0.0
        self.angle = None
        self.conf = 0.0
        self.lost = 0
        self.started_at = 0.0
        self.last_move = 0.0
        self._last_angle = None
        self._accum = 0.0

    # ------------------------------------------------------------ 控制
    def start(self, now: float | None = None) -> None:
        self.state = RUNNING
        self.turns = 0.0
        self.angle = None
        self.conf = 0.0
        self.lost = 0
        self._accum = 0.0
        self._last_angle = None
        self.started_at = time.time() if now is None else now
        self.last_move = self.started_at

    def reset(self) -> None:
        self.state = IDLE
        self.turns = 0.0
        self._accum = 0.0
        self._last_angle = None

    @property
    def running(self) -> bool:
        return self.state == RUNNING

    # ------------------------------------------------------------ 每帧
    def update(self, roi, now: float | None = None) -> None:
        """喂一帧 ROI（BGR）。内部完成角度解卷绕、累积、停手判定。"""
        now = time.time() if now is None else now
        if self.state != RUNNING:
            return
        angle, conf = tip_angle(roi)
        self.conf = conf
        if angle is None:
            self.lost += 1
            return                            # 跟丢只是暂停累积，不结算（等按钮）
        self.lost = 0
        if self._last_angle is None:
            self._last_angle = angle
            self.angle = angle
            return
        delta = (angle - self._last_angle + 180.0) % 360.0 - 180.0   # 折到 (-180,180]
        self._last_angle = angle
        self.angle = angle
        if abs(delta) > self.max_step_deg:    # 这一帧不可信：别累积
            return
        if abs(delta) >= DEFAULT_MIN_MOVE_DEG:
            self._accum += delta
            self.turns = abs(self._accum) / 360.0
            self.last_move = now
        # 不再"停手自动结束"：实测拧的时候和停下来在画面上差别不大，
        # 自动判停会误结算（刚按下就结束）。现在只认按钮：第二次按才结算。

    def finish(self, now: float | None = None) -> None:
        """外部要求结束（第二次按按钮）：立刻结算圈数并判合格/不合格。"""
        self._finish(now)

    def _finish(self, now: float | None = None) -> None:
        now = time.time() if now is None else now
        if self.state != RUNNING:
            return
        self.turns = abs(self._accum) / 360.0
        self.state = OK if self.turns >= self.min_turns else NG

    def describe(self) -> str:
        if self.state == RUNNING:
            return f"支撑杆 {self.turns:.2f} 圈（目标 ≥{self.min_turns:g}）"
        if self.state == OK:
            return f"支撑杆 {self.turns:.2f} 圈 合格"
        if self.state == NG:
            return f"支撑杆只转 {self.turns:.2f} 圈 不合格"
        return "支撑杆 未开始"
