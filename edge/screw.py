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
DEFAULT_MIN_MOVE_DEG = 0.8      # 小于这个角度算"没动"（慢拧每帧只有 1~2°）

IDLE, RUNNING, OK, NG = "idle", "running", "ok", "ng"


DEFAULT_BINS = 180           # 梯度方向直方图分箱：2θ 上 0~360 → 每箱 2°，θ 上 1°
                             # 36 箱时分辨率只有 5°，每帧转 3° 会被量化成 0（实测圈数恒为 0）


def grad_hist(roi, bins: int = DEFAULT_BINS):
    """ROI 的**梯度方向直方图**（按幅值加权）。

    为什么用 2θ：无向边缘的梯度方向每 180° 就重复，直接统计 θ 会把一条直线
    的两个方向拆成两堆。用 2θ 表示后，旋转 φ 对应直方图整体平移 2φ，
    圆周互相关一波就能求出来 —— 不需要找"亮斑/顶点"这种脆弱的特征。
    """
    if roi is None or roi.size == 0:
        return None
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi
    gray = cv2.GaussianBlur(gray.astype(np.float32), (5, 5), 0)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    total = float(mag.sum())
    if total <= 1e-3:                      # 一片死平，没有梯度
        return None
    # 只保留中心圆形区域（去掉外圈轮缘）：圆盘的边缘是各向同性的，会把直方图摊平，
    # 真正随转动变化的是中心那根杆。实测不遮轮缘时置信度只有 1.1~1.4。
    height, width = mag.shape
    yy, xx = np.ogrid[:height, :width]
    cy, cx = height * 0.5, width * 0.5
    radius = min(height, width) * 0.42
    inner = ((xx - cx) ** 2 + (yy - cy) ** 2) <= radius ** 2
    mag = mag * inner
    angle = np.arctan2(gy, gx)             # -π ~ π
    doubled = np.mod(2.0 * angle, 2.0 * np.pi)
    idx = (doubled * (bins / (2.0 * np.pi))).astype(np.int32) % bins
    # 幅值平方加权：杆这类强边缘占比更大，弱纹理（反光/噪声）被压下去
    weights = (mag ** 2).ravel()
    hist = np.bincount(idx.ravel(), weights=weights, minlength=bins).astype(np.float64)
    hist /= (hist.sum() + 1e-9)
    return hist


def hist_shift(prev, cur, bins: int = DEFAULT_BINS):
    """两个方向直方图的圆周互相关 → (旋转角°, 置信度)。

    旋转 φ 会让 2θ 直方图平移 2φ，所以互相关峰在第 δ 个箱时 φ = δ*180/bins 度。
    置信度 = 峰值 / 均值，太低说明画面没有明显方向结构（被挡住/没零件）。
    """
    if prev is None or cur is None or len(prev) != len(cur):
        return None, 0.0
    score = []
    for shift in range(bins):
        rolled = np.roll(cur, shift)
        score.append(float(np.dot(prev, rolled)))
    score = np.asarray(score)
    peak = int(np.argmax(score))
    mean = float(score.mean()) + 1e-9
    conf = float(score[peak]) / mean
    # 抛物线插值：用峰值左右两个点把角度细化到亚箱精度（不插值会被量化误差吃掉）
    left, right = score[(peak - 1) % bins], score[(peak + 1) % bins]
    denom = (left - 2.0 * score[peak] + right)
    offset = 0.0 if abs(denom) < 1e-12 else 0.5 * (left - right) / denom
    offset = max(-0.5, min(0.5, offset))
    shift = peak + offset
    if shift > bins / 2.0:                 # 取最近的旋转方向（±90° 内）
        shift -= bins
    degrees = shift * (180.0 / bins)       # 2θ 每箱 2° → θ 上 1°
    return degrees, conf


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
        self._last_hist = None
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
        self._last_hist = None
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
        # 梯度方向直方图的帧间平移 = 这一帧转过的角度
        hist = grad_hist(roi)
        conf = 0.0
        delta = None
        if hist is not None and self._last_hist is not None:
            delta, conf = hist_shift(self._last_hist, hist)
            if conf < 1.35:                   # 方向结构不明显：这一帧不算
                                              # （实测静态画面 conf≈1.2，真转动时 >10）
                delta = None
        if hist is not None:
            self._last_hist = hist
        self.conf = conf
        if delta is None:
            self.lost += 1
            return                            # 跟丢只是暂停累积，不结算（等按钮）
        self.lost = 0
        self.angle = float(delta)
        if abs(delta) > self.max_step_deg:    # 这一帧不可信：别累积
            return
        if abs(delta) >= DEFAULT_MIN_MOVE_DEG:
            # 累积**转动量**（绝对值）：拧紧只有一个方向，梯度直方图偶尔会给出反号，
            # 直接累加就会出现"圈数往下掉"（实测 0.36 → 变小）。取绝对值就单调增。
            self._accum += abs(delta)
            self.turns = self._accum / 360.0
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
