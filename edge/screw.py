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
DEFAULT_MIN_MOVE_DEG = 1.5      # 小于这个角度算"没动"（再小就是噪声）

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


def motion_angle(prev_gray, cur_gray, center=None,
                 inner: float = 0.12, outer: float = 0.98):
    """偏心运动方向：这一帧**动过的像素**按"离轴心的距离"加权后的圆周平均方向。

    为什么这样算：杆绕轴转，杆上每点的线速度 ∝ 半径，所以**偏心杆头动得最多**。
    把帧间差里动过的像素按半径加权，方向加权平均就指向"动得最厉害的那一头"。
    杆转多少，这个方向就转多少 —— 不需要颜色标记，也不依赖纹理统计。
    """
    if prev_gray is None or cur_gray is None or cur_gray.size == 0:
        return None, 0.0, 0.0
    height, width = cur_gray.shape[:2]
    cx, cy = (width * 0.5, height * 0.5) if center is None else center
    diff = np.abs(cur_gray.astype(np.float32) - prev_gray.astype(np.float32))
    diff = cv2.GaussianBlur(diff, (5, 5), 0)
    threshold = max(6.0, float(diff.mean()) + 2.0 * float(diff.std()))
    ys, xs = np.nonzero(diff > threshold)
    if xs.size < 12:
        return None, 0.0, 0.0
    dx = xs.astype(np.float32) - cx
    dy = ys.astype(np.float32) - cy
    radius = np.hypot(dx, dy)
    limit = min(width, height) * 0.5
    keep = (radius > inner * limit) & (radius < outer * limit)
    if np.count_nonzero(keep) < 12:
        return None, 0.0, 0.0
    weights = diff[ys, xs][keep] * radius[keep]          # 越远权重越大
    energy = float(diff[ys, xs][keep].mean())            # 平均变化强度（噪声约 2~5）
    angles = np.arctan2(dy[keep], dx[keep])
    sin_sum = float((weights * np.sin(angles)).sum())
    cos_sum = float((weights * np.cos(angles)).sum())
    if abs(sin_sum) < 1e-6 and abs(cos_sum) < 1e-6:
        return None, 0.0, 0.0
    angle = math.degrees(math.atan2(sin_sum, cos_sum))
    conf = float(np.hypot(sin_sum, cos_sum) / (weights.sum() + 1e-6))   # 0~1，越高越集中
    return angle, conf, energy


def find_tip_by_motion(prev_gray, cur_gray, center=None, min_frac: float = 0.25):
    """从帧间差里找"动得最厉害、且离轴心足够远"的那个点 = 偏心杆的顶点。

    只在没有顶点可跟时用一次（会话刚开始、或者跟丢了）。
    """
    if prev_gray is None or cur_gray is None:
        return None
    height, width = cur_gray.shape[:2]
    cx, cy = (width * 0.5, height * 0.5) if center is None else center
    diff = np.abs(cur_gray.astype(np.float32) - prev_gray.astype(np.float32))
    diff = cv2.GaussianBlur(diff, (5, 5), 0)
    radius = np.hypot(np.arange(width)[None, :] - cx, np.arange(height)[:, None] - cy)
    limit = min(width, height) * 0.5
    score = diff * radius
    score[radius < min_frac * limit] = 0               # 轴心附近不动，别选
    score[radius > 0.98 * limit] = 0                   # 盘外/边界不要
    if float(score.max()) < 40.0:
        return None
    y, x = np.unravel_index(int(np.argmax(score)), score.shape)
    return float(x), float(y)


def track_tip(prev_gray, cur_gray, tip):
    """LK 光流跟一个点。返回新位置或 None（跟丢）。"""
    if tip is None or prev_gray is None or cur_gray is None:
        return None
    points = np.array([[tip]], np.float32)
    moved, status, _err = cv2.calcOpticalFlowPyrLK(
        prev_gray, cur_gray, points, None,
        winSize=(21, 21), maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03))
    if moved is None or status is None or int(status[0][0]) != 1:
        return None
    return float(moved[0][0][0]), float(moved[0][0][1])


def marker_angle(roi, hsv_lo, hsv_hi, center=None):
    """在 ROI 里找一个**色标**，返回它相对 ROI 中心的角度。

    这是最稳的方式：标记点颜色独特、面积小、每帧质心都干净，
    角度只跟它在圆上的位置有关，不受托盘纹理/光照整体变化影响。
    """
    if roi is None or roi.size == 0:
        return None, 0.0
    height, width = roi.shape[:2]
    cx, cy = (width * 0.5, height * 0.5) if center is None else center
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(hsv_lo, np.uint8), np.array(hsv_hi, np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, 0.0
    area_max = float(width * height) or 1.0
    best = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area / area_max < 0.0008 or area / area_max > 0.15:
            continue
        moments = cv2.moments(contour)
        if moments["m00"] <= 0:
            continue
        mx, my = moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]
        dist = math.hypot(mx - cx, my - cy)
        if dist < 0.10 * min(width, height):      # 贴在轴心上就量不出角度
            continue
        if best is None or area > best[0]:
            best = (area, mx, my)
    if best is None:
        return None, 0.0
    _area, mx, my = best
    return math.degrees(math.atan2(my - cy, mx - cx)), 1.0


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
                 max_step_deg: float = DEFAULT_MAX_STEP_DEG,
                 marker_hsv=None):
        self.min_turns = float(min_turns)
        self.target_max = float(target_max)
        self.stop_seconds = float(stop_seconds)
        self.max_step_deg = float(max_step_deg)
        self.marker_hsv = marker_hsv            # (lo, hi)；给了就走色标，否则走偏心运动
        self.mode = "marker" if marker_hsv else "motion"
        self._prev_gray = None                  # 上一帧 ROI 灰度
        self._tip = None                        # 正在跟的顶点（偏心杆头）
        self._dir = 0                           # 旋转方向锁（+1/-1，0=还没定）
        self._signed = 0.0                      # 带符号累计（只用来定方向）
        self.motion_conf = 0.0
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
        self._prev_gray = None
        self._tip = None
        self._dir = 0
        self._signed = 0.0
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
        # ① 有贴纸色标就用它：直接给出"标记点相对旋转中心"的角度，最稳
        if self.marker_hsv is not None:
            lo, hi = self.marker_hsv
            angle, conf = marker_angle(roi, lo, hi)
            self.conf = conf
            if angle is None:
                self.lost += 1
                return
            self.lost = 0
            self.angle = float(angle)
            if self._last_angle is None:
                self._last_angle = self.angle
                return
            delta = (self.angle - self._last_angle + 180.0) % 360.0 - 180.0
            self._last_angle = self.angle
            if abs(delta) > self.max_step_deg:
                return
            if abs(delta) >= DEFAULT_MIN_MOVE_DEG:
                self._accum += abs(delta)
                self.turns = self._accum / 360.0
                self.last_move = now
            return

        # ② 没色标：**只跟顶点**（偏心杆头）
        #    杆头绕轴画圆 → 角度 = atan2(顶点 - 轴心)，相邻帧角度差解卷绕累计。
        #    之前用"整块运动的圆周平均"，噪声区域多、方向乱，圈数会飙；
        #    现在只维护一个点，噪声完全可控。
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi
        height, width = gray.shape[:2]
        cx, cy = width * 0.5, height * 0.5
        limit = min(width, height) * 0.5
        tip = None
        if self._tip is not None:
            tip = track_tip(self._prev_gray, gray, self._tip)
            if tip is not None:
                # 校验：顶点到轴心的半径应该基本不变（在圆周上）
                ratio = math.hypot(tip[0] - cx, tip[1] - cy) / max(
                    math.hypot(self._tip[0] - cx, self._tip[1] - cy), 1e-6)
                if not (0.75 <= ratio <= 1.33):
                    tip = None
        if tip is None:                              # 跟丢/刚启动：重新找
            tip = find_tip_by_motion(self._prev_gray, gray, (cx, cy))
            if tip is not None:
                self._last_angle = None              # 重新定基准，避免跳变被当成转动
        self._prev_gray = gray
        if tip is None:
            self.lost += 1
            return
        # 注意：位移要在**更新 _tip 之前**算，否则自己跟自己比恒为 0（刚踩过）
        moved_px = (0.0 if self._tip is None
                    else math.hypot(tip[0] - self._tip[0], tip[1] - self._tip[1]))
        self._tip = tip
        self.lost = 0
        radius = math.hypot(tip[0] - cx, tip[1] - cy)
        if radius < 0.15 * limit:                    # 太靠轴心，角度不准
            return
        # 顶点实际位移太小（亚像素抖动）就不算转动：光流在噪声上也会漂 0.5~1 px，
        # 不加这道门 300 帧静止画面能累出 0.3 圈。
        if moved_px < 1.2:
            return
        angle = math.degrees(math.atan2(tip[1] - cy, tip[0] - cx))
        self.angle = angle
        self.conf = 1.0
        if self._last_angle is None:
            self._last_angle = angle
            return
        delta = (angle - self._last_angle + 180.0) % 360.0 - 180.0
        self._last_angle = angle
        if abs(delta) > self.max_step_deg:
            return
        if abs(delta) < DEFAULT_MIN_MOVE_DEG:
            return
        self._signed += delta
        if self._dir == 0:
            if abs(self._signed) > 20.0:            # 累计够 20° 才认方向
                self._dir = 1 if self._signed > 0 else -1
            else:
                return                              # 方向没定之前不计数（噪声阶段）
        if delta * self._dir < 0 and abs(delta) < 30.0:
            return
        self._accum += abs(delta)
        self.turns = self._accum / 360.0
        self.last_move = now
        return

        # ②' （旧的梯度直方图兜底，保留但不再走）
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
