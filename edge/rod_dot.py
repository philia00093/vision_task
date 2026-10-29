#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""红点偏心旋转计圈：杆子转了多大角度。

思路（为什么这次能成）
----------------------
杆子本身是圆的、没有纹理，之前用梯度直方图、极坐标相位相关、两端连线都试过，
实测杆子转动时画面几乎不变（80 帧互相关 0.0°，置信度 30）—— 从上面看下去，
一个光滑圆柱绕自己轴转，本来就看不出来。

现在杆子上贴了**红点**：点在离轴心一段距离处，杆子一转，点就在画一个圆。
所以只要：
  1. 找到红点（HSV 红色阈值，两段色相 0~12 / 165~180）
  2. 估出它在绕的那根轴心（用最近若干帧的点做最小二乘拟合圆心，比用检测框稳）
  3. 算点相对轴心的角度，逐帧**解缠**累加
就能得到杆子累计转过的角度，除以 360 就是圈数。

鲁棒性
------
· 丢帧/遮挡：找不到红点就这一帧不算，角度停在那里，下次接上继续累加
· 跳变：单帧 |Δ角度| > jump_deg（默认 120°）视为认错点，不计入
· 轴心：点数不够（刚开始转）时用 ROI 中心，够了就换成拟合圆心
"""

from __future__ import annotations

import math
import time
from collections import deque

import cv2
import numpy as np

# 阈值放宽一点：小点在转动时会拖影、变暗，S/V 卡太死会一帧都检不到
RED_RANGES = (((0, 70, 45), (14, 255, 255)),
              ((160, 70, 45), (179, 255, 255)))


def red_mask(image: np.ndarray) -> np.ndarray:
    """红色像素掩膜（两种色相区间合起来，抗光照用 S/V 下限）。"""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(RED_RANGES[0][0]), np.array(RED_RANGES[0][1]))
    mask |= cv2.inRange(hsv, np.array(RED_RANGES[1][0]), np.array(RED_RANGES[1][1]))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    return mask


class RodDot:
    """红点偏心旋转计数。"""

    def __init__(self, min_radius: float = 3.0, max_radius: float = 40.0,
                 jump_deg: float = 120.0, fit_points: int = 40,
                 gate_px: float = 80.0, max_speed_rev: float = 4.0):
        self.min_radius = float(min_radius)
        self.max_radius = float(max_radius)
        self.jump_rad = math.radians(abs(float(jump_deg)))
        self.max_speed = abs(float(max_speed_rev)) * 2.0 * math.pi   # rad/s 上限
        self.gate_px = float(gate_px)
        self.points: deque = deque(maxlen=max(8, int(fit_points)))
        self.seed = None                # 外部给的轴心初值（铜盘框心，比 ROI 中心准）
        self.dot = None                 # 上一帧红点（全画面坐标）
        self.center = None              # 旋转轴心（全画面坐标）
        self.last_angle = None
        self.last_time = None
        self.speed = 0.0                # 最近一次的角度速度（deg/s），调试用
        self.total_rad = 0.0            # 解缠后的累计角（有符号）
        self.lost = 0                   # 连续找不到的帧数
        self.hits = 0
        self.skipped = 0                # 因跳变被丢掉的帧数

    # ------------------------------------------------------------ 内部
    def reset(self) -> None:
        self.points.clear()
        self.dot = None
        self.center = None
        self.last_angle = None
        self.last_time = None
        self.speed = 0.0
        self.total_rad = 0.0
        self.lost = 0
        self.hits = 0

    def _fit_center(self, fallback):
        """用最近的点做 Kasa 圆拟合，点数不够/拟合离谱就退回 fallback。"""
        if len(self.points) < 12:
            return fallback
        pts = np.array(self.points, dtype=np.float64)
        x, y = pts[:, 0], pts[:, 1]
        A = np.column_stack([2.0 * x, 2.0 * y, np.ones(len(x))])
        b = x * x + y * y
        try:
            sol, *_ = np.linalg.lstsq(A, b, rcond=None)
        except Exception:
            return fallback
        cx, cy, c = float(sol[0]), float(sol[1]), float(sol[2])
        r2 = c + cx * cx + cy * cy
        if r2 <= 1.0:
            return fallback
        radius = math.sqrt(r2)
        # 偏心距离不可能比杆子半径还大：8~160 px 之外当拟合失败
        if not (8.0 <= radius <= 160.0):
            return fallback
        return (cx, cy)

    @staticmethod
    def _blobs(mask: np.ndarray, min_r: float, max_r: float):
        count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
        out = []
        for index in range(1, count):
            area = float(stats[index, cv2.CC_STAT_AREA])
            radius = math.sqrt(area / math.pi)
            if min_r <= radius <= max_r:
                out.append((radius, float(centroids[index][0]), float(centroids[index][1])))
        out.sort(key=lambda item: -item[0])
        return out

    # ------------------------------------------------------------ 对外
    def update(self, image: np.ndarray, roi=None, seed_center=None) -> dict:
        """喂一帧（BGR），返回 {'ok','dot','center','angle_deg','total_deg','turns'}。

        roi 是 (x, y, w, h)（全画面坐标）；给了就只看这一块，省算力也更稳。
        """
        if image is None or image.size == 0:
            return {"ok": False}
        ox = oy = 0
        crop = image
        if roi is not None:
            rx, ry, rw, rh = (int(v) for v in roi)
            rx = max(0, rx); ry = max(0, ry)
            rw = min(rw, image.shape[1] - rx); rh = min(rh, image.shape[0] - ry)
            if rw < 20 or rh < 20:
                return {"ok": False}
            crop = image[ry:ry + rh, rx:rx + rw]
            ox, oy = rx, ry

        blobs = self._blobs(red_mask(crop), self.min_radius, self.max_radius)
        if not blobs:
            self.lost += 1
            return {"ok": False, "total_deg": math.degrees(self.total_rad),
                    "turns": abs(math.degrees(self.total_rad)) / 360.0,
                    "lost": self.lost}

        # 选点：优先离上一帧最近的那个（防串到别的红东西上），否则取最大的
        chosen = blobs[0]
        if self.dot is not None:
            px, py = self.dot[0] - ox, self.dot[1] - oy
            best = None
            for blob in blobs:
                d = ((blob[1] - px) ** 2 + (blob[2] - py) ** 2) ** 0.5
                if best is None or d < best[0]:
                    best = (d, blob)
            if best is not None and best[0] <= self.gate_px:
                chosen = best[1]

        radius, bx, by = chosen
        self.dot = (bx + ox, by + oy)
        self.points.append(self.dot)
        self.hits += 1
        self.lost = 0

        if seed_center is not None:
            self.seed = (float(seed_center[0]), float(seed_center[1]))
        fallback = self.seed or (ox + crop.shape[1] / 2.0, oy + crop.shape[0] / 2.0)
        fitted = self._fit_center(fallback)
        if fitted is not None and self.seed is not None:
            # 拟合出来的圆心离"盘心"太远，说明点还没转开/认到别的点，别跟着跑
            if ((fitted[0] - self.seed[0]) ** 2 + (fitted[1] - self.seed[1]) ** 2) ** 0.5 > 60:
                fitted = None
        self.center = fitted or fallback
        cx, cy = self.center
        angle = math.atan2(self.dot[1] - cy, self.dot[0] - cx)
        now = time.time()
        dt = None if self.last_time is None else max(1e-3, now - self.last_time)
        if self.last_angle is not None and dt is not None:
            delta = angle - self.last_angle
            while delta > math.pi:
                delta -= 2.0 * math.pi
            while delta < -math.pi:
                delta += 2.0 * math.pi
            # 门限跟着帧间隔走：转得再快，只要不超过 max_speed 就认
            limit = min(self.jump_rad, self.max_speed * dt)
            if abs(delta) <= max(limit, math.radians(8.0)):
                self.total_rad += delta
                self.speed = math.degrees(delta) / dt
            else:
                self.skipped += 1
        self.last_angle = angle
        self.last_time = now
        return {"ok": True, "dot": self.dot, "center": self.center,
                "angle_deg": math.degrees(angle),
                "total_deg": math.degrees(self.total_rad),
                "turns": abs(math.degrees(self.total_rad)) / 360.0,
                "radius": radius, "lost": 0}

    @property
    def total_deg(self) -> float:
        return math.degrees(self.total_rad)

    @property
    def turns(self) -> float:
        return abs(math.degrees(self.total_rad)) / 360.0

    def describe(self) -> str:
        return (f"红点 {self.hits} 帧 | 累计 {math.degrees(self.total_rad):+.0f}° "
                f"= {self.turns:.2f} 圈 | 丢失 {self.lost} | 跳变丢弃 {self.skipped}"
                f" | 角速度 {self.speed:+.0f}°/s")
