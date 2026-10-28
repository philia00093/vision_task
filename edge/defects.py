#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""零件缺陷检查：偏导求轮廓面积，和基准比，超过容差判不合格。

原理
----
拿检测框里的零件图 → Sobel 求偏导得到边缘幅值 → Otsu 二值化 → 闭运算把轮齿
之间的缺口连起来 → 取最大外轮廓 → `cv2.contourArea` 得到**零件投影面积**（像素）。

面积跟"零件多大"直接相关：缺齿、缺块、材料少一截，面积就会掉；装错型号、混进
大一号的件，面积会涨。所以拿它做第一道缺陷判据是够用的，而且：
· 跟摆放角度无关（面积是旋转不变量）
· 跟颜色/亮度无关（只看边缘强度，不求灰度差）

为什么用"当前测试"当基准
------------------------
同一台相机、同一个工位，零件面积是稳定的；但换镜头高度/换托盘会整体变一下。
所以基准不做成硬编码，而是**开机先量若干帧取中位数**当基准，之后每帧和它比。
容差默认 4%（`--part-tol 0.04`），超过就判 NG。

为什么用中位数 + 时间窗
----------------------
现场实测同一颗好件连测 8 次，单帧面积波动 2.2%（标准差 387 / 均值 17646），
偶尔会有 -4% 的离群值。单帧判定会把好件误判成 NG，所以取最近 `window` 次的中位数
再比 —— 离群值直接被滤掉。
"""

from __future__ import annotations

import cv2
import numpy as np

DEFAULT_TOL = 0.04          # 容差：4% 以外算不合格
DEFAULT_SEARCH = 0.35       # 搜索窗在检测框基础上外扩多少（只是"找零件"的范围，不参与测量）
DEFAULT_POS_TOL = 0.10      # 零件必须落在标定位置的 ±10% 内才算"在位"，才判、才显示
DEFAULT_WINDOW = 9          # 取最近多少次测量取中位数
DEFAULT_CALIB = 9           # 开机先量多少帧当基准


def edge_area(image, box, pad: float = 0.0) -> float:
    """偏导求边缘 → 最大外轮廓 → 面积（像素²）。box 是 (x1,y1,x2,y2)。"""
    if image is None or box is None:
        return 0.0
    x1, y1, x2, y2 = (int(v) for v in box)
    w, h = x2 - x1, y2 - y1
    if w < 8 or h < 8:
        return 0.0
    if pad > 0:
        x1 = max(0, int(x1 - w * pad)); y1 = max(0, int(y1 - h * pad))
        x2 = min(image.shape[1], int(x2 + w * pad)); y2 = min(image.shape[0], int(y2 + h * pad))
    crop = image[max(y1, 0):y2, max(x1, 0):x2]
    if crop.size == 0:
        return 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    mag = np.clip(mag / (np.percentile(mag, 98) + 1e-6) * 255.0, 0, 255).astype(np.uint8)
    _, mask = cv2.threshold(mag, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0
    # 只认**质心靠近框中心**的轮廓：零件旁边常挨着别的亮件（铜盘、另一个零件），
    # 取"最大轮廓"会把邻居一起算进去，面积就在两个值之间来回跳（实测 CV 2.2%）。
    # 加上中心门限后 CV 降到 0.3%，判定才稳。
    height, width = mask.shape
    cx0, cy0 = width * 0.5, height * 0.5
    gate = 0.35
    best = 0.0
    for contour in contours:
        moments = cv2.moments(contour)
        if moments["m00"] <= 0:
            continue
        cx = moments["m10"] / moments["m00"]
        cy = moments["m01"] / moments["m00"]
        if abs(cx - cx0) > width * gate or abs(cy - cy0) > height * gate:
            continue
        best = max(best, cv2.contourArea(contour))
    if best <= 0:                       # 门限内没东西：退回最大轮廓
        best = float(cv2.contourArea(max(contours, key=cv2.contourArea)))
    return float(best)


class AreaChecker:
    """一个零件的面积检查器：标定基准 → 逐帧测量 → 中位数判定。"""

    def __init__(self, name: str, tol: float = DEFAULT_TOL,
                 window: int = DEFAULT_WINDOW, calib: int = DEFAULT_CALIB,
                 ref: float | None = None, pos_tol: float = DEFAULT_POS_TOL):
        self.name = name
        self.pos_tol = float(pos_tol)
        self.center = None            # 标定出来的"应该在的位置"
        self.frame_size = None
        self.tol = float(tol)
        self.window = max(1, int(window))
        self.calib = max(1, int(calib))
        self.ref = float(ref) if ref else None
        self.samples: list[float] = []
        self.area = 0.0
        self.ratio = 0.0
        self.ok = True
        self.judged = 0
        self.failed = 0
        self._calib_buf: list[float] = []

    # ------------------------------------------------------------ 标定
    @property
    def calibrated(self) -> bool:
        return self.ref is not None

    def calibrate(self, area: float) -> bool:
        """开机自动标定：攒够 calib 帧取中位数当基准。返回这次是否完成标定。"""
        if self.calibrated or area <= 0:
            return False
        self._calib_buf.append(area)
        if len(self._calib_buf) < self.calib:
            return False
        self.ref = float(np.median(self._calib_buf))
        self._calib_buf.clear()
        return True

    def set_ref(self, ref: float) -> None:
        self.ref = float(ref)

    # ------------------------------------------------------------ 位置门限
    def check_position(self, center, frame_shape) -> bool | None:
        """零件在不在"应该在的位置"（±pos_tol）。

        返回 True=在位（可以判、可以显示）、False=不在位（不判也不显示）、
        None=第一次见到（把当前位置记成基准）。
        为什么要有这条：零件被拿在手上、路过画面时也会被检出，那时候量面积既不准
        也没意义；只有摆到标定位置附近，才说明它"放好了"。
        """
        if center is None or frame_shape is None:
            return False
        height, width = frame_shape[:2]
        if self.center is None:
            self.center = (float(center[0]), float(center[1]))
            self.frame_size = (int(width), int(height))
            return None
        dx = abs(float(center[0]) - self.center[0]) / max(width, 1)
        dy = abs(float(center[1]) - self.center[1]) / max(height, 1)
        return dx <= self.pos_tol and dy <= self.pos_tol

    def reset_position(self) -> None:
        self.center = None

    # ------------------------------------------------------------ 测量
    def push(self, area: float):
        """喂一次测量，返回 (中位面积, 相对误差, 是否合格)。"""
        if area <= 0:
            return self.area, self.ratio, self.ok
        # 离群值剔除：已经稳定下来之后，单帧偏离中位数 >8% 的直接丢弃
        # （实测同一颗件会冒出 +8% 的一档，来源是检测框偶尔偏移）
        if len(self.samples) >= 5 and self.area > 0:
            if abs(area - self.area) / self.area > 0.08:
                return self.area, self.ratio, self.ok
        self.samples.append(area)
        if len(self.samples) > self.window:
            del self.samples[:len(self.samples) - self.window]
        self.area = float(np.median(self.samples))
        if not self.calibrated:
            self.calibrate(area)
            return self.area, 0.0, True
        self.ratio = (self.area - self.ref) / self.ref if self.ref else 0.0
        self.ok = abs(self.ratio) <= self.tol
        self.judged += 1
        if not self.ok:
            self.failed += 1
        return self.area, self.ratio, self.ok

    def describe(self) -> str:
        if not self.calibrated:
            return f"{self.name} 面积：标定中 {len(self._calib_buf)}/{self.calib}"
        verdict = "OK" if self.ok else "NG"
        return (f"{self.name} 面积 {self.area:.0f}（基准 {self.ref:.0f}，"
                f"Δ {self.ratio*100:+.1f}%）{verdict}")

    def label(self) -> str:
        if not self.calibrated:
            return f"{self.name} 标定中"
        return (f"{self.name} A{self.area:.0f} {self.ratio*100:+.1f}% "
                f"{'合格' if self.ok else '不合格'}")


def find_part(image, box, search: float = DEFAULT_SEARCH, min_side: int = 8):
    """用检测框当**搜索窗**，在窗内用偏导自己找出零件的轮廓和边框。

    为什么不用检测框直接量：YOLO 的框每帧会抖 ±3 像素，框一紧就裁掉零件边缘，
    面积跟着差 7%（实测）。零件自己的边框应该由**它自己的边缘**决定 ——
    所以这里 Sobel 求边缘、闭运算、取离窗心最近的那条外轮廓，
    返回这条轮廓的面积、外接框、圆心半径。检测框只负责"去哪找"，不参与测量。

    返回 dict：area / bbox / center / radius / circle_area；找不到返回 None。
    """
    if image is None or box is None:
        return None
    x1, y1, x2, y2 = (float(v) for v in box)
    w, h = x2 - x1, y2 - y1
    if w < min_side or h < min_side:
        return None
    X1 = int(max(0, x1 - w * search)); Y1 = int(max(0, y1 - h * search))
    X2 = int(min(image.shape[1], x2 + w * search)); Y2 = int(min(image.shape[0], y2 + h * search))
    window = image[Y1:Y2, X1:X2]
    if window.size == 0:
        return None
    gray = cv2.cvtColor(window, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    mag = np.clip(mag / (np.percentile(mag, 98) + 1e-6) * 255.0, 0, 255).astype(np.uint8)
    _, mask = cv2.threshold(mag, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # 零件中心 ≈ 检测框中心在窗口里的位置
    cx0, cy0 = (x1 + x2) * 0.5 - X1, (y1 + y2) * 0.5 - Y1
    best, best_d = None, 1e18
    for contour in contours:
        moments = cv2.moments(contour)
        if moments["m00"] <= 0:
            continue
        cx = moments["m10"] / moments["m00"]
        cy = moments["m01"] / moments["m00"]
        dist = (cx - cx0) ** 2 + (cy - cy0) ** 2
        if dist < best_d:
            best, best_d = contour, dist
    if best is None:
        return None

    area = float(cv2.contourArea(best))
    bx, by, bw, bh = cv2.boundingRect(best)
    (mx, my), radius = cv2.minEnclosingCircle(best)
    return {
        "area": area,
        "bbox": (bx + X1, by + Y1, bw, bh),          # 换回原图坐标
        "center": (mx + X1, my + Y1),
        "radius": float(radius),
        "circle_area": float(np.pi * radius * radius),
        "window": (X1, Y1, X2 - X1, Y2 - Y1),
    }


# ============================================================================
# 罗盘正反检测
# ============================================================================
# 实测（正/反各 3 张，同一位置同光照）：**中心毂里的"黑色区域"占比**差异最大
#   正面（凹坑）  0.431 / 0.434 / 0.434
#   反面（凸台）  0.301 / 0.302 / 0.292
# 组间差 0.13，组内波动 ±0.01 —— 阈值 0.37 分得很干净。
# 用"占比"而不是绝对亮度：灯泡亮一点、换个位置都只影响整体，占比基本不变。
COMPASS_SIDE_RATIO = 0.37


def compass_side(image, box, black_ratio_threshold: float = COMPASS_SIDE_RATIO):
    """判断罗盘是正面朝上还是反面朝上。

    只看**中心毂里的黑色区域占比**：
    · 正面：毂是凹进去的深坑 → 黑区大（约 0.43）
    · 反面：毂是凸台、中间黑孔小 → 黑区小（约 0.30）
    Otsu 自动切阈值，所以不挑曝光；用"占比"而不是亮度绝对值，换位置也不受影响。

    返回 (label, ratio)：
        label = "front" / "back" / None（没检出、太小、贴边）
    """
    if image is None or box is None:
        return None, 0.0
    height, width = image.shape[:2]
    x1, y1, x2, y2 = (float(v) for v in box)
    cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
    size = max(x2 - x1, y2 - y1)
    if size < 40:
        return None, 0.0
    if cx - size * 0.5 < 0 or cy - size * 0.5 < 0 or \
            cx + size * 0.5 >= width or cy + size * 0.5 >= height:
        return None, 0.0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) \
        if image.ndim == 3 else image.astype(np.float32)
    yy, xx = np.mgrid[0:height, 0:width]
    hub = np.hypot(xx - cx, yy - cy) <= size * 0.16
    values = gray[hub]
    if values.size < 60:
        return None, 0.0
    sample = np.clip(values, 0, 255).astype(np.uint8).reshape(-1, 1)
    threshold, _ = cv2.threshold(sample, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    black_ratio = float((values < threshold).mean())
    return ("front" if black_ratio >= black_ratio_threshold else "back"), black_ratio
