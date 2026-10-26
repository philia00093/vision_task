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
                 ref: float | None = None):
        self.name = name
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

    # ------------------------------------------------------------ 测量
    def push(self, area: float):
        """喂一次测量，返回 (中位面积, 相对误差, 是否合格)。"""
        if area <= 0:
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
