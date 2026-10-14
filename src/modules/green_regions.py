"""绿色料盘识别（纯 OpenCV，不训练、不依赖模型）。

原理：绿色料盘在 HSV 空间里非常干净（实测 H≈75、S≈120、V≈255），
所以直接用 inRange 阈值分割 + 形态学去噪 + 外接矩形，就能稳定拿到两块料盘。
再按 x 中心排序：左边那块 = 料盘区，右边那块 = 装配区。

为什么用颜色而不是训练检测模型：
    · 料盘是纯色工装，颜色比形状可靠得多，零样本、零训练；
    · 车间换光照时只需调 --green-lo/--green-hi 两个阈值，不用重新标注；
    · 同时给出模型兜底路径，哪天颜色法不够用了可以随时切回去。

注意：本文件的 detect 只做"识别"，参数与阈值都可命令行覆盖，方便现场调。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

# 实测：绿板 H≈75-78 S≈120 V≈224-255。下限留足余量以覆盖阴影里的暗绿，
# 同时把 S/V 下限抬高，避免深色地板、深蓝布料被判成绿色。
DEFAULT_LO = (35, 80, 100)
DEFAULT_HI = (95, 255, 255)

# 太小的一律当噪点丢掉（画面 789x592 时实测两块板面积 7 万 / 8.8 万像素）
MIN_AREA_RATIO = 0.004          # 相对整幅画面的最小面积
MIN_FILL_RATIO = 0.55           # 外接矩形填充率下限，滤掉细长杂散色块


@dataclass
class GreenRegion:
    name: str                       # tray_area（左）/ assembly_area（右）
    box: tuple[int, int, int, int]  # x, y, w, h
    area: float
    fill: float                     # 轮廓面积 / 外接矩形面积

    @property
    def xyxy(self) -> tuple[int, int, int, int]:
        x, y, w, h = self.box
        return x, y, x + w, y + h

    @property
    def center(self) -> tuple[float, float]:
        x, y, w, h = self.box
        return x + w / 2.0, y + h / 2.0

    def contains(self, px: float, py: float) -> bool:
        x1, y1, x2, y2 = self.xyxy
        return x1 <= px <= x2 and y1 <= py <= y2


@dataclass
class GreenConfig:
    """颜色分割参数：全部可在命令行覆盖，方便现场对着画面调。"""
    lo: tuple[int, int, int] = DEFAULT_LO
    hi: tuple[int, int, int] = DEFAULT_HI
    morph: int = 7                  # 形态学核大小（0=不做）
    open_iters: int = 2             # 先开后闭：去噪点 → 补空洞
    close_iters: int = 3
    min_area_ratio: float = MIN_AREA_RATIO
    min_fill_ratio: float = MIN_FILL_RATIO
    merge_gap: int = 0              # >0 时把水平间距小于该值的连通块合并


def _morph_kernel(size: int):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)) if size > 0 else None


def green_mask(frame, config: GreenConfig, soft: bool = False):
    """返回绿色区域的二值掩码（soft=True 时返回 0/1 浮点掩码，用于半透明填充）。"""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(config.lo, np.uint8), np.array(config.hi, np.uint8))
    kernel = _morph_kernel(config.morph)
    if kernel is not None:
        if config.open_iters:
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel,
                                    iterations=config.open_iters)
        if config.close_iters:
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel,
                                    iterations=config.close_iters)
    return mask.astype(np.float32) / 255.0 if soft else mask


def _merge_boxes(boxes: list[tuple[int, int, int, int]], gap: int):
    """把水平相邻（间距 <= gap）的框合并，应对料盘被反光/手分成两块的情况。"""
    if gap <= 0 or len(boxes) < 2:
        return boxes
    boxes = sorted(boxes, key=lambda b: b[0])
    merged = [list(boxes[0])]
    for x, y, w, h in boxes[1:]:
        last = merged[-1]
        if x - (last[0] + last[2]) <= gap:
            right = max(last[0] + last[2], x + w)
            top, bottom = min(last[1], y), max(last[1] + last[3], y + h)
            last[0], last[1], last[2], last[3] = last[0], top, right - last[0], bottom - top
        else:
            merged.append([x, y, w, h])
    return [tuple(b) for b in merged]


def detect_green_regions(frame, config: GreenConfig | None = None,
                         min_count: int = 2) -> tuple[list[GreenRegion], str | None]:
    """识别绿色料盘，按 x 排序后命名为 tray_area（左）/ assembly_area（右）。

    返回 (区域列表, 告警文本)。区域少于 min_count 时告警但照样返回已找到的，
    让上层决定是"只用找到的那块"还是"退回几何兜底"。
    """
    config = config or GreenConfig()
    height, width = frame.shape[:2]
    mask = green_mask(frame, config)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = config.min_area_ratio * width * height
    candidates: list[tuple[tuple[int, int, int, int], float, float]] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if w <= 0 or h <= 0:
            continue
        fill = area / float(w * h)
        if fill < config.min_fill_ratio:
            continue
        candidates.append(((x, y, w, h), area, fill))

    if config.merge_gap:
        boxes = [c[0] for c in candidates]
        merged = _merge_boxes(boxes, config.merge_gap)
        # 每个框用自身外接矩形面积 × 平均填充率估算轮廓面积
        mean_fill = (sum(c[2] for c in candidates) / len(candidates)) if candidates else 1.0
        candidates = [(box, float(box[2] * box[3]) * mean_fill, mean_fill)
                      for box in merged]

    candidates.sort(key=lambda c: c[0][0])          # 左 → 右
    names = ["tray_area", "assembly_area", "extra_1", "extra_2", "extra_3"]
    regions = [GreenRegion(names[i] if i < len(names) else f"green_{i}",
                           box, area, fill)
               for i, (box, area, fill) in enumerate(candidates)]

    warning = None
    if len(regions) < min_count:
        warning = (f"只识别到 {len(regions)} 块绿色料盘（期望 {min_count}）"
                   "：可调 --green-lo/--green-hi 或 --min-area")
    elif len(regions) > 2:
        warning = (f"识别到 {len(regions)} 块绿色区域，取最左两块作为料盘区/装配区；"
                   "可用 --min-area 提高面积阈值过滤小色块")
    return regions[:2] if len(regions) > 2 else regions, warning


def draw_green(frame, regions, mask=None, fill_alpha: float = 0.18):
    """调试视图：半透明填充 + 外框 + 面积/填充率文字。返回新图。"""
    canvas = frame.copy()
    if mask is not None:
        overlay = canvas.copy()
        overlay[mask > 0] = (0, 255, 0)
        canvas = cv2.addWeighted(overlay, fill_alpha, canvas, 1 - fill_alpha, 0)

    palette = {"tray_area": (200, 255, 0), "assembly_area": (0, 165, 255)}
    for region in regions:
        x1, y1, x2, y2 = region.xyxy
        color = palette.get(region.name, (0, 255, 255))
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        text = f"{region.name} {x2 - x1}x{y2 - y1} fill {region.fill:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(canvas, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(canvas, text, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 0), 1, cv2.LINE_AA)
    return canvas


def save_debug(frame, regions, mask, out_dir: str | Path, tag: str = "") -> Path:
    """把原图 / 掩码 / 标注图三张存下来，用于现场调阈值。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    annotated = draw_green(frame, regions, mask)
    mask_path = out_dir / f"green_mask{('_' + tag) if tag else ''}.png"
    view_path = out_dir / f"green_regions{('_' + tag) if tag else ''}.png"
    cv2.imwrite(str(mask_path), mask)
    cv2.imwrite(str(view_path), annotated)
    return view_path
