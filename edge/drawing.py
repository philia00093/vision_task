"""绘制部分：所有画面上的框、轨迹、文字标签、状态栏。

中文用 PIL 渲染后贴回（cv2.putText 画不出中文，会变成问号），结果按(文本,颜色,字号)缓存，实时循环里几乎零开销。"""

from __future__ import annotations

import numpy as np
import cv2
from pathlib import Path

from edge.constants import (CJK_FONTS, DEFAULT_LINE_WIDTH, LABEL_FONT_SIZE, STAGE_COLOR,
                            STATE_COLOR, TRAIL_COLOR)


def draw_overlay(image, text, height=30):
    """在 image 上就地画顶部状态栏。

    注意：为了不与标签定位混淆，本函数返回**状态栏高度**（int），不返回图像。
    需要图像的地方直接用传入的 image（它是就地修改的）。
    """
    cv2.rectangle(image, (0, 0), (image.shape[1], height), (0, 0, 0), -1)
    cv2.putText(image, text, (8, int(height * 0.72)), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 255, 255), 1, cv2.LINE_AA)
    return height

def load_cjk_font(size=LABEL_FONT_SIZE):
    """载入系统中文字体；不可用时返回 None（调用方退回 cv2 英文标签）。"""
    try:
        from PIL import ImageFont
    except ImportError:
        return None
    for path in CJK_FONTS:
        if Path(path).is_file():
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return None

_CJK_FONT_CACHE = {}

_LABEL_CACHE = {}

def _draw_ascii_label(image, text, x, y, color, pad=4, right=False):
    """没中文字体时的兜底：cv2.putText 画 ASCII 标签（中文变 "?"）。"""
    ascii_text = text.encode("ascii", "replace").decode("ascii").replace("?", "?")
    scale, thick = 0.45, 1
    (tw, th), base = cv2.getTextSize(ascii_text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    x1 = int(x - tw - pad if right else x)
    y0 = int(y)
    cv2.rectangle(image, (x1, max(y0 - th - pad, 0)),
                  (x1 + tw + pad * 2, min(y0 + base + pad, image.shape[0] - 1)),
                  tuple(int(c) for c in color), -1)
    cv2.putText(image, ascii_text, (x1 + pad, y0 + 1), cv2.FONT_HERSHEY_SIMPLEX,
                scale, (0, 0, 0), thick, cv2.LINE_AA)
    return image


def draw_text_box(image, text, x, y, color, font=None, pad=4, right=False):
    """在 (x, y) 画一个底色实心、文字为黑的中文标签块。

    cv2.putText 不能画中文（会渲染成 ???），所以用 PIL 渲染后贴回 BGR 图。
    渲染结果按 (文本, 颜色, 字号) 缓存，实时循环里几乎不产生额外开销。
    right=True 时 x 视为贴片右边缘（用于贴画面右边的提示）。
    """
    font = font or _CJK_FONT_CACHE.get("font")
    if font is None and "font" not in _CJK_FONT_CACHE:
        font = load_cjk_font()
        _CJK_FONT_CACHE["font"] = font
    if not text:
        return image
    if font is None:
        # 没有中文字体（树莓派上常见）：用 cv2 画英文兜底，别让标签整块消失。
        # 中文渲染不出来就打成 "?"，至少位置/数值还在。
        return _draw_ascii_label(image, text, x, y, color, pad, right)

    key = (text, tuple(int(c) for c in color), font.size)
    cached = _LABEL_CACHE.get(key)
    if cached is None:
        try:
            from PIL import Image, ImageDraw
        except ImportError:
            return image
        # 先量文字尺寸
        probe = Image.new("RGB", (8, 8))
        left, top, right, bottom = ImageDraw.Draw(probe).textbbox((0, 0), text, font=font)
        width, height = max(1, right - left), max(1, bottom - top)
        patch = Image.new("RGB", (width + pad * 2, height + pad * 2),
                          (int(color[2]), int(color[1]), int(color[0])))   # BGR → RGB
        ImageDraw.Draw(patch).text((pad - left, pad - top), text, font=font, fill=(0, 0, 0))
        cached = cv2.cvtColor(np.array(patch), cv2.COLOR_RGB2BGR)
        _LABEL_CACHE[key] = cached

    height, width = image.shape[:2]
    patch_h, patch_w = cached.shape[:2]
    if right:
        x = int(x) - patch_w                     # 右对齐：x 是贴片右边缘
    else:
        # 左对齐：框太靠画面左边缘时标签会溢出，往右挪到刚好放得下
        x = max(0, min(int(x), max(0, width - patch_w)))
    x = max(0, min(x, max(0, width - patch_w)))
    y = max(0, min(int(y), max(0, height - patch_h)))
    image[y:y + patch_h, x:x + patch_w] = cached
    return image

def _measure_label(text, font):
    """量出标签贴片尺寸（与 draw_text_box 的贴片一致）。"""
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return 0, 0
    probe = Image.new("RGB", (8, 8))
    left, top, right, bottom = ImageDraw.Draw(probe).textbbox((0, 0), text, font=font)
    return max(1, right - left) + 8, max(1, bottom - top) + 8

def draw_zone(image, x1, y1, x2, y2, name, color, font=None, top_margin=0, thickness=None):
    """画一个区域框 + 左上角标签。每个区域只画一层框，不额外加装饰框。"""
    if thickness is None:
        thickness = DEFAULT_LINE_WIDTH
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

    label = f"{name} {x2 - x1}x{y2 - y1}"
    safe_top = max(int(top_margin), 2)          # 状态栏是最后画的，标签必须避开它
    label_h = _measure_label(label, font)[1] if font is not None else 30
    # 标签放框的左上角：优先框外上方，空间不够就贴框内左上角
    text_y = y1 - label_h if y1 - label_h >= safe_top else max(y1 + 2, safe_top + 2)
    draw_text_box(image, label, x1 + 2, text_y, color, font=font)
    return image

def draw_detection(image, box, label, color, font=None, thickness=None, top_margin=0):
    """画一个零件框 + 标签。

    不画虚线、也不写"预测"：**检出的框和卡尔曼补出来的框画法完全一样**，
    操作者只看到一个零件一个框。框外上方放不下标签时改放框内下方，避开状态栏。
    """
    if thickness is None:
        thickness = DEFAULT_LINE_WIDTH
    x1, y1, x2, y2 = (int(v) for v in box)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    if not label:
        return image
    safe_top = max(int(top_margin), 2)
    label_h = _measure_label(label, font)[1] if font is not None else 30
    text_y = y1 - label_h if y1 - label_h >= safe_top else y2 + 2
    draw_text_box(image, label, x1, text_y, color, font=font)
    return image

def draw_trail(image, trail, color=TRAIL_COLOR, thickness=2):
    """画运动轨迹：越旧越暗，末端点一个实心圆，方便看清"从哪来、到哪去"。"""
    if not trail or len(trail) < 2:
        return image
    points = [(int(round(x)), int(round(y))) for x, y in trail]
    span = max(len(points) - 1, 1)
    for index in range(1, len(points)):
        fade = 0.35 + 0.65 * (index / span)          # 0.35=最旧，1.0=最新
        shade = tuple(int(c * fade) for c in color)
        cv2.line(image, points[index - 1], points[index], shade, thickness, cv2.LINE_AA)
    cv2.circle(image, points[-1], max(3, thickness + 1), color, -1, cv2.LINE_AA)
    return image

def draw_stage(image, box, text, color=STAGE_COLOR, font=None, top_margin=0):
    """在区域框的右上方写阶段标签。

    区域名（「装配区 229x396」）占着框左上角那一行，所以阶段标签**再往上抬一行**、右对齐贴框边：
    框窄的时候（料盘只有一两百像素宽）两个标签就会挤在同一行叠在一起，抬一行最省事。
    上面实在没空间（框贴着状态栏）就落到框内，隔一行写，同样不叠。
    """
    if not text:
        return image
    x1, y1, x2, y2 = (int(v) for v in box)
    safe_top = max(int(top_margin), 2)
    label_h = _measure_label(text, font)[1] if font is not None else 30
    if y1 - 2 * label_h >= safe_top:            # 框上方有两行空间：抬一行避开区域名
        text_y = y1 - 2 * label_h
    elif y1 - label_h >= safe_top:              # 只有一行空间：和区域名同一行（框够宽才不叠）
        text_y = y1 - label_h
    else:                                       # 上方没空间：落到框内，隔一行写
        text_y = max(y1 + 2 + label_h, safe_top + 2 + label_h)
    draw_text_box(image, text, x2 - 2, text_y, color, font=font, right=True)
    return image

def draw_machine_state(image, text, font=None, color=STATE_COLOR, margin=10, top=0):
    """把工位状态提示画在**整个屏幕的右上角**（不是某个框上面）。

    位置：状态栏正下方再让开一行，避免和贴着顶部的「装配区」标签挤在一起。
    """
    if not text:
        return image
    patch_h = _measure_label(text, font)[1] if font is not None else 30
    y = max(int(top), 0) + patch_h + 10
    draw_text_box(image, text, image.shape[1] - margin, y, color, font=font, right=True)
    return image
