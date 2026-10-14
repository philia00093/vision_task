"""画面绘制：区域框、手框 + 跟踪 ID、21 点骨架、状态栏与告警。

颜色约定（和区域语义一致）：
    料盘区 = 青绿，装配区 = 橙，手 = 洋红；
    手进入的区域边框会被加粗高亮，一眼能看出"手现在在哪个区"。
"""

from __future__ import annotations

import cv2

from ..common import put_label

COLOR_TRAY = (200, 255, 0)        # BGR：青绿
COLOR_ASSEMBLY = (0, 165, 255)    # 橙
COLOR_HAND = (255, 0, 220)        # 洋红
COLOR_TIP = (0, 255, 255)         # 黄
COLOR_WARN = (0, 0, 255)

# 21 点手部骨架连接（MediaPipe 顺序）：掌根→五指，指节依次相连
HAND_SKELETON = [
    (0, 1), (1, 2), (2, 3), (3, 4),                    # 拇指
    (0, 5), (5, 6), (6, 7), (7, 8),                    # 食指
    (5, 9), (9, 10), (10, 11), (11, 12),               # 中指
    (9, 13), (13, 14), (14, 15), (15, 16),             # 无名指
    (13, 17), (17, 18), (18, 19), (19, 20),            # 小指
    (0, 17),                                           # 掌根横向闭合
]

ZONE_TEXT = {"tray_area": "料盘区", "assembly_area": "装配区",
             "outside": "区域外", "none": "无手"}


def draw_regions(image, state, thickness=2, highlight: str = ""):
    """画区域框：label:conf，命中区加粗。"""
    for region in state.regions:
        x1, y1, x2, y2 = (int(round(v)) for v in region.box)
        color = COLOR_TRAY if region.name == "tray_area" else COLOR_ASSEMBLY
        thick = thickness + 2 if region.name == highlight else thickness
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thick)
        text = region.name if region.source in ("green", "geometry") \
            else f"{region.name} {region.conf:.2f}"
        if region.source == "green":
            text += " [绿色]"
        elif region.source == "geometry":
            text += " [兜底]"
        put_label(image, text, x1 + 4, y1 + 24, color, thickness=1)
    return image


def draw_hand(image, hand, index=0, show_keypoints=True):
    """画手框 + 跟踪 ID + 21 点骨架。track_id < 0 表示跟踪器还没分配 ID。"""
    x1, y1, x2, y2 = (int(round(v)) for v in hand.box)
    cv2.rectangle(image, (x1, y1), (x2, y2), COLOR_HAND, 2)
    name = f"hand#{hand.track_id}" if hand.track_id >= 0 else "hand(?)"
    put_label(image, f"{name} {hand.conf:.2f}", x1 + 4, y1 + 24, COLOR_HAND, thickness=1)

    if not show_keypoints or not hand.keypoints:
        return image

    scale = max(2, int(round(min(image.shape[:2]) / 240.0)))
    for a, b in HAND_SKELETON:
        if a >= len(hand.keypoints) or b >= len(hand.keypoints):
            continue
        xa, ya, va = hand.keypoints[a]
        xb, yb, vb = hand.keypoints[b]
        if va <= 0 or vb <= 0:
            continue  # 未标或框外的点不连线
        cv2.line(image, (int(xa), int(ya)), (int(xb), int(yb)), COLOR_HAND, max(1, scale - 1),
                 cv2.LINE_AA)
    for i, (x, y, v) in enumerate(hand.keypoints):
        if v <= 0:
            continue
        color = COLOR_TIP if i in (4, 8, 12, 16, 20) else (255, 255, 255)
        cv2.circle(image, (int(x), int(y)), scale, color, -1, cv2.LINE_AA)
    # 食指尖单独标出来，因为越区判断用的就是它
    tip = hand.tip(8)
    if tip is not None:
        cv2.circle(image, (int(tip[0]), int(tip[1])), scale + 3, COLOR_TIP, 2, cv2.LINE_AA)
    return image


def draw_status(image, lines, height=None):
    """顶部黑底状态栏，多行。"""
    if isinstance(lines, str):
        lines = [lines]
    scale = max(0.4, min(0.7, image.shape[1] / 2200.0))
    line_h = int(22 * scale / 0.6)
    height = height or (line_h * len(lines) + 8)
    cv2.rectangle(image, (0, 0), (image.shape[1], height), (0, 0, 0), -1)
    for i, text in enumerate(lines):
        cv2.putText(image, text, (8, int(line_h * (i + 1))), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, (0, 255, 255), 1, cv2.LINE_AA)
    return height


def draw_warnings(image, warnings, top=0):
    """把告警画在状态栏下面，红底，不要静默失败。"""
    if not warnings:
        return top
    scale = max(0.4, min(0.7, image.shape[1] / 2200.0))
    for i, text in enumerate(warnings[:3]):
        y = top + int(20 * (i + 1))
        put_label(image, f"! {text}", 8, y, COLOR_WARN, scale=scale, thickness=1)
    return top + 20 * min(len(warnings), 3)


def annotate(image, state, hands, fps=0.0, extra_lines=None, show_keypoints=True):
    """把一帧画完整：区域 → 手 → 状态栏 → 告警。返回新图（不破坏原帧）。"""
    canvas = image.copy()
    draw_regions(canvas, state, highlight=state.hand_zone)
    for i, hand in enumerate(hands):
        draw_hand(canvas, hand, index=i, show_keypoints=show_keypoints)

    lines = [f"fps {fps:.1f} | hand {ZONE_TEXT.get(state.hand_zone, state.hand_zone)}"
             f"(框) {ZONE_TEXT.get(state.hand_zone_tip, state.hand_zone_tip)}(指尖)"
             f" | hands {len(hands)}"]
    if state.step_label:
        lines.append(f"step {state.step_label} {state.step_conf:.2f}")
    if extra_lines:
        lines.extend(extra_lines)
    bottom = draw_status(canvas, lines)
    draw_warnings(canvas, state.warnings, top=bottom)
    return canvas
