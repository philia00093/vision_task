"""帧与图像的基础处理（不含相机、不含识别）。

· split_stereo / take_right：双目拼接帧取右目
· prepare_detect_input：按 --detect-width 缩放，返回缩放比
· contours_to_boxes：掩膜 → 框（按 x 排序）
· save_image：兼容中文路径的存图"""

from __future__ import annotations

import cv2
from pathlib import Path

from edge.constants import GREEN_MIN_AREA_RATIO, GREEN_MIN_FILL_RATIO


def take_right(frame, layout):
    """取出右目画面：layout=none 表示整帧即单目，不做拆分。

    双目拼接帧里右目是后半幅，所以取 split_stereo 的索引 1（左目是索引 0）。
    """
    if layout == "none":
        return frame
    return split_stereo(frame, layout)[1]

def prepare_detect_input(view, width):
    """识别输入缩放：--detect-width 指定模型输入宽度，返回 (图像, 缩放比)。

    缩放比用于把模型输出的框换算回原分辨率；width<=0 或与原图同宽则不动。
    """
    if not width or width <= 0 or int(width) == view.shape[1]:
        return view, 1.0
    scale = float(width) / float(view.shape[1])
    resized = cv2.resize(view, (int(width), max(1, int(round(view.shape[0] * scale)))),
                         interpolation=cv2.INTER_AREA)
    return resized, scale

def contours_to_boxes(mask, width, height, min_area_ratio=GREEN_MIN_AREA_RATIO,
                      min_fill_ratio=GREEN_MIN_FILL_RATIO, merge_gap=0):
    """掩码 → 按 x 排序的框列表 [(x1, y1, x2, y2, area), ...]。"""
    min_area = min_area_ratio * width * height
    boxes = []
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if w <= 0 or h <= 0 or area / float(w * h) < min_fill_ratio:
            continue
        boxes.append([x, y, x + w, y + h, area])

    if merge_gap > 0 and len(boxes) > 1:            # 料盘被反光/手切成两块时合并
        boxes.sort(key=lambda b: b[0])
        merged = [boxes[0]]
        for box in boxes[1:]:
            last = merged[-1]
            if box[0] - last[2] <= merge_gap:
                last[2], last[3] = max(last[2], box[2]), max(last[3], box[3])
                last[1], last[4] = min(last[1], box[1]), last[4] + box[4]
            else:
                merged.append(box)
        boxes = merged

    boxes.sort(key=lambda b: b[0])                  # 左 → 右
    return [tuple(b) for b in boxes]

def split_stereo(frame, layout="auto"):
    """把双目相机的拼接帧拆成 (左视图, 右视图)。

    layout 取值：
        "auto"          按长宽比自动判断（宽 >= 高 视为左右并排）；
        "side_by_side"  左右并排；
        "top_bottom"    上下堆叠。
    """
    if frame is None:
        raise ValueError("split_stereo 收到空帧")
    height, width = frame.shape[:2]
    if layout == "auto":
        layout = "side_by_side" if width >= height else "top_bottom"
    if layout == "side_by_side":
        half = width // 2
        return frame[:, :half].copy(), frame[:, half:half * 2].copy()
    if layout == "top_bottom":
        half = height // 2
        return frame[:half, :].copy(), frame[half:half * 2, :].copy()
    raise ValueError(f"未知的拼接方式：{layout}")

def save_image(path, image):
    """保存图像：自动创建父目录，兼容中文路径。返回 Path。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower() or ".jpg"
    if suffix not in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"):
        suffix = ".png"
        path = path.with_suffix(".png")
    ok, buffer = cv2.imencode(suffix, image)
    if not ok:
        raise IOError(f"图像编码失败：{path}")
    buffer.tofile(str(path))
    return path
