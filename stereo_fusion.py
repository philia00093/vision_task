"""双目图像基础工具：拼接帧拆分与图像读写。

split_stereo 把双目相机输出的拼接帧拆成左右两幅视图；
save_image / load_image 用 imencode + tofile 实现，兼容中文与空格路径。
"""

from pathlib import Path

import cv2
import numpy as np


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


def load_image(path, flags=cv2.IMREAD_COLOR):
    """读取图像，兼容中文路径；文件不存在或解码失败时返回 None。"""
    path = Path(path)
    if not path.exists():
        return None
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), flags)


def compose_pair(left, right, gap=0):
    """把左右视图并排合成一帧（调试用，验证 split_stereo 的逆操作）。"""
    if gap > 0:
        filler = np.zeros((left.shape[0], gap, 3), left.dtype)
        return np.hstack([left, filler, right])
    return np.hstack([left, right])
