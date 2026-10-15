"""项目公共工具：路径、类别表、配置加载、中文路径图像读写。

所有脚本统一从这里取路径，避免各自写死绝对路径；
项目整体搬目录时只需改 ROOT 的推导结果，或设置环境变量 VERSION_TASK_ROOT。
"""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np

_env_root = os.environ.get("VERSION_TASK_ROOT")
ROOT = (Path(_env_root).expanduser().resolve() if _env_root
       else Path(__file__).resolve().parents[1])   # = wersion_train/

CONFIGS = ROOT / "configs"
WEIGHTS = ROOT / "weights"
RUNS = ROOT / "runs"
OUTPUT = ROOT / "output"
DATASETS = ROOT / "datasets"
# 兼容旧引用：本文件的 ROOT 现在就是 wersion_train/ 目录本身
WERSION_TRAIN = ROOT

# 预训练权重（用户的 yolo26 环境里已有 yolo26n，这里统一放 wersion_train/weights/ 下引用）
DEFAULT_DETECT_WEIGHTS = WEIGHTS / "yolo26n.pt"
DEFAULT_POSE_WEIGHTS = WEIGHTS / "yolo26n-pose.pt"

# 全部 60 个 yolo26 资源里的检测权重，按尺寸排序，便于换大模型
DETECT_SIZES = ("n", "s", "m", "l", "x")


def ensure_dirs() -> None:
    for directory in (WEIGHTS, RUNS, OUTPUT, CONFIGS, DATASETS):
        directory.mkdir(parents=True, exist_ok=True)


def read_classes(path: str | Path) -> list[str]:
    """读取类别表（每行一个，# 后面是注释，行内注释也去掉）。"""
    names = []
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if line:
            names.append(line)
    return names


def load_image(path: str | Path, flags: int = cv2.IMREAD_COLOR):
    """读图，兼容中文/空格路径；失败返回 None。与 stereo_fusion.load_image 同语义。"""
    path = Path(path)
    if not path.is_file():
        return None
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), flags)


def save_image(path: str | Path, image) -> Path:
    """存图，兼容中文/空格路径；自动建父目录。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower() or ".jpg"
    if suffix not in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"):
        suffix, path = ".png", path.with_suffix(".png")
    ok, buffer = cv2.imencode(suffix, image)
    if not ok:
        raise IOError(f"图像编码失败：{path}")
    buffer.tofile(str(path))
    return path


def font_scale_for(width: int) -> float:
    """按画面宽度给出可读的字号，避免 640 宽和 2560 宽用同一字号。"""
    return max(0.45, min(1.2, width / 1400.0))


def put_label(image, text, x, y, color, scale=None, thickness=2, bg=True):
    """在 (x, y) 处画带底色标签，自动防止越界。"""
    scale = scale or font_scale_for(image.shape[1])
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    height, width = image.shape[:2]
    x = max(0, min(int(x), width - tw - 4))
    y = max(th + base + 2, int(y))
    if bg:
        cv2.rectangle(image, (x - 2, y - th - base - 2), (x + tw + 2, y + 2), color, -1)
        cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0),
                    thickness, cv2.LINE_AA)
    else:
        cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color,
                    thickness, cv2.LINE_AA)
    return image


def stage_of(step_name: str) -> int:
    """把 step_2 / complete 这类类别名换算成可比较的进度序号，用于防倒退判断。"""
    name = str(step_name).strip().lower()
    if name.startswith("step_"):
        tail = name[5:]
        if tail.isdigit():
            return int(tail)
    if name in ("complete", "finished", "done", "完整体"):
        return 99
    return -1
