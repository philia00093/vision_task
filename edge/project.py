"""项目根目录与目录工具。

`ROOT` 是权重 / configs / output 的基准；find_root 会按环境变量 → 脚本目录 → 向上找 → 当前目录 的顺序确定它。"""

from __future__ import annotations

import os
import re
from pathlib import Path



def find_root(start: Path | None = None) -> Path:
    """确定项目根目录：权重、configs、output 都以它为基准。

    依次尝试：
        1. 环境变量 VISION_TASK_ROOT —— 显式指定，只要目录存在就以它为准，优先级最高
        2. 脚本所在目录（若其下有 vision_yolo_train/weights/ 或 vision_yolo_train/configs/）
        3. 从脚本目录逐级向上找带 vision_yolo_train/weights/ 或 vision_yolo_train/configs/ 的目录
        4. 当前工作目录
        5. 兜底：脚本所在目录
    """
    # 1) 显式指定优先：存在即采纳，避免被脚本目录"抢答"
    # 新旧环境变量名都认，避免旧脚本失效
    env = os.environ.get("VISION_TASK_ROOT") or os.environ.get("VESION_TASK_ROOT")
    if env:
        candidate = Path(env).expanduser()
        if candidate.is_dir():
            return candidate.resolve()

    here = (start or Path(__file__)).resolve().parent
    # 2) + 3) 脚本目录及其各级父目录
    for candidate in (here, *here.parents):
        try:
            if (candidate / "vision_yolo_train").is_dir() or (candidate / "weights").is_dir():
                return candidate
        except OSError:
            continue
    # 4) 当前工作目录
    try:
        cwd = Path.cwd()
        if (cwd / "vision_yolo_train").is_dir() or (cwd / "weights").is_dir():
            return cwd
    except OSError:
        pass
    # 5) 兜底
    return here

def ensure_dir(path: Path, what: str = "目录") -> Path:
    """确保目录存在；建不出来时给出可操作的报错，而不是让后面写到一半才失败。"""
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise SystemExit(f"无法创建{what}：{path}\n  原因：{error}\n"
                         f"  可用 --save-dir 指定一个可写目录，或设置环境变量 VISION_TASK_ROOT")
    return path

ROOT = find_root()
