"""vision_yolo_train 的统一路径定义 —— 所有脚本都从这里取路径。

为什么集中定义：
    以前每个脚本各自写 `Path(__file__).resolve().parents[N]`，脚本一挪层级就错；
    而且 configs/datasets/weights 的拼法散落各处，改名要全仓库搜。
    现在只有本文件认识目录结构，其余脚本一律 from paths import ...。

目录结构（本文件所在目录 = 训练根）：
    vision_yolo_train/
      train_detect.py      训练入口
      paths.py             本文件
      tools/               抽帧 / 转换标注 / 划分 / 体检 / 下权重 / 建骨架
      configs/             数据集 yaml、类别表
      weights/             预训练权重（.pt 不入库，用 download_weights.py 重下）
      images/              待标注的原始图片：parts/（零件识别）、steps/（步骤识别）
      datasets/            YOLO 数据集：<数据集>/{images,labels}/{train,val,raw}
        annotate/          按类别分开的标注工作区：parts/<类别>/、steps/<步骤>/
      annotations/         X-AnyLabeling 导出统一放这
      raw_frames/          视频抽帧原始帧
      runs/                训练输出（权重、曲线）
"""

from __future__ import annotations

import os
from pathlib import Path

# 训练根目录（= 本文件所在目录）；可用环境变量覆盖，便于在别处引用权重/数据集
_ENV = os.environ.get("VISION_YOLO_TRAIN_ROOT")
ROOT: Path = Path(_ENV).expanduser().resolve() if _ENV and Path(_ENV).is_dir() \
    else Path(__file__).resolve().parent

TOOLS = ROOT / "tools"
CONFIGS = ROOT / "configs"
WEIGHTS = ROOT / "weights"
IMAGES = ROOT / "images"              # 待标注图片：parts/ steps/
DATASETS = ROOT / "datasets"          # YOLO 数据集
ANNOTATE = DATASETS / "annotate"      # 按类别分开的标注工作区
ANNOTATIONS = ROOT / "annotations"    # X-AnyLabeling 导出
RAW_FRAMES = ROOT / "raw_frames"
RUNS = ROOT / "runs"
EVENTS = ROOT / "events"              # 预留

# 两个数据集：零件识别 / 步骤识别
DATASET_NAMES = ("wersion_parts", "wersion_steps")
DATASET_YAML = {
    "wersion_parts": CONFIGS / "data_wersion_parts.yaml",
    "wersion_steps": CONFIGS / "data_wersion_steps.yaml",
}
DATASET_DIR = {name: DATASETS / name for name in DATASET_NAMES}

# 预训练权重
DEFAULT_DETECT_WEIGHTS = WEIGHTS / "yolo26n.pt"

# 图片分类目录（images/ 下按识别任务分两部分）
IMAGE_GROUPS = {"parts": IMAGES / "parts", "steps": IMAGES / "steps"}


def ensure_dirs(*extra: Path) -> None:
    """确保目录存在（幂等）。"""
    for directory in (TOOLS, CONFIGS, WEIGHTS, IMAGES, DATASETS, ANNOTATE,
                      ANNOTATIONS, RAW_FRAMES, RUNS, *extra):
        Path(directory).mkdir(parents=True, exist_ok=True)


def describe() -> str:
    """一行式结构说明，供各脚本启动时打印，便于确认路径解析是否正确。"""
    return (f"训练根目录 {ROOT}\n"
            f"  权重 {WEIGHTS}\n"
            f"  配置 {CONFIGS}\n"
            f"  数据集 {DATASETS}\n"
            f"  待标注图片 {IMAGES}（parts / steps）")


if __name__ == "__main__":
    print(describe())
    for name, path in (("tools", TOOLS), ("configs", CONFIGS), ("weights", WEIGHTS),
                       ("images", IMAGES), ("datasets", DATASETS),
                       ("datasets/annotate", ANNOTATE), ("annotations", ANNOTATIONS),
                       ("raw_frames", RAW_FRAMES), ("runs", RUNS)):
        print(f"  {'OK ' if path.is_dir() else '缺失'} {name:20} {path}")
    for name, yaml_path in DATASET_YAML.items():
        print(f"  {'OK ' if yaml_path.is_file() else '缺失'} {name:20} {yaml_path}")
