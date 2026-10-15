"""把 labels/raw + images/raw 划分成 train/val，并同步更新 data yaml。

划分策略：按"每张图包含的主要类别"分层，保证每个类别在 val 里都有代表，
否则小类别可能一张 val 都进不去，训完看不出好坏。

用法：
    python wersion_train/tools/split_dataset.py --data wersion_parts
    python wersion_train/tools/split_dataset.py --data wersion_steps --val-ratio 0.2 --seed 0
    python wersion_train/tools/split_dataset.py --data wersion_hand_pose --move   # 移动而非复制

默认复制（原图留在 images/raw 里方便回看）；--move 省磁盘。
同时会重写 wersion_train/configs/data_<name>.yaml 的 train/val/path，保证训练脚本拿到正确路径。
"""

from __future__ import annotations

import argparse
import random
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
WERSION = ROOT / "wersion_train"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# 数据集 → 对应的 data yaml（训练入口都用这个文件）
YAML_FOR = {
    "wersion_parts": ROOT / "configs" / "data_wersion_parts.yaml",
    "wersion_steps": ROOT / "configs" / "data_wersion_steps.yaml",
    "wersion_regions": ROOT / "configs" / "data_regions.yaml",
    "wersion_hand_pose": ROOT / "configs" / "data_hand_pose.yaml",
}


def main_classes(label_path: Path) -> set[int]:
    classes = set()
    if not label_path.is_file():
        return classes
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if parts:
            try:
                classes.add(int(float(parts[0])))
            except ValueError:
                continue
    return classes


def update_yaml(yaml_path: Path, dataset_dir: Path) -> None:
    """只改 path/train/val 三行，其余注释原样保留。"""
    if not yaml_path.is_file():
        print(f"[警告] 数据配置文件不存在，跳过更新：{yaml_path}")
        return
    text = yaml_path.read_text(encoding="utf-8")
    lines = text.splitlines()
    wanted = {"path": str(dataset_dir), "train": "images/train", "val": "images/val"}
    seen = set()
    for i, line in enumerate(lines):
        match = re.match(r"^(\s*)(path|train|val)\s*:", line)
        if match and match.group(2) in wanted:
            key = match.group(2)
            lines[i] = f"{key}: {wanted[key]}"
            seen.add(key)
    for key in ("path", "train", "val"):
        if key not in seen:
            lines.append(f"{key}: {wanted[key]}")
    yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[配置] 已更新 {yaml_path.name}：path={dataset_dir}")


def main() -> int:
    parser = argparse.ArgumentParser(description="划分 train/val")
    parser.add_argument("--data", required=True, help="数据集名，例如 wersion_parts")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="验证集比例")
    parser.add_argument("--seed", type=int, default=0, help="随机种子（保证可复现）")
    parser.add_argument("--move", action="store_true", help="移动原图而不是复制")
    parser.add_argument("--clean", action="store_true", help="先清空 train/val 再划分")
    parser.add_argument("--no-yaml", action="store_true", help="不更新 data yaml")
    args = parser.parse_args()

    dataset_dir = WERSION / "datasets" / args.data
    raw_images = dataset_dir / "images" / "raw"
    raw_labels = dataset_dir / "labels" / "raw"
    if not raw_images.is_dir():
        print(f"[错误] 找不到 {raw_images}；先跑 convert_anylabeling.py")
        return 2

    images = [p for p in sorted(raw_images.iterdir()) if p.suffix.lower() in IMAGE_SUFFIXES]
    if not images:
        print(f"[错误] {raw_images} 里没有图片")
        return 2

    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        target = dataset_dir / sub
        if args.clean and target.is_dir():
            for item in target.iterdir():
                if item.is_file():
                    item.unlink()
        target.mkdir(parents=True, exist_ok=True)

    # 按类别分层
    buckets: dict[tuple, list[Path]] = defaultdict(list)
    missing_labels = []
    for image in images:
        label = raw_labels / f"{image.stem}.txt"
        if not label.is_file():
            missing_labels.append(image.name)
        key = tuple(sorted(main_classes(label))) or (-1,)   # -1 = 负样本（无目标）
        buckets[key].append(image)

    rng = random.Random(args.seed)
    train, val = [], []
    for key, items in sorted(buckets.items()):
        rng.shuffle(items)
        cut = max(1, int(round(len(items) * args.val_ratio))) if len(items) > 1 else 0
        val += items[:cut]
        train += items[cut:]

    moved = 0
    for split, items in (("train", train), ("val", val)):
        for image in items:
            label = raw_labels / f"{image.stem}.txt"
            transfer = shutil.move if args.move else shutil.copy2
            transfer(str(image), str(dataset_dir / "images" / split / image.name))
            if label.is_file():
                transfer(str(label), str(dataset_dir / "labels" / split / f"{image.stem}.txt"))
            moved += 1

    print(f"[完成] train {len(train)}，val {len(val)}（共 {len(images)} 张）")
    print(f"        分层键（类别组合）：{len(buckets)} 组")
    if missing_labels:
        print(f"[警告] {len(missing_labels)} 张图没有标签文件（会被当负样本），"
              f"前 5 个：{missing_labels[:5]}")
    if not val:
        print("[警告] 验证集是空的：图片太少，先把 val-ratio 调小或补数据")

    if not args.no_yaml:
        yaml_path = YAML_FOR.get(args.data)
        if yaml_path:
            update_yaml(yaml_path, dataset_dir)
    print(f"下一步：python wersion_train/tools/check_labels.py --data {args.data}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
