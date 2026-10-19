"""把 labels/raw + images/raw 划分成 train/val（可再留出 test），并同步更新 data yaml。

划分策略：按"每张图包含的主要类别"分层，保证每个类别在 val 里都有代表，
否则小类别可能一张 val 都进不去，训完看不出好坏。

测试集：加 --test-prefix test --test-count 20，会从 test_*.png 里随机抽 20 张单独放
images/test + labels/test，并往 yaml 里写 test: images/test。
抽签用的是 --seed，所以同一个 seed 每次抽到的是同一批图，划分可复现。

用法：
    python vision_yolo_train/tools/split_dataset.py --data wersion_parts
    python vision_yolo_train/tools/split_dataset.py --data wersion_steps --val-ratio 0.2 --seed 0

    # 零件数据集：留 20 张 test 当测试集，其余分层分 train/val（补标后重跑记得 --clean）
    python vision_yolo_train/tools/split_dataset.py --data wersion_parts \
        --test-prefix test --test-count 20 --clean

默认复制（原图留在 images/raw 里方便回看）；--move 省磁盘。
同时会重写 vision_yolo_train/configs/data_<name>.yaml 的 train/val/path，保证训练脚本拿到正确路径。

注意：不加 --clean 时只会往里复制、不会清旧文件。补标之后重跑划分一定要加 --clean，
否则上一轮 split 的图会残留在原处，同一张图可能同时出现在 train 和 val 里。
"""

from __future__ import annotations

import argparse
import random
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import paths  # noqa: E402  统一路径定义（见 vision_yolo_train/paths.py）


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# 数据集 → 对应的 data yaml（训练入口都用这个文件）
YAML_FOR = {
    **paths.DATASET_YAML,
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


def update_yaml(yaml_path: Path, dataset_dir: Path, with_test: bool) -> None:
    """只改 path/train/val(/test) 几行，其余注释原样保留。"""
    if not yaml_path.is_file():
        print(f"[警告] 数据配置文件不存在，跳过更新：{yaml_path}")
        return
    text = yaml_path.read_text(encoding="utf-8")
    lines = text.splitlines()
    wanted = {"path": str(dataset_dir), "train": "images/train", "val": "images/val"}
    if with_test:
        wanted["test"] = "images/test"
    seen = set()
    for i, line in enumerate(lines):
        match = re.match(r"^(\s*)(path|train|val|test)\s*:", line)
        if match and match.group(2) in wanted:
            key = match.group(2)
            lines[i] = f"{key}: {wanted[key]}"
            seen.add(key)
    for key in ("path", "train", "val", "test"):
        if key in wanted and key not in seen:
            lines.append(f"{key}: {wanted[key]}")
    yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[配置] 已更新 {yaml_path.name}：path={dataset_dir}"
          + ("（含 test）" if with_test else ""))


def class_names(data: str) -> list[str]:
    """读 configs/classes_<data>.txt，只为把统计表里的类别 id 显示成名字。"""
    path = paths.CONFIGS / f"classes_{data}.txt"
    if not path.is_file():
        return []
    names = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if line:
            names.append(line)
    return names


def count_instances(items: list[Path], raw_labels: Path) -> Counter:
    """统计一批图里各类别的实例数（按 YOLO 标签首列）。"""
    counter: Counter = Counter()
    for image in items:
        label = raw_labels / f"{image.stem}.txt"
        if not label.is_file():
            continue
        for line in label.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if not parts:
                continue
            try:
                counter[int(float(parts[0]))] += 1
            except ValueError:
                continue
    return counter


def main() -> int:
    parser = argparse.ArgumentParser(description="划分 train/val")
    parser.add_argument("--data", required=True, help="数据集名，例如 wersion_parts")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="验证集比例")
    parser.add_argument("--seed", type=int, default=0, help="随机种子（保证可复现）")
    parser.add_argument("--move", action="store_true", help="移动原图而不是复制")
    parser.add_argument("--clean", action="store_true", help="先清空 train/val 再划分")
    parser.add_argument("--no-yaml", action="store_true", help="不更新 data yaml")
    parser.add_argument("--test-prefix", default=None,
                        help="留作测试集的图片名前缀（例如 test）；配合 --test-count 用")
    parser.add_argument("--test-count", type=int, default=20,
                        help="从 --test-prefix 前缀的图里随机抽几张做测试集（默认 20）")
    args = parser.parse_args()

    dataset_dir = paths.DATASET_DIR[args.data]
    raw_images = dataset_dir / "images" / "raw"
    raw_labels = dataset_dir / "labels" / "raw"
    if not raw_images.is_dir():
        print(f"[错误] 找不到 {raw_images}；先跑 convert_anylabeling.py")
        return 2

    images = [p for p in sorted(raw_images.iterdir()) if p.suffix.lower() in IMAGE_SUFFIXES]
    if not images:
        print(f"[错误] {raw_images} 里没有图片")
        return 2

    # 先挑测试集：从 raw 里按前缀取候选，随机抽固定张数。
    # 抽签用 --seed 的随机源，所以同一个 seed 每次抽到的是同一批图（划分可复现）。
    rng = random.Random(args.seed)
    test_images: list[Path] = []
    if args.test_prefix:
        pattern = re.compile(rf"^{re.escape(args.test_prefix)}_\d+$")
        candidates = [p for p in images if pattern.match(p.stem)]
        if len(candidates) < args.test_count:
            print(f"[错误] 前缀 {args.test_prefix}_* 的图只有 {len(candidates)} 张，"
                  f"不够抽 {args.test_count} 张做测试集")
            return 2
        test_images = sorted(rng.sample(sorted(candidates), args.test_count))
        chosen = {p.stem for p in test_images}
        images = [p for p in images if p.stem not in chosen]
        print(f"[测试集] 从 {len(candidates)} 张 {args.test_prefix}_* 里抽了 "
              f"{len(test_images)} 张（seed={args.seed}）：")
        print("         " + ", ".join(p.stem for p in test_images))

    splits = ["images/train", "images/val", "labels/train", "labels/val"]
    if args.test_prefix:
        splits += ["images/test", "labels/test"]
    for sub in splits:
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

    train, val = [], []
    for key, items in sorted(buckets.items()):
        rng.shuffle(items)
        cut = max(1, int(round(len(items) * args.val_ratio))) if len(items) > 1 else 0
        val += items[:cut]
        train += items[cut:]

    moved = 0
    for split, items in (("train", train), ("val", val), ("test", test_images)):
        for image in items:
            label = raw_labels / f"{image.stem}.txt"
            transfer = shutil.move if args.move else shutil.copy2
            transfer(str(image), str(dataset_dir / "images" / split / image.name))
            if label.is_file():
                transfer(str(label), str(dataset_dir / "labels" / split / f"{image.stem}.txt"))
            moved += 1

    print(f"[完成] train {len(train)}，val {len(val)}"
          + (f"，test {len(test_images)}" if args.test_prefix else "")
          + f"（待划分 {len(images)} 张）")
    print(f"        分层键（类别组合）：{len(buckets)} 组")

    # 分层统计：确认每个 split 里四个类别都有代表
    names = class_names(args.data)
    header = "  ".join(
        f"{(names[i] if i < len(names) else str(i)):>8}" for i in range(len(names))
    ) if names else ""
    print("\n[划分统计]（实例数）")
    print(f"  {'split':<6}{'图片':>6}   {header}")
    for split, items in (("train", train), ("val", val), ("test", test_images)):
        if split == "test" and not args.test_prefix:
            continue
        counts = count_instances(items, raw_labels)
        cells = "  ".join(f"{counts.get(i, 0):>8}" for i in range(len(names))) if names else ""
        print(f"  {split:<6}{len(items):>6}   {cells}")
    if missing_labels:
        print(f"[警告] {len(missing_labels)} 张图没有标签文件（会被当负样本），"
              f"前 5 个：{missing_labels[:5]}")
    if not val:
        print("[警告] 验证集是空的：图片太少，先把 val-ratio 调小或补数据")

    if not args.no_yaml:
        yaml_path = YAML_FOR.get(args.data)
        if yaml_path:
            update_yaml(yaml_path, dataset_dir, with_test=bool(args.test_prefix))
    next_cmd = f"python vision_yolo_train/tools/check_labels.py --data {args.data}"
    if args.test_prefix:
        next_cmd += " --splits train val test"
    print(f"下一步：{next_cmd}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
