"""训练前体检：标签格式、坐标范围、类别分布、图片/标签配对。

这一步的意义：YOLO 训练遇到坏标签时往往只是精度莫名偏低，
不会明确报错，所以在开训前把问题一次性摊开。

用法：
    python wersion_train/tools/check_labels.py --data wersion_parts
    python wersion_train/tools/check_labels.py --data wersion_steps --fix    # 顺手清理坏行

退出码：0 正常；1 有错误（--strict 时连警告也算失败）。
"""

from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

for _stream in (sys.stdout, sys.stderr):   # 防止中文提示在 cmd 里变乱码
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass

WERSION = ROOT / "wersion_train"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
KPT_SHAPE = 21


def read_names_from_yaml(path: Path) -> list[str]:
    """只解析 names: 段，避免为了一件小事引入 yaml 依赖。"""
    if not path.is_file():
        return []
    names: dict[int, str] = {}
    inside = False
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        if line.strip().startswith("names:"):
            inside = True
            continue
        if inside:
            if not line.startswith((" ", "\t")):
                break
            body = line.strip()
            if ":" in body:
                key, value = body.split(":", 1)
                try:
                    names[int(key)] = value.strip()
                except ValueError:
                    pass
    return [names[i] for i in sorted(names)]


def check_split(dataset_dir: Path, split: str, names: list[str], pose: bool,
                strict: bool, fix: bool):
    images_dir = dataset_dir / "images" / split
    labels_dir = dataset_dir / "labels" / split
    problems: list[str] = []
    warnings: list[str] = []
    class_counts = collections.Counter()
    instances = 0

    if not images_dir.is_dir():
        return [f"缺少目录 {images_dir}"], [], class_counts, 0

    images = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        warnings.append(f"{split}: 目录里没有图片")

    stems = {p.stem for p in images}
    for stray in sorted(labels_dir.glob("*.txt")) if labels_dir.is_dir() else []:
        if stray.stem not in stems:
            warnings.append(f"{split}: 标签 {stray.name} 没有对应图片")

    for image in images:
        label = labels_dir / f"{image.stem}.txt"
        if not label.is_file():
            warnings.append(f"{split}: {image.name} 没有标签（按负样本处理）")
            continue
        raw_lines = label.read_text(encoding="utf-8").splitlines()
        good_lines = []
        for number, line in enumerate(raw_lines, 1):
            if not line.strip():
                continue
            parts = line.split()
            expected = 5 + KPT_SHAPE * 3 if pose else 5
            if len(parts) != expected:
                problems.append(f"{split}/{label.name}:{number} 字段数 {len(parts)} != {expected}")
                continue
            try:
                values = [float(v) for v in parts]
            except ValueError:
                problems.append(f"{split}/{label.name}:{number} 含非数字")
                continue
            class_id = int(values[0])
            if class_id < 0 or (names and class_id >= len(names)):
                problems.append(f"{split}/{label.name}:{number} 类别 {class_id} 超出范围"
                                f"（共 {len(names)} 类）")
                continue
            box = values[1:5]
            if any(v < 0.0 or v > 1.0 for v in box):
                problems.append(f"{split}/{label.name}:{number} 框坐标超出 0~1：{box}")
                continue
            if box[2] <= 0.0 or box[3] <= 0.0:
                problems.append(f"{split}/{label.name}:{number} 框宽或高为 0：{box}")
                continue
            if pose:
                bad = 0
                for i in range(KPT_SHAPE):
                    x, y, v = values[5 + i * 3: 8 + i * 3]
                    if v not in (0.0, 1.0, 2.0):
                        problems.append(f"{split}/{label.name}:{number} 关键点{i} 可见性 {v} "
                                        "不是 0/1/2")
                        bad += 1
                    elif v > 0 and not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                        problems.append(f"{split}/{label.name}:{number} 关键点{i} 坐标越界"
                                        f"：({x:.3f}, {y:.3f})")
                        bad += 1
                if bad:
                    continue
            class_counts[(class_id, names[class_id] if names and class_id < len(names)
                          else str(class_id))] += 1
            instances += 1
            good_lines.append(line)

        if fix and len(good_lines) != len([ln for ln in raw_lines if ln.strip()]):
            label.write_text("\n".join(good_lines) + ("\n" if good_lines else ""),
                             encoding="utf-8")
            warnings.append(f"{split}: 已清理 {label.name} 的坏行")

    if not class_counts and images:
        warnings.append(f"{split}: 没有任何有效标注")
    return problems, warnings, class_counts, instances


def main() -> int:
    parser = argparse.ArgumentParser(description="数据集体检")
    parser.add_argument("--data", required=True, help="数据集名，例如 wersion_parts")
    parser.add_argument("--yaml", default=None, help="直接指定 data yaml（默认按名字找）")
    parser.add_argument("--strict", action="store_true", help="有警告也返回失败")
    parser.add_argument("--fix", action="store_true", help="顺手删掉坏标签行")
    parser.add_argument("--splits", nargs="*", default=["train", "val"])
    args = parser.parse_args()

    dataset_dir = WERSION / "datasets" / args.data
    if not dataset_dir.is_dir():
        print(f"[错误] 数据集目录不存在：{dataset_dir}；先跑 init_datasets.bat")
        return 2

    yaml_map = {
        "wersion_parts": ROOT / "configs" / "data_wersion_parts.yaml",
        "wersion_steps": ROOT / "configs" / "data_wersion_steps.yaml",
    }
    yaml_path = Path(args.yaml) if args.yaml else yaml_map.get(args.data)
    names = read_names_from_yaml(yaml_path) if yaml_path else []
    if not names:
        class_file = dataset_dir / "classes.txt"
        if class_file.is_file():
            names = [ln.split("#", 1)[0].strip() for ln in
                     class_file.read_text(encoding="utf-8").splitlines()
                     if ln.split("#", 1)[0].strip()]
    pose = False            # 手部 pose 数据集已移除，只做检测

    print(f"[体检] {args.data}（{'pose 21 点' if pose else 'detect'}）")
    print(f"       类别：{names or '未找到（只做格式检查）'}")
    print(f"       目录：{dataset_dir}\n")

    all_problems, all_warnings = [], []
    totals: dict[str, tuple[int, int]] = {}
    merged = collections.Counter()
    image_total = 0
    for split in args.splits:
        problems, warnings, counts, instances = check_split(
            dataset_dir, split, names, pose, args.strict, args.fix)
        all_problems += problems
        all_warnings += warnings
        merged.update(counts)
        totals[split] = (sum(counts.values()), instances)
        images_dir = dataset_dir / "images" / split
        if images_dir.is_dir():
            image_total += sum(1 for p in images_dir.iterdir()
                               if p.suffix.lower() in IMAGE_SUFFIXES)
        print(f"  {split:>5}: 实例 {instances:>5}")

    if image_total == 0:
        all_problems.append(
            f"{dataset_dir} 里一张图片都没有：先抽帧、标注，再跑 convert_anylabeling.py")

    print("\n[类别分布]")
    if merged:
        for (class_id, name), count in sorted(merged.items()):
            bar = "#" * min(60, max(1, count // max(1, max(merged.values()) // 40)))
            print(f"  {class_id} {name:<18} {count:>6}  {bar}")
        counts = list(merged.values())
        if len(counts) > 1 and max(counts) > min(counts) * 5:
            all_warnings.append(
                f"类别极不平衡（最多 {max(counts)} vs 最少 {min(counts)}），"
                "考虑补数据或调 cls 权重")
    else:
        all_warnings.append("一个有效标注都没有，先标数据")

    if all_warnings:
        print(f"\n[警告] {len(all_warnings)} 条：")
        for item in all_warnings[:20]:
            print(f"   · {item}")
    if all_problems:
        print(f"\n[错误] {len(all_problems)} 条：")
        for item in all_problems[:20]:
            print(f"   · {item}")

    failed = bool(all_problems) or (args.strict and bool(all_warnings))
    print("\n[结论] " + ("不通过，先修上面的错误" if failed else "通过，可以开训"))
    if not failed:
        print(f"        python wersion_train/src/train_detect.py --data {args.data}")
        if pose:
            print(f"        python wersion_train/train_pose.py --data {args.data}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
