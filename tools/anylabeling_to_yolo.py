#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AnyLabeling 标注结果 → YOLO 训练目录。

AnyLabeling（X-AnyLabeling）在图片旁边写一个同名 .json：

    {
      "imagePath": "step_0_1.jpg",
      "imageWidth": 900, "imageHeight": 500,
      "shapes": [
        {"label": "base", "shape_type": "rectangle",
         "points": [[x1, y1], [x2, y2]]},
        ...
      ]
    }

这个脚本把它转成 YOLO 要的：

    <输出目录>/images/train/*.jpg   <输出目录>/labels/train/*.txt
    <输出目录>/images/val/*.jpg     <输出目录>/labels/val/*.txt
    <输出目录>/data.yaml

用法：

    python tools/anylabeling_to_yolo.py                       # 默认读 output/snapshots/steps
    python tools/anylabeling_to_yolo.py --split 0.2 --copy    # 复制图片（默认是硬链接）

没标完也能跑：没 .json 的图片会被跳过并统计出来；空标注（0 个框）的图同样跳过
（YOLO 里"有图没框"会被当成负样本，容易把模型带偏，除非你确实想要背景图）。
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SRC = ROOT / "output" / "snapshots" / "steps"
DEFAULT_OUT = ROOT / "vision_yolo_train" / "datasets" / "steps_yolo"
DEFAULT_CLASSES = ["base", "support", "compass", "top"]


def read_classes(src: Path, fallback: list[str]) -> list[str]:
    file = src / "classes.txt"
    if file.is_file():
        names = [line.strip() for line in file.read_text(encoding="utf-8").splitlines()
                 if line.strip()]
        if names:
            return names
    return list(fallback)


def box_of(shape: dict):
    """AnyLabeling 的 shape → (x1, y1, x2, y2)，像素坐标。"""
    points = shape.get("points") or []
    if not points:
        return None
    xs = [float(p[0]) for p in points]
    ys = [float(p[1]) for p in points]
    return min(xs), min(ys), max(xs), max(ys)


def convert(src: Path, out: Path, classes: list[str], split: float, copy: bool,
            seed: int = 0) -> int:
    images = sorted(src.glob("*.jpg")) + sorted(src.glob("*.png"))
    labelled, missing, empty, skipped = [], 0, 0, []
    for image in images:
        meta_path = image.with_suffix(".json")
        if not meta_path.is_file():
            missing += 1
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        width = float(meta.get("imageWidth") or image.shape[1] if hasattr(image, "shape") else 0)
        height = float(meta.get("imageHeight") or 0)
        if not width or not height:                     # 老版本可能没写尺寸，读图补
            import cv2
            probe = cv2.imread(str(image))
            if probe is None:
                skipped.append(f"{image.name}（读不出图）")
                continue
            height, width = probe.shape[:2]
        lines = []
        for shape in meta.get("shapes") or []:
            label = str(shape.get("label", "")).strip()
            if label not in classes:
                skipped.append(f"{image.name}（未知类别 {label}）")
                continue
            box = box_of(shape)
            if box is None:
                continue
            x1, y1, x2, y2 = box
            cx, cy = (x1 + x2) / 2 / width, (y1 + y2) / 2 / height
            bw, bh = (x2 - x1) / width, (y2 - y1) / height
            if bw <= 0 or bh <= 0:
                continue
            lines.append(f"{classes.index(label)} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        if not lines:
            empty += 1
            continue
        labelled.append((image, lines))

    if not labelled:
        print(f"没有可用的标注（缺 json {missing} 张）")
        return 1

    random.Random(seed).shuffle(labelled)
    cut = max(1, int(round(len(labelled) * split))) if split > 0 else 0
    buckets = {"val": labelled[:cut], "train": labelled[cut:]}

    for part, items in buckets.items():
        image_dir = out / "images" / part
        label_dir = out / "labels" / part
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        for image, lines in items:
            target = image_dir / image.name
            if target.exists():
                target.unlink()
            if copy:
                shutil.copy2(image, target)
            else:
                try:
                    os.link(image, target)              # 硬链接：不占额外空间
                except OSError:
                    shutil.copy2(image, target)
            (label_dir / f"{image.stem}.txt").write_text("\n".join(lines) + "\n",
                                                         encoding="utf-8")

    yaml = (f"path: {out.as_posix()}\n"
            f"train: images/train\nval: images/val\n"
            f"nc: {len(classes)}\n"
            f"names: {classes}\n")
    (out / "data.yaml").write_text(yaml, encoding="utf-8")

    print(f"训练图 {len(buckets['train'])} 张，验证图 {len(buckets['val'])} 张")
    print(f"类别 {classes}")
    print(f"输出 {out}")
    print(f"data.yaml:\n{yaml}")
    if missing:
        print(f"提醒：{missing} 张图片还没有 .json（未标注）")
    if empty:
        print(f"提醒：{empty} 张标了但一个框都没有，已跳过")
    if skipped:
        print("个别跳过：")
        for item in skipped[:10]:
            print("  ", item)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="AnyLabeling 标注 → YOLO 数据集")
    parser.add_argument("--src", default=str(DEFAULT_SRC), help="图片和 .json 所在目录")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="输出数据集目录")
    parser.add_argument("--split", type=float, default=0.2, help="验证集比例（默认 0.2）")
    parser.add_argument("--classes", nargs="*", default=DEFAULT_CLASSES, help="类别顺序")
    parser.add_argument("--copy", action="store_true", help="复制图片（默认硬链接）")
    args = parser.parse_args()

    src = Path(args.src).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()
    if not src.is_dir():
        print(f"目录不存在：{src}")
        return 1
    classes = read_classes(src, args.classes)
    return convert(src, out, classes, args.split, args.copy)


if __name__ == "__main__":
    sys.exit(main())
