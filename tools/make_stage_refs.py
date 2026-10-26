#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把「现场存下来的帧」变成阶段参考图。

为什么需要它：阶段匹配比的是**画面长什么样**，而画面随灯光变。
换了时间/开了灯/挪了台灯，同一台机器拍出来就完全是两种像素 —— 参考图必须跟
现场同一个光底下拍，不然距离会从 0.2 飙到 0.9（实测，托盘反光条一没就废）。

用法
----
1. 现场把零件摆成第 N 步的样子，在监控台（端口 2000）里点「保存当前帧」，
   标签填 `step0`（或 step1/2/3）——存到 `output/snapshots/`。
2. 四步都存完（每步建议 2~3 张、摆位稍微错开），跑：

       python tools/make_stage_refs.py --dry-run     # 先看会挑哪些
       python tools/make_stage_refs.py               # 真的写进参考图目录

3. 重新部署：`bash deploy/deploy_pi.sh`

默认每个阶段取最新的 3 张；旧的参考图会先备份到 `<目录>.bak-时间戳/`。
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SNAPSHOT_DIR = ROOT / "output" / "snapshots"
STAGE_DIR = ROOT / "vision_yolo_train" / "images" / "steps"

_STEP_RE = re.compile(r"step[_\-]?(\d+)", re.IGNORECASE)


def stem_label(path: Path):
    found = _STEP_RE.search(path.stem)
    return f"step{found.group(1)}" if found else None


def pick(snapshot_dir: Path, keep: int) -> dict[str, list[Path]]:
    """按标签挑出每个阶段最新的 N 张（文件名里带时间戳，按名字排序就是时间顺序）。"""
    buckets: dict[str, list[Path]] = {}
    for path in sorted(snapshot_dir.glob("*.jpg")):
        label = stem_label(path)
        if label:
            buckets.setdefault(label, []).append(path)
    return {label: paths[-keep:] for label, paths in buckets.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description="把现场快照转成阶段参考图")
    parser.add_argument("--snapshot-dir", default=str(SNAPSHOT_DIR))
    parser.add_argument("--stage-dir", default=str(STAGE_DIR))
    parser.add_argument("--keep", type=int, default=3, help="每个阶段保留最新几张（默认 3）")
    parser.add_argument("--dry-run", action="store_true", help="只列出会做什么，不写文件")
    parser.add_argument("--no-backup", action="store_true", help="不备份旧的参考图")
    args = parser.parse_args()

    snapshot_dir = Path(args.snapshot_dir)
    stage_dir = Path(args.stage_dir)
    if not snapshot_dir.is_dir():
        print(f"找不到快照目录：{snapshot_dir}")
        return 1

    chosen = pick(snapshot_dir, max(args.keep, 1))
    if not chosen:
        print(f"{snapshot_dir} 里没有带 stepN 标签的帧 —— 先在监控台用标签 step0..step3 保存几帧")
        return 1

    total = sum(len(v) for v in chosen.values())
    print(f"从 {snapshot_dir} 挑出 {total} 张：")
    for label in sorted(chosen):
        for path in chosen[label]:
            print(f"  {label:6} {path.name}")

    if args.dry_run:
        print("\n--dry-run：什么都没写")
        return 0

    if not args.no_backup and stage_dir.is_dir() and any(stage_dir.iterdir()):
        stamp = datetime.now().strftime("%m%d-%H%M")
        backup = stage_dir.with_name(f"{stage_dir.name}.bak-{stamp}")
        shutil.copytree(stage_dir, backup)
        print(f"\n旧参考图备份到 {backup}")

    stage_dir.mkdir(parents=True, exist_ok=True)
    for old in stage_dir.glob("*.jpg"):
        old.unlink()
    for label in sorted(chosen):
        for i, path in enumerate(chosen[label], 1):
            target = stage_dir / f"{path.stem}_ref{i}.jpg"
            shutil.copy2(path, target)
            print(f"  → {target.name}")
    print(f"\n写好 {total} 张到 {stage_dir}")
    print("接着跑：bash deploy/deploy_pi.sh")
    return 0


if __name__ == "__main__":
    sys.exit(main())
