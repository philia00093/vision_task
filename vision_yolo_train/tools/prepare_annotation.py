"""标注前准备：把 images/parts/ 里按前缀采集的图片归档到各类别目录。

为什么要先归档再标注：
    split_dataset.py 只处理"已标注"的 labels/raw，图片没标注就直接分割，
    会把没标注的图分进 train/val。正确顺序是：
        归档 → 标注 → 划分 train/val → 体检 → 训练

类别名以**图片文件名前缀为准**（用户确认），与 configs/classes_*.txt 保持一致：
    base → base、support → support、compass → compass、top → top
`test` 是通用样本（一张图可能含多个类别），不属于单类归档，默认跳过。

用法：
    python vision_yolo_train/tools/prepare_annotation.py            # 预览
    python vision_yolo_train/tools/prepare_annotation.py --apply    # 实际复制

默认**复制**而不是移动：原图留在 images/parts/ 里作为原始采集记录，
有疑虑时用 --move 改为移动（省一份磁盘占用）。
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import paths  # noqa: E402  统一路径定义


def collect(source: Path) -> dict[str, list[Path]]:
    """按文件名前缀分组。返回 {前缀: [图片路径...]}，按编号排序。"""
    groups: dict[str, list[Path]] = {}
    if not source.is_dir():
        return groups
    for item in sorted(source.iterdir()):
        if not item.is_file() or item.suffix.lower() not in (".png", ".jpg", ".jpeg", ".bmp"):
            continue
        stem = item.stem
        prefix = stem.rsplit("_", 1)[0] if "_" in stem else stem
        groups.setdefault(prefix, []).append(item)

    def number(path: Path) -> int:
        tail = path.stem.rsplit("_", 1)[-1]
        return int(tail) if tail.isdigit() else 0

    for prefix in groups:
        groups[prefix].sort(key=number)
    return groups


def main() -> int:
    parser = argparse.ArgumentParser(description="把采集图片按类别归档，准备标注")
    parser.add_argument("--apply", action="store_true", help="实际执行（默认只预览）")
    parser.add_argument("--move", action="store_true", help="移动而非复制")
    parser.add_argument("--source", default=None, help="图片来源目录，默认 images/parts/")
    parser.add_argument("--skip", nargs="*", default=["test"],
                        help="不参与单类归档的前缀（默认 test，因为通用样本可能含多个类）")
    parser.add_argument("--only", nargs="*", default=None, help="只处理这些前缀")
    parser.add_argument("--overwrite", action="store_true", help="目标已存在时覆盖")
    args = parser.parse_args()

    source = Path(args.source).expanduser() if args.source else paths.IMAGE_GROUPS["parts"]
    if not source.is_dir():
        print(f"[错误] 图片目录不存在：{source}")
        return 2

    groups = collect(source)
    if not groups:
        print(f"[错误] {source} 里没有图片")
        return 2

    print(f"来源：{source}")
    print(f"按前缀分组：{', '.join(f'{k}×{len(v)}' for k, v in sorted(groups.items()))}\n")

    do_copy = not args.move
    plan: list[tuple[Path, Path, str]] = []
    for prefix, files in sorted(groups.items()):
        if args.only and prefix not in args.only:
            print(f"  跳过 {prefix}（--only 未包含）")
            continue
        if prefix in args.skip:
            print(f"  跳过 {prefix}×{len(files)}（通用样本，不参与单类归档）")
            continue
        target_dir = paths.ANNOTATE / "parts" / prefix / "images"
        for src in files:
            dst = target_dir / src.name
            if dst.exists() and not args.overwrite:
                plan.append((src, dst, "已存在，跳过"))
            else:
                plan.append((src, dst, "复制" if do_copy else "移动"))

    to_do = [(s, d, act) for s, d, act in plan if act != "已存在，跳过"]
    skipped = [(s, d, act) for s, d, act in plan if act == "已存在，跳过"]

    print(f"计划：{'复制' if do_copy else '移动'} {len(to_do)} 张"
          f"，跳过已存在 {len(skipped)} 张")
    for src, dst, _act in to_do[:6]:
        print(f"   {src.name}  ->  {dst.parent.relative_to(paths.ROOT)}/")
    if len(to_do) > 6:
        print(f"   ...（其余 {len(to_do) - 6} 张同理）")

    if not args.apply:
        print("\n以上为预览。确认后执行：")
        print("    python vision_yolo_train/tools/prepare_annotation.py --apply")
        return 0

    done = 0
    for src, dst, _act in to_do:
        dst.parent.mkdir(parents=True, exist_ok=True)
        if do_copy:
            shutil.copy2(src, dst)
        else:
            shutil.move(str(src), str(dst))
        done += 1

    print(f"\n[完成] 已{'复制' if do_copy else '移动'} {done} 张到 datasets/annotate/parts/")
    print("\n各类别现有待标注图片：")
    for folder in sorted((paths.ANNOTATE / "parts").iterdir()):
        if folder.is_dir():
            count = len(list((folder / "images").glob("*"))) if (folder / "images").is_dir() else 0
            print(f"   {folder.name:16} {count:4} 张   {folder / 'images'}")
    print("\n下一步：用 X-AnyLabeling 打开上面某个目录标注")
    print("        导出格式选 YOLO Detection，导出到 vision_yolo_train/annotations/")
    print("        标注完再划分：python vision_yolo_train/tools/split_dataset.py --data wersion_parts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
