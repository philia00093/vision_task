"""把 X-AnyLabeling 的标注转成 YOLO 标签（检测框 / 21 点姿态都支持）。

支持两种输入，脚本会自动识别：
    A. 已经是 YOLO 格式的 .txt（X-AnyLabeling 导出时直接选了 "YOLO Detection" /
       "YOLO Pose"）→ 只做体检后原样收集，不做转换；
    B. X-AnyLabeling 的原生 JSON（每张图一个同名 .json）→ 转成 YOLO 格式。

JSON 里识别这些形状：
    rectangle   → 检测框（label 必须是本数据集已知类别）
    polygon     → 取外接矩形当检测框（如果你用了分割标注）

用法：
    python vision_yolo_train/tools/convert_anylabeling.py --data wersion_parts
    python vision_yolo_train/tools/convert_anylabeling.py --data wersion_steps --dry-run

输出：
    datasets/<data>/images/raw/*.jpg   图片汇总
    datasets/<data>/labels/raw/*.txt   YOLO 标签（未划分，交给 split_dataset.py 分）
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import paths  # noqa: E402  统一路径定义（见 vision_yolo_train/paths.py）


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def read_names(path: Path) -> list[str]:
    names = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if line:
            names.append(line)
    return names


def clamp01(value: float) -> float:
    return min(1.0, max(0.0, value))


def shape_to_line(shape: dict, names: list[str], width: int, height: int,
                  pose: bool) -> tuple[str | None, str | None]:
    """把单个 shape 转成一行 YOLO 标签；返回 (行, 错误说明)。"""
    label = str(shape.get("label", "")).strip()
    if label not in names:
        return None, f"未知类别 '{label}'（不在 {names}）"
    class_id = names.index(label)

    points = shape.get("points") or []
    shape_type = str(shape.get("shape_type", "rectangle")).lower()
    keypoints = shape.get("keypoints")

    if pose:
        if not keypoints:
            return None, "pose 数据缺少 keypoints 字段"
        if len(keypoints) != 21:
            return None, f"关键点数量 {len(keypoints)} != 21"
        xs = [float(p[0]) for p in keypoints]
        ys = [float(p[1]) for p in keypoints]
        x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
        if x2 <= x1 or y2 <= y1:
            return None, "关键点包围盒非法（宽或高为 0）"
        cx, cy = (x1 + x2) / 2 / width, (y1 + y2) / 2 / height
        bw, bh = (x2 - x1) / width, (y2 - y1) / height
        parts = [f"{class_id}", f"{cx:.6f}", f"{cy:.6f}", f"{bw:.6f}", f"{bh:.6f}"]
        for point in keypoints:
            px, py = float(point[0]), float(point[1])
            # X-AnyLabeling 里没标的点会写成 -1，按可见性 0 处理
            visible = 0 if (px < 0 or py < 0) else 2
            parts += [f"{clamp01(px / width):.6f}", f"{clamp01(py / height):.6f}", str(visible)]
        return " ".join(parts), None

    if len(points) < 2:
        return None, "检测框点数不足 2"
    if shape_type == "rectangle" and len(points) >= 2:
        (x1, y1), (x2, y2) = points[0], points[1]
    else:  # polygon 或其他：取外接矩形
        xs = [float(p[0]) for p in points]
        ys = [float(p[1]) for p in points]
        x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
    x1, x2 = sorted((float(x1), float(x2)))
    y1, y2 = sorted((float(y1), float(y2)))
    if x2 - x1 < 1 or y2 - y1 < 1:
        return None, "检测框宽或高小于 1 像素"
    cx, cy = (x1 + x2) / 2 / width, (y1 + y2) / 2 / height
    bw, bh = (x2 - x1) / width, (y2 - y1) / height
    return f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}", None


def convert_json(json_path: Path, names: list[str], pose: bool
                 ) -> tuple[list[str], list[str], list[str]]:
    """返回 (标签行, 错误列表, 警告列表)。"""
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    width = int(payload.get("imageWidth") or 0)
    height = int(payload.get("imageHeight") or 0)
    if width <= 0 or height <= 0:
        return [], [f"{json_path.name}: 缺少 imageWidth/imageHeight"], []

    lines, errors, warnings = [], [], []
    for shape in payload.get("shapes", []):
        line, problem = shape_to_line(shape, names, width, height, pose)
        if line:
            lines.append(line)
        elif problem:
            errors.append(f"{json_path.name}: {problem}")
        if str(shape.get("shape_type", "")).lower() not in ("rectangle", "polygon"):
            warnings.append(f"{json_path.name}: shape_type={shape.get('shape_type')} 已按外接矩形处理")

    if not lines:
        warnings.append(f"{json_path.name}: 没有任何可用标注（空图？）")
    if pose and lines and len(lines[0].split()) != 5 + 21 * 3:
        errors.append(f"{json_path.name}: pose 标签字段数 {len(lines[0].split())} "
                      f"应为 {5 + 21 * 3}")
    return lines, errors, warnings


def main() -> int:
    parser = argparse.ArgumentParser(description="X-AnyLabeling → YOLO 标签转换")
    parser.add_argument("--data", required=True,
                        help="数据集名：wersion_parts（零件）/ wersion_steps（步骤）")
    parser.add_argument("--src", default=None,
                        help="标注所在目录；默认自动找 vision_yolo_train/annotations 与 "
                             "vision_yolo_train/<parts|steps>/<类别>")
    parser.add_argument("--classes", default=None, help="类别表文件，默认用数据集 classes.txt")
    parser.add_argument("--dry-run", action="store_true", help="只报告不写文件")
    parser.add_argument("--copy-images", action="store_true", default=True,
                        help="把标注目录里的图片汇总到 images/raw（默认开）")
    args = parser.parse_args()

    dataset_dir = paths.DATASET_DIR[args.data]
    classes_file = Path(args.classes) if args.classes else dataset_dir / "classes.txt"
    if not classes_file.is_file():
        print(f"[错误] 找不到类别表 {classes_file}；先跑 init_datasets.bat")
        return 2
    names = read_names(classes_file)
    pose = False            # 手部 pose 数据集已移除，只做检测框转换
    print(f"[数据集] {args.data}（{'pose' if pose else 'detect'}）类别 {names}")

    # 找标注来源
    sources: list[Path] = []
    if args.src:
        sources = [Path(args.src)]
    else:
        candidates = [paths.ANNOTATIONS]
        kind = "parts" if args.data == "wersion_parts" else (
            "steps" if args.data == "wersion_steps" else None)
        if kind:
            candidates += [p for p in sorted((paths.ROOT / kind).glob("*")) if p.is_dir()]
        sources = [p for p in candidates if p.is_dir()]
        if not sources:
            print("[错误] 没找到标注目录；用 --src 指定，或先把 X-AnyLabeling 导出放到 "
                  f"{paths.ROOT / 'annotations'}")
            return 2
    print(f"[来源] {', '.join(str(s) for s in sources)}")

    out_images = dataset_dir / "images" / "raw"
    out_labels = dataset_dir / "labels" / "raw"
    if not args.dry_run:
        out_images.mkdir(parents=True, exist_ok=True)
        out_labels.mkdir(parents=True, exist_ok=True)

    images = [p for src in sources for p in sorted(src.rglob("*"))
              if p.suffix.lower() in IMAGE_SUFFIXES]
    json_files = [p for src in sources for p in sorted(src.rglob("*.json"))]
    txt_files = [p for src in sources for p in sorted(src.rglob("*.txt"))
                 if p.name.lower() not in ("classes.txt", "readme.txt")]

    converted = passthrough = empty = 0
    errors: list[str] = []
    warnings: list[str] = []

    # A. 已经是 YOLO 标签的 .txt：直接收集
    for txt in txt_files:
        content = [ln for ln in txt.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if not content:
            empty += 1
            warnings.append(f"{txt.name}: 空标签文件（可能是负样本，确认后保留）")
        if args.dry_run:
            passthrough += 1
            continue
        shutil.copy2(txt, out_labels / f"{txt.stem}.txt")
        passthrough += 1

    # B. 原生 JSON：转换
    for json_path in json_files:
        lines, errs, warns = convert_json(json_path, names, pose)
        errors += errs
        warnings += warns
        if args.dry_run:
            converted += 1 if lines else 0
            continue
        (out_labels / f"{json_path.stem}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        converted += 1 if lines else 0
        img = next((p for p in images if p.stem == json_path.stem), None)
        if img is not None:
            shutil.copy2(img, out_images / img.name)
        else:
            warnings.append(f"{json_path.name}: 同目录下没找到同名图片")

    # 顺便把没标 JSON 但有图的情况报出来，这种往往是漏标
    labelled = {p.stem for p in json_files} | {p.stem for p in txt_files}
    unlabelled = [p.name for p in images if p.stem not in labelled]

    print(f"\n[结果] 直接收集 YOLO 标签 {passthrough}，JSON 转换 {converted}，"
          f"空标签 {empty}")
    if unlabelled:
        print(f"[提醒] {len(unlabelled)} 张图没有对应标注（前 5 个：{unlabelled[:5]}）")
    if warnings:
        print(f"[警告] {len(warnings)} 条，前 10 条：")
        for item in warnings[:10]:
            print(f"   · {item}")
    if errors:
        print(f"[错误] {len(errors)} 条，前 10 条：")
        for item in errors[:10]:
            print(f"   · {item}")
    if not args.dry_run:
        print(f"[输出] {out_labels}\n        {out_images}")
    print("下一步：python vision_yolo_train/tools/split_dataset.py --data " + args.data)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
