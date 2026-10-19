"""把标注转成 YOLO 标签（支持 X-AnyLabeling / labelImg / labelme 的常见输出）。

支持四种输入，脚本自动识别，可以混着放：
    A. YOLO 格式的 .txt
       · X-AnyLabeling 的「导出 YOLO」产物（xlabel2yolo）
       · labelImg 切到 YOLO 模式后的产物
       → 原样收集，不做坐标转换
    B. X-AnyLabeling 原生 JSON（每张图一个同名 .json）→ 转成 YOLO 格式
       识别 rectangle（检测框）与 polygon（取外接矩形）
    C. labelImg 默认的 PascalVOC XML（每张图一个同名 .xml）→ 转成 YOLO 格式
       这样即使忘了切 YOLO 模式、存成 XML 也能用
    D. labelme 的 JSON（shapes 里 points 为多边形）→ 按外接矩形转换

用法：
    # 最常用：标完直接转（读 vision_yolo_train/annotations/）
    python vision_yolo_train/tools/convert_anylabeling.py --data wersion_parts

    # 先干跑看结果，不写文件
    python vision_yolo_train/tools/convert_anylabeling.py --data wersion_parts --dry-run

    # 转换后把源文件移走（避免下次重复收集，适合反复标注的场景）
    python vision_yolo_train/tools/convert_anylabeling.py --data wersion_parts --move

输出：
    datasets/<data>/images/raw/*    图片汇总
    datasets/<data>/labels/raw/*.txt YOLO 标签（未划分，交给 split_dataset.py 分）
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import xml.etree.ElementTree as ET
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
    # 统一对所有点取外接矩形。
    # 注意：X-AnyLabeling 的矩形存的是 4 个角点（左上/右上/右下/左下），
    # labelme 的矩形只存 2 个对角点。以前这里固定取 points[0]、points[1] 当对角，
    # 碰到 4 角点算出来的就是"同一条上边"，高度为 0 →
    # 每张图都报"检测框宽或高小于 1 像素"，标注全丢。
    # 改成对全部点取外接矩形后，两种格式都能正确转成 YOLO 标签。
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


def convert_voc_xml(xml_path: Path, names: list[str]
                    ) -> tuple[list[str], list[str], list[str]]:
    """PascalVOC XML → YOLO 标签行。

    labelImg 默认存这个格式（除非手动点 YOLO 按钮）。支持它等于给标注上了保险：
    就算忘了切格式、存成 XML，转换也照样能读，不用重标。

    XML 结构：
        <annotation><size><width>W</width><height>H</height></size>
          <object><name>base</name><bndbox>
            <xmin>..</xmin><ymin>..</ymin><xmax>..</xmax><ymax>..</ymax>
          </bndbox></object>
        </annotation>
    """
    lines: list[str] = []
    errors: list[str] = []
    warnings: list[str] = []
    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError as error:
        return [], [f"{xml_path.name}: XML 解析失败（{error}）"], []

    size = root.find("size")
    width = height = 0
    if size is not None:
        width = int(float(size.findtext("width", "0") or 0))
        height = int(float(size.findtext("height", "0") or 0))
    if width <= 0 or height <= 0:                 # 退而求其次：从 <path> 读图尺寸
        errors.append(f"{xml_path.name}: 缺少合法的 <size> 宽高，无法归一化")
        return [], errors, []

    for obj in root.findall("object"):
        label = (obj.findtext("name") or "").strip()
        if label not in names:
            errors.append(f"{xml_path.name}: 未知类别 '{label}'（不在 {names}）")
            continue
        box = obj.find("bndbox")
        if box is None:
            errors.append(f"{xml_path.name}: 类别 {label} 缺少 <bndbox>")
            continue
        try:
            x1 = float(box.findtext("xmin", "0"))
            y1 = float(box.findtext("ymin", "0"))
            x2 = float(box.findtext("xmax", "0"))
            y2 = float(box.findtext("ymax", "0"))
        except ValueError:
            errors.append(f"{xml_path.name}: 类别 {label} 的坐标不是数字")
            continue
        x1, x2 = sorted((x1, x2))
        y1, y2 = sorted((y1, y2))
        if x2 - x1 < 1 or y2 - y1 < 1:
            errors.append(f"{xml_path.name}: 类别 {label} 的框宽或高小于 1 像素")
            continue
        class_id = names.index(label)
        cx = (x1 + x2) / 2 / width
        cy = (y1 + y2) / 2 / height
        bw = (x2 - x1) / width
        bh = (y2 - y1) / height
        lines.append(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

    if not lines:
        warnings.append(f"{xml_path.name}: 没有任何可用标注（空图？）")
    return lines, errors, warnings


def main() -> int:
    parser = argparse.ArgumentParser(description="标注 → YOLO 标签转换")
    parser.add_argument("--data", required=True,
                        help="数据集名：wersion_parts（零件）/ wersion_steps（步骤）")
    parser.add_argument("--src", default=None,
                        help="标注所在目录；默认自动找 vision_yolo_train/annotations 与 "
                             "vision_yolo_train/<parts|steps>/<类别>")
    parser.add_argument("--classes", default=None,
                        help="类别表文件；默认用 configs/classes_<数据集>.txt")
    parser.add_argument("--dry-run", action="store_true", help="只报告不写文件")
    parser.add_argument("--move", action="store_true",
                        help="转换后把源文件移走（而非复制），避免重复标注时反复堆积")
    parser.add_argument("--copy-images", action="store_true", default=True,
                        help="把标注目录里的图片汇总到 images/raw（默认开）")
    args = parser.parse_args()

    dataset_dir = paths.DATASET_DIR[args.data]
    # 类别表统一放在 configs/：classes_wersion_parts.txt / classes_wersion_steps.txt
    if args.classes:
        classes_file = Path(args.classes)
    else:
        classes_file = paths.CONFIGS / f"classes_{args.data}.txt"
        if not classes_file.is_file():                    # 兜底：数据集目录下的旧位置
            classes_file = dataset_dir / "classes.txt"
    if not classes_file.is_file():
        print(f"[错误] 找不到类别表 {classes_file}\n"
              f"       期望：{paths.CONFIGS / f'classes_{args.data}.txt'}\n"
              f"       可用 --classes 指定，或跑 init_datasets.py 重建")
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
            # X-AnyLabeling 直接在 images/parts/ 里打开图片时，json 是存在图片旁边的
            # （自动保存默认就写同目录）。所以默认来源必须包含它，否则标完跑转换会
            # 报"没找到标注目录"，让人以为白标了。
            candidates += [paths.IMAGE_GROUPS[kind]]
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

    # 图片查找范围：来源目录优先，再兜底扫常见放图位置。
    # 为什么需要兜底：标注流程是 annotate_parts.bat 打开 datasets/annotate/parts/<类别>/images/、
    # 把 json 导出到 annotations/，此时 annotations/ 里只有 json 没有图。
    # 若只在来源目录里找同名图片，结果是"标签收进来了、图片一张没收集"，
    # 下一步 split_dataset.py 直接报"没有图片"。
    image_roots = [*sources,
                   paths.IMAGE_GROUPS["parts"], paths.IMAGE_GROUPS["steps"],
                   paths.ANNOTATE]
    image_index: dict[str, Path] = {}
    for root in image_roots:
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*")):
            if path.suffix.lower() in IMAGE_SUFFIXES:
                image_index.setdefault(path.stem, path)
    for path in images:                      # 来源目录里的图优先于兜底位置
        image_index[path.stem] = path

    def find_image(stem: str) -> Path | None:
        return image_index.get(stem)

    def place_image(image: Path) -> None:
        """收集图片：兜底位置找到的图一律复制，只有来源目录里的图跟随 --move。"""
        if move and any(image.is_relative_to(src) for src in sources):
            shutil.move(str(image), str(out_images / image.name))
        else:
            shutil.copy2(image, str(out_images / image.name))

    json_files = [p for src in sources for p in sorted(src.rglob("*.json"))]
    txt_files = [p for src in sources for p in sorted(src.rglob("*.txt"))
                 if p.name.lower() not in ("classes.txt", "readme.txt")]
    xml_files = [p for src in sources for p in sorted(src.rglob("*.xml"))]

    converted = passthrough = empty = xml_converted = 0
    errors: list[str] = []
    warnings: list[str] = []
    move = bool(args.move)

    def place(source: Path, target: Path) -> None:
        """按 --move 决定搬运还是复制。"""
        if move:
            shutil.move(str(source), str(target))
        else:
            shutil.copy2(source, target)

    # A. 已经是 YOLO 标签的 .txt：直接收集
    for txt in txt_files:
        content = [ln for ln in txt.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if not content:
            empty += 1
            warnings.append(f"{txt.name}: 空标签文件（可能是负样本，确认后保留）")
        if args.dry_run:
            passthrough += 1
            continue
        place(txt, out_labels / f"{txt.stem}.txt")
        passthrough += 1
        # --move 时图片一并搬走，否则源目录会残留图片
        img = find_image(txt.stem)
        if img is not None:
            place_image(img)
        else:
            warnings.append(f"{txt.name}: 没找到同名图片（只收集了标签）")

    # B. 原生 JSON（X-AnyLabeling / labelme）：转换
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
        img = find_image(json_path.stem)
        if img is not None:
            place_image(img)
        else:
            warnings.append(f"{json_path.name}: 没找到同名图片（只写了标签，图片未收集）")
        if move:
            json_path.unlink(missing_ok=True)

    # C. PascalVOC XML（labelImg 默认格式）：转换。
    #    有它兜底，就算标注时忘了切 YOLO 模式也不会白标。
    for xml_path in xml_files:
        lines, errs, warns = convert_voc_xml(xml_path, names)
        errors += errs
        warnings += warns
        if args.dry_run:
            xml_converted += 1 if lines else 0
            continue
        (out_labels / f"{xml_path.stem}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        xml_converted += 1 if lines else 0
        img = find_image(xml_path.stem)
        if img is not None:
            place_image(img)
        else:
            # 注意：这里的文件名必须用 xml_path。以前误写成上一轮 json 循环的残留变量，
            # 报错信息会指向另一张图，排查时会被带偏。
            warnings.append(f"{xml_path.name}: 没找到同名图片（只写了标签，图片未收集）")
        if move:
            xml_path.unlink(missing_ok=True)

    # 顺便把没标注但有图的情况报出来，这种往往是漏标
    labelled = ({p.stem for p in json_files} | {p.stem for p in txt_files}
                | {p.stem for p in xml_files})
    unlabelled = [p.name for p in images if p.stem not in labelled]

    print(f"\n[结果] 直接收集 YOLO 标签 {passthrough}，JSON 转换 {converted}，"
          f"XML 转换 {xml_converted}，空标签 {empty}")
    if move and not args.dry_run:
        print("[搬运] 源文件已移走（--move），annotations/ 不会重复堆积")
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
