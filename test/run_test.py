"""末端「无通讯」版 —— 一台机器上跑完，不连网、不调服务、不监听端口。

和 `edge/run_edge.py` 的区别：

| | edge/run_edge.py | test/run_test.py（本文件） |
|---|---|---|
| 推理后端 | 本机 torch / 本机 ONNX / **远程 HTTP** | **只有本机**（torch 或 ONNX） |
| 网络相关参数 | 有 | **一个都没有** |
| 用途 | 正式部署（树莓派 + 电脑） | **自己往里加逻辑的沙盒**，比如阶段识别 |

零件检测、料盘判定、卡尔曼跟踪、绘制、采集全部复用 `edge/` 里已经验证过的实现，
所以这里只有"主循环 + 你的逻辑"，好读也好改。

跑法：
    python test/run_test.py                                  # 相机 + 本机模型
    python test/run_test.py --source 某张图.png --snapshot   # 离线跑一张图
    python test/run_test.py --no-yolo                        # 只跑料盘判定（不加载模型）
    python test/run_test.py --stage-mode parts               # 阶段改按"零件组合"推断
    python test/run_test.py --stage-mode both                # 两条阶段路线一起跑，对照看

阶段识别有两条路，**互不干扰，可以只开一条也可以同时开**：

    template（默认）  基础视觉：把装配区裁剪图旋转 1° 遍历一圈，跟参考图比相似度
                      → edge/stage_match.py，参考图在 vision_yolo_train/images/steps
    parts            规则推断：按装配区里检出了哪些零件走到第几步
                      → test/step_recognizer.py（**你要改规则就改这个文件**）
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))     # 让 edge/ 可 import
sys.path.insert(0, str(HERE))            # 让同目录的 step_recognizer 可 import

from edge.args import DETECT_ARGUMENTS                                    # noqa: E402
from edge.camera import open_camera, probe_modes                          # noqa: E402
from edge.capture import CAPTURE_SPECS, CaptureDeck, on_mouse             # noqa: E402
from edge.constants import (IMAGE_SUFFIXES, WINDOW, ZONE_COLORS, PART_COLOR,       # noqa: E402
                            TRAIL_COLOR, GREEN_MIN_FILL_RATIO, DEFAULT_TRAY_MIN_FILL,
                            DEFAULT_PART_NAMES)
from edge.drawing import (draw_detection, draw_machine_state, draw_overlay,  # noqa: E402
                          draw_stage, draw_trail, draw_zone, load_cjk_font)
from edge.imaging import (contours_to_boxes, prepare_detect_input,          # noqa: E402
                          save_image, take_right)
from edge.parts import PartTracker                                        # noqa: E402
from edge.project import ROOT, ensure_dir                                 # noqa: E402
from edge.stage_match import StageMatcher                                 # noqa: E402
from edge.zone_tracker import ZoneTracker, load_zones, parse_zones        # noqa: E402
from edge.zones import zone_mask                                          # noqa: E402

from step_recognizer import StepRecognizer                                # noqa: E402

# 这个版本**故意不提供**这些参数（它们都属于"通讯/上传"）
COMM_SKIP_FLAGS = {"--infer-backend", "--server-url", "--server-token", "--wire",
                   "--jpeg-quality", "--server-timeout", "--upload-crop"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="末端无通讯版：相机 + 料盘判定 + 本机零件检测 + 阶段识别（自己扩展）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # ---- 相机/输入 ----
    parser.add_argument("--camera", type=int, default=0, help="相机序号")
    parser.add_argument("--source", default=None, help="视频/图片路径（给了就不用相机）")
    parser.add_argument("--backend", choices=("auto", "dshow", "msmf"), default="dshow")
    parser.add_argument("--mode", default="2560x960", help="请求分辨率，形如 2560x960")
    parser.add_argument("--width", type=int, help="直接指定帧宽")
    parser.add_argument("--height", type=int, help="直接指定帧高")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--fourcc", default="MJPG")
    parser.add_argument("--buffersize", type=int, default=1)
    parser.add_argument("--layout", choices=("auto", "side_by_side", "top_bottom", "none"),
                        default="auto", help="auto 对半切；none 表示整帧即单目")
    parser.add_argument("--warmup", type=int, default=30, help="启动丢弃帧数")
    parser.add_argument("--resize", type=int, default=0, help="缩放到该宽度（0 不缩放）")
    parser.add_argument("--roi-px", type=int, nargs=4, metavar=("X", "Y", "W", "H"))
    parser.add_argument("--no-window", action="store_true", help="不开预览窗口（无显示器时）")
    parser.add_argument("--snapshot", action="store_true", help="抓一帧保存后退出")
    parser.add_argument("--save", action="store_true", help="把带识别结果的画面存到 --save-dir")
    parser.add_argument("--probe", action="store_true", help="只测各模式帧率后退出")
    parser.add_argument("--save-dir", default="output")
    parser.add_argument("--no-capture", action="store_true", help="不显示采集按钮")
    parser.add_argument("--capture-overlay", action="store_true",
                        help="采集时把识别结果也存进图片（默认存干净帧）")
    parser.add_argument("--capture-root", default=None)
    parser.add_argument("--no-detect", action="store_true", help="关掉识别，只看画面")
    # ---- 识别相关（直接复用 edge/ 的参数定义，去掉通讯那一组）----
    # 注意：--no-yolo / --no-trail 等都在 DETECT_ARGUMENTS 里，这里不要再声明一遍，
    # 否则 argparse 会报 "conflicting option string"。
    group = parser.add_argument_group("料盘识别 + 零件检测 + 跟踪")
    for flags, options in DETECT_ARGUMENTS:
        if flags[0] in COMM_SKIP_FLAGS:
            continue
        group.add_argument(*flags, **options)
    parser.add_argument("--stage-mode", choices=("template", "parts", "both"),
                        default="template",
                        help="阶段识别走哪条路：template=基础视觉旋转匹配（edge/stage_match.py）"
                             " | parts=按装配区零件组合推断（test/step_recognizer.py）"
                             " | both=两条都跑，画面上对照看")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    detect = not args.no_detect

    if args.probe:
        probe_modes(args.camera, args.backend, args.fourcc)
        return 0

    if args.width and args.height:
        width, height = args.width, args.height
    else:
        try:
            width, height = (int(v) for v in args.mode.lower().split("x"))
        except ValueError:
            raise SystemExit(f"--mode 格式应为 宽x高，例如 2560x960，收到 {args.mode!r}")

    # ---------------- 输入源 ----------------
    source_path = Path(args.source).expanduser() if args.source else None
    if source_path is not None:
        if not source_path.is_absolute() and not source_path.is_file():
            source_path = ROOT / source_path
        source_path = source_path.resolve()
    is_image = bool(source_path and source_path.suffix.lower() in IMAGE_SUFFIXES)
    is_video = bool(source_path and not is_image)
    if source_path and not source_path.is_file():
        raise SystemExit(f"找不到文件：{source_path}")

    frame0, cap = None, None
    if is_image:
        frame0 = cv2.imdecode(np.fromfile(str(source_path), dtype=np.uint8),
                              cv2.IMREAD_COLOR)
        if frame0 is None:
            raise SystemExit(f"图片读取失败：{source_path}")
        info = {"width": frame0.shape[1], "height": frame0.shape[0],
                "fps": 0.0, "fourcc": "IMG"}
        print(f"图片源 {source_path.name} {info['width']}x{info['height']}")
    elif is_video:
        cap = cv2.VideoCapture(str(source_path))
        if not cap.isOpened():
            raise SystemExit(f"视频打不开：{source_path}")
        info = {"width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "fps": cap.get(cv2.CAP_PROP_FPS) or 0.0,
                "fourcc": "VID"}
        print(f"视频源 {source_path.name} {info['width']}x{info['height']}")
    else:
        cap, info = open_camera(args.camera, args.backend, width, height, args.fps,
                                args.fourcc, args.buffersize)
        print(f"相机 → 实际 {info['width']}x{info['height']} "
              f"@{info['fps']:.1f} {info['fourcc']}")

    # ---------------- 区域（料盘）------------ 只读本机文件，不联网 ----------------
    zone_file = Path(args.zones_file).expanduser() if args.zones_file else \
        ROOT / "zones.json"
    if not zone_file.is_absolute():
        zone_file = ROOT / zone_file
    preset_zones = parse_zones(args.zones) if args.zones else load_zones(zone_file)
    zone_mode = args.zone_mode
    green_lo, green_hi = tuple(args.green_lo), tuple(args.green_hi)
    label_font = load_cjk_font(args.label_size) if detect else None

    # ---------------- 零件（**只用本机后端**）----------------
    tracker_parts = None
    if detect and not args.no_yolo:
        weights = Path(args.yolo_weights).expanduser() if args.yolo_weights else \
            ROOT / "vision_yolo_train/runs/wersion_parts/weights/best.pt"
        if not weights.is_absolute():
            weights = ROOT / weights
        backend = "onnx" if weights.suffix.lower() == ".onnx" else "ultralytics"
        tracker_parts = PartTracker(weights.resolve(), backend=backend,
                                    conf=args.yolo_conf, iou=args.yolo_iou,
                                    imgsz=args.yolo_imgsz, device=args.yolo_device,
                                    trail_len=args.trail_len,
                                    track_max_age=args.track_max_age,
                                    kf_acc=args.kf_acc, kf_meas=args.kf_meas)
        print(tracker_parts.describe())
        if not tracker_parts.enabled:
            print(f"  → 零件检测未启用（{tracker_parts.runner.error}）")

    # ---------------- 你的逻辑：阶段识别（两条路，可同时跑）----------------
    # template：基础视觉，把装配区那一块裁剪图旋转一圈跟参考图比相似度（不依赖 YOLO）
    # parts   ：按装配区里出现了哪些零件推断（规则写在 test/step_recognizer.py）
    use_template = detect and (args.stage_mode in ("template", "both")) and not args.no_stage
    use_parts = detect and (args.stage_mode in ("parts", "both"))

    stage = None
    if use_template:
        stage_dir = Path(args.stage_dir).expanduser()
        if not stage_dir.is_absolute():
            stage_dir = ROOT / stage_dir
        stage = StageMatcher.from_directory(
            stage_dir, size=args.stage_size, angle_step=args.angle_step,
            angle_range=(args.angle_min, args.angle_max),
            refine_step=args.refine_step, margin=args.stage_margin,
            rotate=not args.no_rotate, min_score=args.stage_min_score)
        print(stage.describe())
        if not stage.ready:
            print(f"  参考图目录 {stage_dir} 里没有可用图片 → 本版块跳过")
            stage = None
        else:
            window = (f"--stage-region {' '.join(str(v) for v in args.stage_region)}"
                      if args.stage_region else "锁定后的装配区框")
            print(f"  裁剪窗口 {window} | 每 {args.stage_interval} 帧认一次"
                  f" | 相似度 <{args.stage_min_score:.2f} 记 unknown")

    step_recognizer = StepRecognizer() if use_parts else None
    if detect and not use_template and not use_parts:
        print("阶段识别：已关闭（--no-stage 或 --stage-mode 组合导致两条路都不走）")

    save_dir = Path(args.save_dir).expanduser()
    if not save_dir.is_absolute():
        save_dir = ROOT / save_dir
    save_dir = ensure_dir(save_dir.resolve(), "输出目录")
    print(f"项目根目录：{ROOT}")
    print(f"输出目录  ：{save_dir}")
    show_window = ((not args.snapshot) and (not args.no_window)
                   and not is_image and hasattr(cv2, "imshow"))
    if show_window:
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    else:
        print("无 GUI 或 --snapshot/图片源：只保存图像，不显示窗口")

    tracker_zones = ZoneTracker(occlude_ratio=args.occlude_ratio, zones=preset_zones,
                                zones_file=(None if (args.no_zone_save or args.zones)
                                            else zone_file),
                                lock_frames=args.zone_lock_frames,
                                cover_margin=args.cover_margin,
                                cover_delta=args.cover_delta,
                                cover_window=args.cover_window)
    detections: list[dict] = []
    step_text = "—"                  # parts 路线（零件组合）的结果
    stage_result = None              # template 路线（旋转匹配）的结果；间隔之间沿用
    stage_cost = 0.0                 # 上次模板匹配耗时（ms）

    capture_enabled = not args.no_capture
    capture_root = Path(args.capture_root).expanduser().resolve() if args.capture_root \
        else ROOT
    deck = CaptureDeck(capture_root, font=label_font, enabled=capture_enabled)
    if capture_enabled:
        deck.layout(frame0 if is_image else
                    np.empty((max(info["height"], 1), max(info["width"], 1), 3),
                             dtype=np.uint8))
        print("采集按钮（点按钮或按数字键）：")
        for name, prefix, rel, hotkey in CAPTURE_SPECS:
            start = CaptureDeck.scan_highest(capture_root / rel, prefix) + 1
            print(f"   [{hotkey}] {name:10} -> {rel}\\{prefix}_N.png   从 {prefix}_{start} 续起")
        if show_window:
            cv2.setMouseCallback(WINDOW, on_mouse, deck)

    frame_count, last_time, last_count, fps_now = 0, time.time(), 0, 0.0
    keys_pressed = 255
    try:
        while True:
            if is_image:
                if frame_count:
                    break
                frame, frame_count = frame0, frame_count + 1
            else:
                ok, frame = cap.read()
                if not ok:
                    print(f"视频结束，共处理 {frame_count} 帧" if is_video else "读取画面失败")
                    break
                frame_count += 1
                if frame_count <= args.warmup or float(frame[::8, ::8].mean()) < 12:
                    continue

            view = take_right(frame, args.layout)
            if args.resize and args.resize != view.shape[1]:
                scale = args.resize / float(view.shape[1])
                view = cv2.resize(view, (args.resize, int(round(view.shape[0] * scale))),
                                  interpolation=cv2.INTER_AREA)
            clean_view = view.copy() if capture_enabled else None
            if args.roi_px:
                x, y, w, h = args.roi_px
                cv2.rectangle(view, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # ---------- 料盘区域 ----------
            if detect and (frame_count == 1
                           or frame_count % max(args.zone_interval, 1) == 0):
                zone_input, zone_scale = prepare_detect_input(view, args.zone_width)
                mask = zone_mask(zone_input, zone_mode, green_lo=green_lo,
                                 green_hi=green_hi, v_max=args.dark_v_max,
                                 scan_top=args.zone_scan_top)
                fill_floor = (GREEN_MIN_FILL_RATIO if zone_mode == "green"
                              else DEFAULT_TRAY_MIN_FILL)
                boxes = contours_to_boxes(mask, zone_input.shape[1], zone_input.shape[0],
                                          min_area_ratio=args.min_area,
                                          min_fill_ratio=fill_floor,
                                          merge_gap=args.merge_gap)
                zone_area = float(zone_input.shape[0] * zone_input.shape[1])
                boxes = [b for b in boxes if b[4] >= args.zone_min_area * zone_area]
                v_channel = (cv2.cvtColor(zone_input, cv2.COLOR_BGR2HSV)[:, :, 2]
                             if zone_mode != "green" else None)
                tracker_zones.update(boxes, mask, zone_area, v_channel, zone_scale)

            # ---------- 零件（本机推理 + 每帧卡尔曼）----------
            if detect and tracker_parts is not None:
                if frame_count == 1 or frame_count % max(args.yolo_interval, 1) == 0:
                    detect_input, detect_scale = prepare_detect_input(view, args.detect_width)
                    tracker_parts.detect(detect_input, detect_scale,
                                         view_size=(view.shape[1], view.shape[0]))
                detections = tracker_parts.tick()

            # ================== 阶段识别：两条路互不干扰 ==================
            # template 路线：只吃「装配区」那一块裁剪图，跟零件框无关
            if stage is not None:
                if frame_count == 1 or frame_count % max(args.stage_interval, 1) == 0:
                    region = (tuple(args.stage_region) if args.stage_region
                              else tracker_zones.region("装配区"))
                    if region is not None:
                        started = time.time()
                        stage_result = stage.match(view, region=region)
                        stage_cost = (time.time() - started) * 1000.0

            # parts 路线：按装配区里检出的零件组合推断（规则在 test/step_recognizer.py）
            if step_recognizer is not None:
                step_text = step_recognizer.update(tracker_zones.zones, detections)
            # ==============================================================

            # ---------- 绘制 ----------
            if detect:
                status_height = draw_overlay(view, "")
                for index, zone in enumerate(tracker_zones.drawable()):
                    x1, y1, x2, y2 = zone["box"]
                    draw_zone(view, x1, y1, x2, y2, zone["name"],
                              ZONE_COLORS[index % len(ZONE_COLORS)], font=label_font,
                              top_margin=status_height, thickness=args.line_width)
                if not args.no_trail:
                    for det in detections:
                        if det["trail"]:
                            draw_trail(view, det["trail"], TRAIL_COLOR, thickness=1)
                for det in detections:
                    draw_detection(view, det["box"],
                                   f"{det['name']} {det['conf']:.2f}", PART_COLOR,
                                   font=label_font, thickness=args.line_width,
                                   top_margin=status_height)

                # 模板匹配的阶段标签贴装配区框的右上角（左上角是区域名）
                if stage is not None:
                    region = (tuple(args.stage_region) if args.stage_region
                              else tracker_zones.region("装配区"))
                    if region is not None:
                        rx, ry, rw, rh = region
                        if stage_result is None:
                            stage_text = "阶段 ?"
                        elif stage_result["label"] == "unknown":
                            stage_text = f"阶段 ? {stage_result['score']:.2f}"
                        else:
                            stage_text = (f"阶段 {stage_result['label']}"
                                          f" {stage_result['score']:.2f}"
                                          f" {stage_result['angle']:+.1f}°")
                        draw_stage(view, (rx, ry, rx + rw, ry + rh), stage_text,
                                   font=label_font, top_margin=status_height)

            now = time.time()
            if now - last_time >= 0.5:
                fps_now = (frame_count - last_count) / max(now - last_time, 1e-6)
                last_time, last_count = now, frame_count
            gray = cv2.cvtColor(view, cv2.COLOR_BGR2GRAY) if frame_count % 10 == 0 else None
            sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var()) if gray is not None \
                else 0.0
            status = (f"LOCAL {view.shape[1]}x{view.shape[0]} | fps {fps_now:.1f}"
                      f" | sharp {sharpness:.0f}")
            if detect:
                status += (f" | trays {len(tracker_zones.drawable())}"
                           f" | parts {len(detections)}")
                if tracker_parts is not None and tracker_parts.enabled:
                    status += f" | yolo {tracker_parts.last_cost_ms:.0f}ms"
                if stage is not None:
                    # 状态栏只能画英文：度数写 deg
                    if stage_result is None:
                        status += " | stage -"
                    else:
                        status += (f" | stage {stage_result['label']}"
                                   f" {stage_result['score']:.2f}"
                                   f" {stage_result['angle']:+.1f}deg"
                                   f" {stage_cost:.0f}ms")
                if step_recognizer is not None:
                    status += f" | step {step_text}"
            draw_overlay(view, status)
            annotate = view

            if detect:
                draw_machine_state(view, tracker_zones.machine_state(),
                                   font=label_font, top=status_height)

            # ---------- 采集 ----------
            if capture_enabled:
                hit_index = deck.hotkey_index(keys_pressed)
                if hit_index is None and deck.pending is not None:
                    hit_index, deck.pending = deck.pending, None
                if hit_index is not None:
                    shot = view if args.capture_overlay else clean_view
                    saved = deck.capture(shot, hit_index)
                    if saved is not None:
                        print(f"已采集：{saved}")
                deck.layout(view)
                deck.draw(view)

            if args.save:
                out = (save_dir / f"识别结果_{source_path.stem}.png" if is_image else
                       save_dir / f"结果_{datetime.now():%Y%m%d_%H%M%S}.jpg")
                save_image(out, annotate)
                if is_image or frame_count % 100 == 0:
                    print(f"已保存：{out}")

            if args.snapshot:
                out = save_dir / "右目画面.jpg"
                save_image(out, annotate)
                print(f"已保存：{out}")
                if detect:
                    print(f"识别结果：料盘区域 {len(tracker_zones.drawable())} 个，"
                          f"零件 {len(detections)} 个")
                    if stage is not None:
                        if stage_result is None:
                            print("阶段（模板）：这一帧没认出来（装配区未锁定或裁剪非法）")
                        else:
                            print(f"阶段（模板）：{stage_result['label']}"
                                  f"（相似度 {stage_result['score']:.3f}，"
                                  f"摆角 {stage_result['angle']:+.1f}°，"
                                  f"耗时 {stage_cost:.0f} ms）")
                    if step_recognizer is not None:
                        print(f"阶段（零件组合）：{step_text}")
                break

            if show_window:
                cv2.imshow(WINDOW, annotate)
                key = cv2.waitKey(1) & 0xFF
                keys_pressed = key
                if key in (ord("q"), 27):
                    break
                if key == ord("s"):
                    out = save_dir / f"右目_{datetime.now():%Y%m%d_%H%M%S}.jpg"
                    save_image(out, annotate)
                    print(f"已保存：{out}")
            else:
                keys_pressed = 255
    finally:
        if cap is not None:
            cap.release()
        if show_window:
            cv2.destroyAllWindows()
        if capture_enabled:
            print(deck.summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
