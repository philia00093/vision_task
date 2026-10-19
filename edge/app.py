"""主循环：采集 → 料盘判定 → 零件推理 → 绘制 → 按键/存图。

上游模块都只是"工具"，这里才是把节奏串起来的编排层：两种节流（--zone-interval / --yolo-interval）+ 每帧卡尔曼补框。"""

from __future__ import annotations

import numpy as np
import cv2
import time
from datetime import datetime
from pathlib import Path

from edge.args import build_parser
from edge.camera import fourcc_str, open_camera, probe_modes
from edge.capture import CaptureDeck, on_mouse
from edge.constants import CAPTURE_SPECS, DEFAULT_TRAY_MIN_FILL, DEFAULT_YOLO_WEIGHTS, DEFAULT_ZONES_FILE, GREEN_MIN_FILL_RATIO, HAND_COLOR, IMAGE_SUFFIXES, PART_COLOR, TRAIL_COLOR, WINDOW, ZONE_COLORS
from edge.drawing import draw_detection, draw_machine_state, draw_overlay, draw_stage, draw_trail, draw_zone, load_cjk_font
from edge.imaging import contours_to_boxes, prepare_detect_input, save_image, take_right
from edge.parts import PartTracker
from edge.stage_match import StageMatcher
from edge.project import ROOT, ensure_dir
from edge.zone_tracker import ZoneTracker, load_zones, parse_zones
from edge.zones import zone_mask


def main():
    args = build_parser().parse_args()
    detect = not args.no_detect

    if args.probe:
        probe_modes(args.camera, args.backend, args.fourcc)
        return

    if args.width and args.height:
        width, height = args.width, args.height
    else:
        try:
            width, height = (int(v) for v in args.mode.lower().split("x"))
        except ValueError:
            raise SystemExit(f"--mode 格式应为 宽x高，例如 2560x960，收到 {args.mode!r}")

    # 视频源：--source 给的是视频/图片文件时走离线路径，否则开相机
    source_path = Path(args.source).expanduser() if args.source else None
    if source_path is not None:
        # 相对路径按"当前工作目录"解析；找不到再试项目根目录，并打印解析结果
        if not source_path.is_absolute() and not source_path.is_file():
            source_path = ROOT / source_path
        source_path = source_path.resolve()
    is_image = bool(source_path and source_path.suffix.lower() in IMAGE_SUFFIXES)
    is_video = bool(source_path and not is_image)
    if source_path and not source_path.is_file():
        raise SystemExit(f"找不到文件：{source_path}\n"
                         f"  已按「当前目录」和「项目根 {ROOT}」查找过；"
                         f"请检查路径或改用绝对路径")

    if is_image:
        frame0 = cv2.imdecode(np.fromfile(str(source_path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame0 is None:
            raise SystemExit(f"图片读取失败：{source_path}")
        info = {"width": frame0.shape[1], "height": frame0.shape[0], "fps": 0.0,
                "fourcc": "IMG"}
        cap = None
        print(f"图片源 {source_path.name} {info['width']}x{info['height']}")
    elif is_video:
        cap = cv2.VideoCapture(str(source_path))
        if not cap.isOpened():
            raise SystemExit(f"视频打不开：{source_path}")
        info = {"width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "fps": cap.get(cv2.CAP_PROP_FPS) or 0.0,
                "fourcc": fourcc_str(cap.get(cv2.CAP_PROP_FOURCC))}
        print(f"视频源 {source_path.name} {info['width']}x{info['height']} @{info['fps']:.1f}")
    else:
        cap, info = open_camera(args.camera, args.backend, width, height, args.fps,
                                args.fourcc, args.buffersize)
        print(f"请求 {width}x{height}@{args.fps} {args.fourcc} → "
              f"实际 {info['width']}x{info['height']} "
              f"@{info['fps']:.1f} {info['fourcc']}")
        if info["fourcc"].upper() != args.fourcc.upper():
            print(f"警告：相机未按 {args.fourcc} 协商（当前 {info['fourcc']}）；"
                  "YUY2 在 2560x960 下仅 5 FPS，会非常卡。可试 --backend msmf 或降分辨率。")
        if (info["width"], info["height"]) != (width, height):
            print(f"警告：实际分辨率与请求不一致，驱动可能回退到了默认模式")

    if args.fps_test and cap is not None:
        for _ in range(args.warmup):                      # 预热
            cap.read()
        count, start, shape = 0, time.time(), None
        for _ in range(args.fps_test):
            ok, frame = cap.read()
            if not ok:
                break
            view = take_right(frame, args.layout)          # 与实时流程相同的处理量
            if args.resize and args.resize != view.shape[1]:
                scale = args.resize / float(view.shape[1])
                view = cv2.resize(view, (args.resize, int(round(view.shape[0] * scale))),
                                  interpolation=cv2.INTER_AREA)
            _ = draw_overlay(view.copy(), "fps-test")     # 计入与实时流程相同的绘制开销
            count += 1
            shape = frame.shape
        elapsed = max(time.time() - start, 1e-6)
        print(f"端到端实测：{count} 帧 / {elapsed:.2f} s = {count / elapsed:.1f} FPS"
              f"（{info['width']}x{info['height']} {info['fourcc']}）")
        cap.release()
        return

    # 识别准备：料盘分割方式 + 零件模型，参数全部来自命令行
    green_lo = tuple(args.green_lo)
    green_hi = tuple(args.green_hi)
    zone_mode = args.zone_mode
    label_font = load_cjk_font(args.label_size) if detect else None
    # 区域来源：--zones 手工指定 → zones.json（上次锁定的结果）→ 都没有就自动识别
    zone_file = Path(args.zones_file).expanduser() if args.zones_file else \
        ROOT / DEFAULT_ZONES_FILE
    if not zone_file.is_absolute():
        zone_file = ROOT / zone_file
    preset_zones = parse_zones(args.zones) if args.zones else load_zones(zone_file)
    zone_source = ("命令行 --zones" if args.zones else
                   (f"{zone_file.name}（上次锁定的）" if preset_zones else "自动识别中"))
    # 零件模型：--no-yolo 或加载失败都不报错退出，只是降级成"只识别料盘"
    tracker_parts = None
    stage = None
    if detect and not args.no_yolo:
        # ONNX 后端默认找同名 .onnx；其余用 .pt
        default_weights = ROOT / DEFAULT_YOLO_WEIGHTS
        if args.infer_backend == "onnx":
            default_weights = default_weights.with_suffix(".onnx")
        weights = Path(args.yolo_weights).expanduser() if args.yolo_weights else default_weights
        if not weights.is_absolute():
            weights = ROOT / weights
        tracker_parts = PartTracker(weights.resolve(), backend=args.infer_backend,
                                    conf=args.yolo_conf, iou=args.yolo_iou,
                                    imgsz=args.yolo_imgsz, device=args.yolo_device,
                                    trail_len=args.trail_len,
                                    track_max_age=args.track_max_age,
                                    kf_acc=args.kf_acc, kf_meas=args.kf_meas,
                                    server_url=args.server_url,
                                    server_token=args.server_token,
                                    jpeg_quality=args.jpeg_quality,
                                    server_timeout=args.server_timeout,
                                    wire=args.wire)
    if detect:
        if zone_mode == "green":
            print(f"料盘识别：绿色 HSV {green_lo}~{green_hi}（--zone-mode green）")
        else:
            print(f"料盘识别：黑色料盘 V<{args.dark_v_max}，只看画面下 "
                  f"{(1.0 - args.zone_scan_top) * 100:.0f}%（顶部黑色机器臂已排除）")
        print(f"          最小面积 {args.min_area:.1%} | 区域面积门槛 {args.zone_min_area:.1%}"
              f" | 料盘判定每 {args.zone_interval} 帧"
              f"（宽 {args.zone_width or '原分辨率'}）"
              f" | YOLO 每 {args.yolo_interval} 帧"
              f"（输入宽 {args.detect_width or '原分辨率'}）")
        print(f"区域来源：{zone_source}；左→蓝框「装配区」，右→红框「物料区」"
              f"，中文字体 {'已载入' if label_font else '未找到（标签用拼音代替）'}")
        if not preset_zones:
            print(f"          自动识别：连续 {args.zone_lock_frames} 帧都认出两个料盘就锁定"
                  + (f"并写入 {zone_file.name}" if not args.no_zone_save else ""))
            print("          锁不准就用手工指定：--zones \"x1,y1,x2,y2;x1,y1,x2,y2\""
                  "（前=左=装配区，后=右=物料区）")
        if tracker_parts is not None:
            print(tracker_parts.describe())
            if not tracker_parts.enabled:
                print(f"  → 已降级为「只识别料盘」：{tracker_parts.runner.error}")
                if args.infer_backend == "http":
                    print("     先在电脑上起服务：")
                    print("       D:\\anaconda3\\envs\\yolo26\\python.exe server\\run_server.py "
                          "--host 0.0.0.0")
                    print(f"     再确认树莓派能访问 {args.server_url}"
                          "（防火墙放行 / 同一网段）")
                else:
                    print("     用 yolo26 环境跑可启用本机零件跟踪："
                          "D:\\anaconda3\\envs\\yolo26\\python.exe edge\\run_edge.py")
        else:
            print("零件跟踪：已用 --no-yolo 关闭")

        # ---- 阶段识别：基础视觉（旋转遍历 + 模板匹配），只吃装配区裁剪图 ----
        # 和上面零件跟踪并列共存：一个管"每件零件在哪"，一个管"装到第几步"，互不等待。
        if args.no_stage:
            print("阶段识别：已用 --no-stage 关闭")
        else:
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
                print(f"         参考图目录：{stage_dir}")
                print("         平铺 4 张（文件名=阶段名）或每阶段建子目录放多张样本即可启用")
                stage = None
            else:
                window = (f"--stage-region {' '.join(str(v) for v in args.stage_region)}"
                          if args.stage_region else "锁定后的装配区框")
                print(f"         裁剪窗口：{window} | 每 {args.stage_interval} 帧认一次"
                      f" | 相似度 <{args.stage_min_score:.2f} 记为 unknown")
                if args.stage_region is None:
                    print("         区域锁定前不输出阶段（没有装配区框就不知道裁哪）"
                          "；也可用 --stage-region 固定窗口")

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
        print("无 GUI 或使用 --snapshot/图片源：只保存图像，不显示窗口")

    frame_count, last_time, last_count, fps_now = 0, time.time(), 0, 0.0
    sharpness, brightness = 0.0, 0.0
    keys_pressed = 255                                        # 上一帧的按键（255 = 无）
    tracker_zones = ZoneTracker(occlude_ratio=args.occlude_ratio, zones=preset_zones,
                                zones_file=(None if (args.no_zone_save or args.zones)
                                            else zone_file),
                                lock_frames=args.zone_lock_frames,
                                cover_margin=args.cover_margin, cover_delta=args.cover_delta,
                                cover_window=args.cover_window)
    detections: list[dict] = []      # 零件检测结果；识别间隔之间沿用上一帧
    stage_result = None              # 阶段识别结果；间隔之间沿用上一次（标签不闪）
    stage_cost = 0.0                 # 上一次阶段识别的耗时（ms，Pi 上调参数看这个数）

    # ---- 数据采集按钮：图片落到 vision_yolo_train/images/parts/ ----
    capture_enabled = not args.no_capture
    capture_root = Path(args.capture_root).expanduser().resolve() if args.capture_root else ROOT
    deck = CaptureDeck(capture_root, font=label_font, enabled=capture_enabled)
    if capture_enabled:
        # 用画面尺寸先排一次版。注意 view 在循环里才产生，这里不能引用它，
        # 所以按已知的输出尺寸造一个临时占位数组（布局只看尺寸）。
        layout_size = (frame0 if is_image
                       else np.empty((max(info["height"], 1), max(info["width"], 1), 3),
                                     dtype=np.uint8))
        deck.layout(layout_size)
        print("采集按钮（点按钮或按数字键）：")
        for name, prefix, rel, hotkey in CAPTURE_SPECS:
            folder = capture_root / rel
            start = CaptureDeck.scan_highest(folder, prefix) + 1
            print(f"   [{hotkey}] {name:10} -> {rel}\\{prefix}_N.png   从 {prefix}_{start} 续起")
        if not show_window:
            print("   提示：当前没有窗口，按钮不可用（需相机/视频源且非 --snapshot）")
        else:
            cv2.setMouseCallback(WINDOW, on_mouse, deck)
    try:
        while True:
            if is_image:                                  # 图片源只处理一帧
                if frame_count:
                    break
                frame, frame_count = frame0, frame_count + 1
            else:
                ok, frame = cap.read()
                if not ok:
                    # 视频文件读完是正常结束；相机读不到才是真故障
                    print(f"视频结束，共处理 {frame_count} 帧" if is_video else "读取画面失败")
                    break
                frame_count += 1
                if frame_count <= args.warmup or float(frame[::8, ::8].mean()) < 12:
                    continue                              # 抽样亮度：比整帧 mean 快一个量级

            view = take_right(frame, args.layout)
            if args.resize and args.resize != view.shape[1]:
                scale = args.resize / float(view.shape[1])
                view = cv2.resize(view, (args.resize, int(round(view.shape[0] * scale))),
                                  interpolation=cv2.INTER_AREA)
            # 采集用的干净帧：在画任何东西（ROI/区域框/零件框/状态栏/按钮）之前留一份。
            # 采集时默认存这一份，免得把误识别的框烧进训练数据。
            clean_view = view.copy() if capture_enabled else None
            if args.roi_px:
                x, y, w, h = args.roi_px
                cv2.rectangle(view, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # ---------------- 料盘区域：每 --zone-interval 帧量一次 ----------------
            # 掩码可以在缩小图上算（--zone-width），Pi 上省 CPU；框会换算回原分辨率。
            now = time.time()
            # 第 1 帧强制跑一次：图片源只有一帧，否则永远轮不到；视频源也能立刻出框
            if detect and (frame_count == 1
                           or frame_count % max(args.zone_interval, 1) == 0):
                zone_input, zone_scale = prepare_detect_input(view, args.zone_width)
                mask = zone_mask(zone_input, zone_mode, green_lo=green_lo,
                                 green_hi=green_hi, v_max=args.dark_v_max,
                                 scan_top=args.zone_scan_top)
                # 填充率门槛按模式给：黑料盘中间有大圆凹槽，填充率天然偏低，
                # 沿用绿料盘时代的 0.55 会把料盘整个滤掉。
                fill_floor = (GREEN_MIN_FILL_RATIO if zone_mode == "green"
                              else DEFAULT_TRAY_MIN_FILL)
                boxes = contours_to_boxes(mask, zone_input.shape[1], zone_input.shape[0],
                                          min_area_ratio=args.min_area,
                                          min_fill_ratio=fill_floor,
                                          merge_gap=args.merge_gap)
                # 桌面上的小杂物按面积滤掉，否则它会抢走"物料区"的框
                zone_area = float(zone_input.shape[0] * zone_input.shape[1])
                zone_floor = args.zone_min_area * zone_area
                boxes = [b for b in boxes if b[4] >= zone_floor]
                # 遮挡判据二要用 V 通道（绿料盘模式用不到，省一次转换）
                v_channel = (cv2.cvtColor(zone_input, cv2.COLOR_BGR2HSV)[:, :, 2]
                             if zone_mode != "green" else None)
                tracker_zones.update(boxes, mask, zone_area, v_channel, zone_scale)

            # ---------------- 零件：每 --yolo-interval 帧跑一次 YOLO + 每帧卡尔曼预测 ----
            if detect and tracker_parts is not None:
                if frame_count == 1 or frame_count % max(args.yolo_interval, 1) == 0:
                    # 只上传"工作区"（--upload-crop）：面积小 → JPEG 小、编码快，
                    # 而且模型在同样 imgsz 下看到的零件更大更准。
                    crop = args.upload_crop
                    if crop:
                        cx, cy, cw, ch = crop
                        source = view[max(0, cy):max(0, cy) + cw, max(0, cx):max(0, cx) + ch]
                        if source.size == 0:            # 裁到画面外了，退回整幅
                            source, offset = view, (0.0, 0.0)
                        else:
                            offset = (float(cx), float(cy))
                    else:
                        source, offset = view, (0.0, 0.0)
                    detect_input, detect_scale = prepare_detect_input(source, args.detect_width)
                    tracker_parts.detect(detect_input, detect_scale, offset,
                                         view_size=(view.shape[1], view.shape[0]))
                    # --snapshot + http 后端：结果要一个来回才到，单帧抓图会存出"没框"的图。
                    # 这里再提交几次直到拿到结果（只影响一次性抓图，实时预览不受影响）。
                    if args.snapshot and tracker_parts.runner.remote is not None:
                        deadline = time.time() + max(args.server_timeout, 1.0)
                        while time.time() < deadline and not tracker_parts.tracks:
                            time.sleep(0.03)
                            tracker_parts.detect(detect_input, detect_scale, offset,
                                                 view_size=(view.shape[1], view.shape[0]))
                detections = tracker_parts.tick()      # 每帧都跑：预测补框，显示才连贯

            # ---------------- 阶段：每 --stage-interval 帧认一次「装配区」 ----------------
            # 只裁装配区那一块图，跟零件框无关；旋转遍历一整圈（默认 1°），取最像的那张参考图。
            # 间隔之间沿用上次结果，画面上标签就不会每帧跳。
            if detect and stage is not None:
                if frame_count == 1 or frame_count % max(args.stage_interval, 1) == 0:
                    region = (tuple(args.stage_region) if args.stage_region
                              else tracker_zones.region("装配区"))
                    if region is not None:
                        started = time.time()
                        stage_result = stage.match(view, region=region)
                        stage_cost = (time.time() - started) * 1000.0

            if detect:
                # 先占住状态栏区域拿到高度，区域标签与右上角提示据此避开顶部
                status_height = draw_overlay(view, "")
                for index, zone in enumerate(tracker_zones.drawable()):
                    x1, y1, x2, y2 = zone["box"]
                    color = ZONE_COLORS[index % len(ZONE_COLORS)]
                    draw_zone(view, x1, y1, x2, y2, zone["name"], color, font=label_font,
                              top_margin=status_height, thickness=args.line_width)

                # 零件：**一个零件只画一个框，线细**；检出的和卡尔曼补的画法完全一样，
                # 不再区分"预测/实际"，也不写"预测"标签。
                if not args.no_trail:
                    for det in detections:
                        if det["trail"]:
                            draw_trail(view, det["trail"], TRAIL_COLOR, thickness=1)
                for det in detections:
                    draw_detection(view, det["box"],
                                   f"{det['name']} {det['conf']:.2f}", PART_COLOR,
                                   font=label_font, thickness=args.line_width,
                                   top_margin=status_height)

                # 阶段标签贴「装配区」框的右上角（左上角是区域名），和零件框互不遮挡
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

                # 手部框：帧格式里预留的槽位。电脑端将来接了手部识别，这里自动就画出来；
                # 现在服务端一般发 valid=0，也就是不画。
                if tracker_parts is not None and tracker_parts.runner.remote is not None:
                    hand = tracker_parts.runner.remote.hand
                    if hand:
                        draw_detection(view, hand["xyxy"], f"hand {hand['conf']:.2f}",
                                       HAND_COLOR, font=label_font,
                                       thickness=args.line_width, top_margin=status_height)
            now = time.time()
            if now - last_time >= 0.5:                      # 滑动窗口瞬时帧率（不含启动预热）
                fps_now = (frame_count - last_count) / max(now - last_time, 1e-6)
                last_time, last_count = now, frame_count
            gray = None
            if frame_count % 10 == 0 or sharpness is None:   # 清晰度每 10 帧算一次
                gray = cv2.cvtColor(view, cv2.COLOR_BGR2GRAY)
                sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
                brightness = float(gray.mean())
            status = (f"RIGHT EYE {view.shape[1]}x{view.shape[0]} {info['fourcc']} | "
                      f"fps {fps_now:.1f} | mean {brightness:.0f} | sharp {sharpness:.1f}")
            if detect:
                # 状态栏是 cv2.putText 画的，画不了中文（会变成 ??????），所以用英文。
                # L/R 后面两个数是"暗色可见比 / 亮占比增量"；pred=预测框数量，
                # yolo=上次推理耗时（Pi 上调 --yolo-interval / --yolo-imgsz 就看这个数）
                status += (f" | trays {len(tracker_zones.drawable())}"
                           f" | parts {len(detections)}")
                if tracker_parts is not None and tracker_parts.enabled:
                    remote = tracker_parts.runner.remote
                    if remote is not None:
                        # 本机耗时不代表什么，真正要看的是"整轮往返"和"掉帧数"
                        status += (f" | net {remote.round_trip_ms:.0f}ms"
                                   f" drop {remote.dropped}")
                        if remote.wire_format == "binary":
                            # 下行帧里电脑端报的：装配步数 + 3 个布尔（b101 = 1,0,1）
                            status += (f" | srv step{remote.step}"
                                       f" b{''.join(str(v) for v in remote.flags)}"
                                       f" {remote.last_cost_ms:.0f}ms")
                    else:
                        status += f" | yolo {tracker_parts.last_cost_ms:.0f}ms"
                if tracker_zones.locked:
                    left, right = tracker_zones.zones
                    status += (f" | L{left['visible_ratio']:.2f}/{left['cover_delta']:.2f}"
                               f" R{right['visible_ratio']:.2f}/{right['cover_delta']:.2f}")
                if stage is not None:
                    # 状态栏是 cv2.putText 画的，只能用英文；度数写 deg（° 画不出来）
                    if stage_result is None:
                        status += " | stage -"
                    else:
                        status += (f" | stage {stage_result['label']}"
                                   f" {stage_result['score']:.2f}"
                                   f" {stage_result['angle']:+.1f}deg"
                                   f" {stage_cost:.0f}ms")
            # 状态栏就地画在 view 上并盖掉刚才的占位黑条；
            # draw_overlay 返回的是高度（int），图像仍是 view 本身
            draw_overlay(view, status)
            annotate = view

            if detect:
                # 工位状态提示固定在整个屏幕的右上角（状态栏正下方）
                draw_machine_state(view, tracker_zones.machine_state(), font=label_font,
                                   top=status_height)

            # ---- 数据采集：数字键或按钮点击 ----
            # 默认存**干净帧**（没有区域框/零件框/状态栏/按钮），这样误识别的框不会被
            # 烧进训练图；要看带识别的版本就加 --capture-overlay。
            if capture_enabled:
                hit_index = deck.hotkey_index(keys_pressed)   # 数字键 1~5
                if hit_index is None and deck.pending is not None:
                    hit_index, deck.pending = deck.pending, None   # 鼠标点击
                if hit_index is not None:
                    shot = view if args.capture_overlay else clean_view
                    saved = deck.capture(shot, hit_index)
                    if saved is not None:
                        print(f"已采集：{saved}")
                deck.layout(view)                     # 画面尺寸变了会重排按钮
                deck.draw(view)

            if args.save:                                     # 存带识别结果的这一帧
                if is_image:
                    out = save_dir / f"识别结果_{source_path.stem}.png"
                else:
                    out = save_dir / f"结果_{datetime.now():%Y%m%d_%H%M%S}.jpg"
                save_image(out, annotate)
                if is_image or frame_count % 100 == 0:         # 视频源每 100 帧存一张
                    print(f"已保存：{out}")

            if args.snapshot:
                gray = cv2.cvtColor(view, cv2.COLOR_BGR2GRAY)
                out = save_dir / "右目画面.jpg"
                save_image(out, annotate)
                print(f"已保存：{out}")
                print(f"右目画面：{view.shape[1]} × {view.shape[0]}，格式 {info['fourcc']}，"
                      f"亮度均值 {float(gray.mean()):.1f}，"
                      f"清晰度 {float(cv2.Laplacian(gray, cv2.CV_64F).var()):.1f}")
                if detect:
                    print(f"识别结果：料盘区域 {len(tracker_zones.drawable())} 个，"
                          f"零件 {len(detections)} 个")
                    if stage is not None:
                        if stage_result is None:
                            print("阶段识别：这一帧没认出来（装配区框未锁定或裁剪非法）")
                        else:
                            print(f"阶段识别：{stage_result['label']}"
                                  f"（相似度 {stage_result['score']:.3f}，"
                                  f"摆角 {stage_result['angle']:+.1f}°，"
                                  f"耗时 {stage_cost:.0f} ms）")
                    if tracker_parts is not None and not tracker_parts.enabled:
                        print(f"零件跟踪未启用：{tracker_parts.error}")
                break

            if show_window:
                cv2.imshow(WINDOW, annotate)
                key = cv2.waitKey(1) & 0xFF
                keys_pressed = key                        # 下一帧开头消费，用于采集/退出
                if key in (ord("q"), 27):
                    break
                if key == ord("s"):
                    out = save_dir / f"右目_{datetime.now():%Y%m%d_%H%M%S}.jpg"
                    save_image(out, annotate)              # 存带识别结果的那一帧
                    print(f"已保存：{out}")
            else:
                keys_pressed = 255                        # 无窗口：视为没有按键
    finally:
        if cap is not None:                               # 图片源没有 VideoCapture
            cap.release()
        if tracker_parts is not None:
            tracker_parts.runner.close()                  # http 后端的后台线程要收干净
        if show_window:
            cv2.destroyAllWindows()
        if capture_enabled:
            print(deck.summary())
