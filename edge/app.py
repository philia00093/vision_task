"""主循环：采集 → 料盘判定 → 零件推理 → 绘制 → 按键/存图。

上游模块都只是"工具"，这里才是把节奏串起来的编排层：两种节流（--zone-interval / --yolo-interval）+ 每帧卡尔曼补框。"""

from __future__ import annotations

import numpy as np
import cv2
import time
from collections import OrderedDict, deque
from datetime import datetime
from pathlib import Path

from edge.args import build_parser
from edge.camera import fourcc_str, open_camera, probe_modes
from edge.capture import CaptureDeck, on_mouse
from edge.constants import CAPTURE_SPECS, DEFAULT_TRAY_MIN_FILL, DEFAULT_YOLO_WEIGHTS, DEFAULT_ZONES_FILE, GREEN_MIN_FILL_RATIO, HAND_COLOR, IMAGE_SUFFIXES, PART_COLOR, TRAIL_COLOR, WINDOW, ZONE_COLORS
from edge.defects import AreaChecker, find_part
from edge.drawing import (_measure_label, draw_area_verdict, draw_detection, draw_machine_state,
                          draw_overlay, draw_stage, draw_trail, draw_zone,
                          draw_text_box, load_cjk_font)
from edge.framebuffer import FbWriter
from edge.grabber import FrameGrabber
from edge.imaging import contours_to_boxes, prepare_detect_input, save_image, take_right
from edge.parts import PartTracker
from edge.stage_match import CascadeStageMatcher, SiftStageMatcher, StageMatcher
from edge.stage_parts import UNKNOWN, PartCounter, stage_from_parts
from edge.stage_worker import StageWorker
from edge.project import ROOT, ensure_dir
from edge.zone_tracker import ZoneTracker, load_zones, parse_zones
from edge.zones import zone_mask


def commit_stage(history, window, now):
    """把最近 `window` 秒的阶段判定做多数表决，返回要显示的结论（可能 None）。

    "连续 1 秒识别"就是这里：单帧认出来不算数，窗口里同样的标签要占够数才采纳；
    票数不够就沿用上一次的结论（不改 stage_result），标签不会一闪一闪。
    """
    horizon = now - max(window, 0.1)
    while history and history[0][0] < horizon:
        history.popleft()
    if not history:
        return None
    votes = {}
    for _ts, item in history:
        votes.setdefault(item.get("label") or "", []).append(item)
    label, items = max(votes.items(), key=lambda kv: (len(kv[1]), kv[1][-1].get("score", 0)))
    need = max(2, int(len(history) * 0.6 + 0.999))
    if len(items) < need and len(history) > 1:
        return None
    winner = dict(items[-1])
    winner["score"] = sum(i.get("score", 0.0) for i in items) / len(items)
    winner["votes"] = f"{len(items)}/{len(history)}"
    return winner


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
    # 阶段标签单独大一号：屏上最要紧的就是"装到第几步"，小了根本看不清
    stage_font = load_cjk_font(args.stage_label_size) if detect else None
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
                                    wire=args.wire, window=args.http_window,
                                    result_hold=args.result_hold,
                                    track=not args.no_track)
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
        stage_worker = None          # 后台识别线程（stage 没启用就是 None）
        part_counter = None          # 数零件定阶段（4 类零件少几个 = 第几步）
        area_check = None            # 零件面积检查（缺陷识别 1）
        if args.no_stage:
            print("阶段识别：已用 --no-stage 关闭")
        else:
            stage_dir = Path(args.stage_dir).expanduser()
            if not stage_dir.is_absolute():
                stage_dir = ROOT / stage_dir
            if args.stage_method == "cascade":
                stage = CascadeStageMatcher.from_directory(
                    stage_dir, ref_size=max(args.stage_size * 4, 128),
                    min_score=args.stage_min_score,
                    kp_gate=args.stage_kp_gate,
                    min_inliers=args.stage_min_inliers,
                    edges=args.stage_edges,
                    nn_max_dist=args.stage_nn_max_dist,
                    nn_kwargs={"margin": args.stage_nn_margin})
            elif args.stage_method == "sift":
                stage = SiftStageMatcher.from_directory(
                    stage_dir, ref_size=max(args.stage_size * 4, 128),
                    min_score=args.stage_min_score, edges=args.stage_edges)
            else:
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
                print(f"         裁剪窗口：{window} | 每 {args.stage_period:.1f}s 认一次"
                      f" | 判定窗口 {args.stage_window:.1f}s（多数一致才采纳）"
                      f" | 相似度 <{args.stage_min_score:.2f} 记为 unknown")
                if args.stage_region is None:
                    print("         区域锁定前不输出阶段（没有装配区框就不知道裁哪）"
                          "；也可用 --stage-region 固定窗口")
                # 识别放后台线程：主循环到点丢一帧就走，不等那 130ms
                stage_worker = StageWorker(stage)
                # 数零件定阶段：4 类零件里物料区少了几个就是第几步（比图像比对稳）
                if tracker_parts is not None:
                    part_counter = PartCounter(tracker_parts.names.values(),
                                               memory=args.part_memory)
                    if not args.no_part_check:
                        # 缺陷识别 1：compass 的面积（偏导求轮廓）跟基准比，超 4% 判 NG
                        area_check = AreaChecker("compass", tol=args.part_tol,
                                                 window=args.part_window,
                                                 calib=args.part_calib,
                                                 ref=args.part_area_ref)
                print(f"         匹配在后台线程跑（主循环不被卡）；"
                      f"识别异常会打在状态栏 stage error 上")

    save_dir = Path(args.save_dir).expanduser()
    if not save_dir.is_absolute():
        save_dir = ROOT / save_dir
    save_dir = ensure_dir(save_dir.resolve(), "输出目录")
    print(f"项目根目录：{ROOT}")
    print(f"输出目录  ：{save_dir}")
    show_window = ((not args.snapshot) and (not args.no_window)
                   and not is_image and hasattr(cv2, "imshow") and not args.fb)
    if show_window:
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    elif args.fb:
        print(f"framebuffer 输出：{args.fb}")
    else:
        print("无 GUI 或使用 --snapshot/图片源：只保存图像，不显示窗口")

    # 外接屏（framebuffer）：树莓派 Lite 没装 X，cv2.imshow 不可用，直接把画面写 /dev/fb0
    fb_screen = FbWriter(args.fb, mode=args.fb_mode, rotate=args.fb_rotate) if args.fb else None
    if fb_screen is not None:
        print(f"          屏幕 {fb_screen.describe()}；Ctrl+C 退出，按键/采集按钮不可用")

    # 相机取帧放进独立线程：解码 20.8ms 不再阻塞主循环（只对实时相机，视频文件要顺序读）
    grabber = FrameGrabber(cap, timeout=args.server_timeout) if (cap is not None and not is_video) else None
    last_serial = -1
    frame_stamp = 0.0
    # ---- 画面跟着结果走 ----
    # sync_ring：seq -> (那一帧, 那一帧时刻的框, 采集时刻)。结果回来时按 ref_seq 取对应帧，
    # 这样"屏幕上看到的画面"和"画在上面的框"永远是同一时刻的，代价是整屏慢约一个网络往返。
    sync_display = not args.no_sync_display
    # 环形缓冲：seq -> (那一帧, 那一帧时刻的框, 采集时刻)。按"时间"淘汰，
    # 不是按帧数——网络往返 500ms 时，8 帧（0.27s）根本盖不住，结果永远找不到对应帧，
    # 只能退回实时帧，看起来就"框和画面对不上 + 跳"。
    sync_ring: "OrderedDict[int, tuple]" = OrderedDict()
    assembled_step = 0               # 已经确认完成的装配步数（手离开物料区时锁定）
    material_occluded = False        # 上一帧物料区是否被遮
    sync_hits = sync_misses = 0
    display_stamp = 0.0
    display_seq = -1                 # 已经显示到哪一帧：只许往前，不许倒退
    play_seq = 0                     # 入队用的本地帧号（不依赖网络是否发出去）
    last_display = None              # 上一帧显示内容，乱序时复用它而不是跳回实时帧
    # ---- 接收缓冲的"一次性预充" ----
    # 启动阶段先攒够 --preroll 帧（约 0.1~0.3 秒）再开始出图，避免开局抖动/空跳；
    # 攒够之后这个判断就永久关掉，不再影响后续显示。
    preroll_done = args.preroll <= 0
    preroll_deadline = time.time() + max(args.preroll_timeout, 0.1)

    frame_count, last_time, last_count, fps_now = 0, time.time(), 0, 0.0
    last_report = 0.0
    tick_period = (1.0 / args.loop_fps) if args.loop_fps and args.loop_fps > 0 else 0.0
    next_tick = time.time()                                 # 固定节拍的下一个时间点
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
    last_infer_ts = 0.0              # 上一次提交推理的时间（--infer-fps 用）
    last_source = None               # 最近一次提交给模型的裁剪图（面积检查在它上面量）
    last_scale, last_offset = 1.0, (0.0, 0.0)
    stage_history = deque()          # 最近 --stage-window 秒的判定：(ts, 结果)
    stage_last_ts = 0.0              # 上一次跑阶段识别的时间（0 = 立刻跑第一次）

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
            elif grabber is not None:
                # 相机：从后台线程拿"最新一帧"（解码与主循环并行）
                frame, last_serial, frame_stamp, fresh = grabber.read(last_serial)
                if frame is None:
                    print(f"读取画面失败（{grabber.error or '相机没有输出'}）")
                    break
                frame_count += 1
                if frame_count <= args.warmup or float(frame[::8, ::8].mean()) < 12:
                    continue
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
                # 注意：这里用**原始**暗色掩膜。料盘顶部的反光会把掩膜挖掉一块，
                # 但那是"开口"不是"洞"，闭运算补不回来；框的尺寸由后面的
                # normalize_tray_boxes（两个同型号方盘的先验）修正。
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
                # 装配区框内的零件不跟踪（放进去就装好了，静态的）
                if tracker_parts is not None:
                    r = tracker_zones.region("装配区")
                    tracker_parts.exclude_rect = ((r[0], r[1], r[0] + r[2], r[1] + r[3])
                                                  if r else None)

            # ---------------- 零件：每 --yolo-interval 帧跑一次 YOLO + 每帧卡尔曼预测 ----
            submitted_now = False
            if detect and tracker_parts is not None:
                # 两种节流：按帧数（--yolo-interval）或按时间（--infer-fps）。
                # 链路差的时候按帧数塞会把 TCP 窗口灌满，帧就变成"一阵一阵到"，
                # 识别看起来就是断断续续；按秒限速能让到达变平整。
                if args.infer_fps > 0:
                    due = (time.time() - last_infer_ts) >= (1.0 / args.infer_fps)
                else:
                    due = frame_count == 1 or frame_count % max(args.yolo_interval, 1) == 0
                if due:
                    last_infer_ts = time.time()
                    # 只上传"工作区"（--upload-crop）：面积小 → JPEG 小、编码快，
                    # 而且模型在同样 imgsz 下看到的零件更大更准。
                    crop = args.upload_crop
                    if crop:
                        cx, cy, cw, ch = crop
                        # 注意：行是 y 方向（高 ch），列是 x 方向（宽 cw）；原来写反了
                        x0, y0 = max(0, cx), max(0, cy)
                        source = view[y0:y0 + ch, x0:x0 + cw]
                        if source.size == 0:            # 裁到画面外了，退回整幅
                            source, offset = view, (0.0, 0.0)
                        else:
                            offset = (float(cx), float(cy))
                    else:
                        source, offset = view, (0.0, 0.0)
                    detect_input, detect_scale = prepare_detect_input(source, args.detect_width)
                    last_source, last_scale, last_offset = source, detect_scale, offset
                    tracker_parts.detect(detect_input, detect_scale, offset,
                                         view_size=(view.shape[1], view.shape[0]))
                    submitted_now = True
                    # --snapshot + http 后端：结果要一个来回才到，单帧抓图会存出"没框"的图。
                    # 这里再提交几次直到拿到结果（只影响一次性抓图，实时预览不受影响）。
                    if args.snapshot and tracker_parts.runner.remote is not None:
                        deadline = time.time() + max(args.server_timeout, 1.0)
                        while time.time() < deadline and not tracker_parts.tracks:
                            time.sleep(0.03)
                            tracker_parts.detect(detect_input, detect_scale, offset,
                                                 view_size=(view.shape[1], view.shape[0]))
                detections = tracker_parts.tick()      # 每帧都跑：预测补框，显示才连贯
                # ---- 缺陷识别 1：compass 面积（偏导 → 轮廓 → 面积 → 和基准比）----
                if area_check is not None and submitted_now and last_source is not None:
                    target = next((d for d in detections if d["name"] == "compass"), None)
                    if target is not None:
                        # 框是 view 坐标，先换回"模型看到的那张裁剪图"的坐标再量：
                        # 同一分辨率下量出来的面积才稳（全分辨率 view 上互换两个峰，CV 2%）
                        ox, oy = last_offset
                        bx1, by1, bx2, by2 = target["box"]
                        box_crop = ((bx1 - ox) / last_scale, (by1 - oy) / last_scale,
                                    (bx2 - ox) / last_scale, (by2 - oy) / last_scale)
                        # 检测框只当"搜索窗"，边框和面积都由偏导找出来的零件轮廓决定
                        # （实测框抖 ±5px，轮廓面积只变 0.46%；直接拿框裁会变 7%）
                        part = find_part(last_source, box_crop, search=args.part_search)
                        if part is not None:
                            was_ok = area_check.ok
                            area_check.push(part["area"])
                            if (area_check.calibrated and area_check.ok != was_ok
                                    and area_check.judged > 0):
                                tag = "合格" if area_check.ok else "不合格"
                                print(f"[缺陷] compass 轮廓面积 {area_check.area:.0f}"
                                      f"（基准 {area_check.ref:.0f}，"
                                      f"误差 {area_check.ratio*100:+.1f}%）→ {tag}", flush=True)
                # 每一帧都入队（包括因为窗口满被背压丢掉、没发给电脑的那些）。
                # 之前只存"发出去的帧"，链路一旦掉速，队列就被抽干 → 水位不足 → 画面卡住。
                if sync_display:
                    play_seq += 1
                    if True:
                        now_t = frame_stamp or time.time()
                        sync_ring[play_seq] = (view.copy(), detections, now_t)
                        # 按时间淘汰（留够覆盖 RTT 的历史），再按帧数兜底防爆内存
                        limit = now_t - max(args.sync_seconds, 0.2)
                        while sync_ring and next(iter(sync_ring.values()))[2] < limit:
                            sync_ring.popitem(last=False)
                        while len(sync_ring) > max(args.sync_max_frames, 4):
                            sync_ring.popitem(last=False)

            # ---------------- 阶段：每 --stage-period 秒认一次「装配区」 ----------------
            # 只裁装配区那一块图，跟零件框无关；用时间而不是帧数当间隔，
            # 帧率高低都不会让识别变得忽快忽慢。
            # 「连续 1 秒识别」：窗口内的判定要**多数一致**才采纳，单帧抖一下不会翻牌。
            if detect and stage is not None:
                now_stage = time.time()
                # 后台线程算完的结果先收进来（不阻塞主循环）
                for _stamp, _fresh in stage_worker.drain():
                    stage_history.append((_stamp, _fresh))
                    stage_cost = _fresh.get("cost_ms", stage_cost)
                if now_stage - stage_last_ts >= max(args.stage_period, 0.05):
                    stage_last_ts = now_stage
                    region = (tuple(args.stage_region) if args.stage_region
                              else tracker_zones.region("装配区"))
                    # 先用零件类别走规则（2/3 步能直接定，还省掉 130ms 图像匹配）；
                    # 规则说"0/1 分不出"（返回 None）时才落到图像分支。
                    rule_stage, rule_reason, rule_score = (UNKNOWN, "", 0.0)
                    material_rect = tracker_zones.region("物料区")
                    if part_counter is not None and region is not None:
                        rule_stage, rule_reason, rule_score = part_counter.update(
                            detections, tracker_parts.names, region, material_rect)
                    elif tracker_parts is not None and region is not None:
                        rule_stage, rule_reason, rule_score = stage_from_parts(
                            detections, tracker_parts.names, region, material_rect)
                    if rule_stage is not None:
                        stage_history.append((now_stage, {
                            "label": rule_stage, "raw_label": rule_stage,
                            "score": rule_score, "angle": 0.0, "method": "parts",
                            "kp": 0, "note": rule_reason}))
                        stage_cost = 0.0
                    elif region is not None:
                        # 参考图是 /save 存下来的"上传裁剪图"（900x500），
                        # 所以现场也要用同一块做匹配，否则坐标系对不上。
                        crop = args.upload_crop
                        if crop:
                            cx, cy, cw, ch = crop
                            src = view[cy:cy + ch, cx:cx + cw]
                            region_c = (region[0] - cx, region[1] - cy, region[2], region[3])
                        else:
                            src, region_c = view, region
                        # 只丢帧、不等结果：识别在后台线程里跑（cv2 计算时会放开 GIL）
                        stage_worker.submit(src.copy(), region_c, now_stage)
                # 窗口内多数表决：票数不够就沿用上一次结论（不翻牌）
                committed = (commit_stage(stage_history, args.stage_window, now_stage)
                             if stage_history else None)
                if committed is not None:
                    if committed.get("label") != (stage_result or {}).get("label"):
                        method = committed.get("method")
                        if method == "nn":
                            extra = f"距离 {committed.get('dist', 0.0):.3f}"
                        elif method == "parts":
                            extra = committed.get("note", "")
                        else:
                            extra = f"内点 {committed.get('inliers', 0)}"
                        print(f"[阶段] {(stage_result or {}).get('label', '-')} → "
                              f"{committed.get('label')}"
                              f"（{committed.get('method')} 相似度"
                              f" {committed.get('score', 0.0):.3f}，{extra}，"
                              f"票 {committed.get('votes', '?')}，"
                              f"关键点 {committed.get('kp', 0)}，"
                              f"{stage_cost:.0f}ms）", flush=True)
                    stage_result = committed

            # ---- 手遮住物料区 = 这一次装配完成，锁定当前阶段 ----
            if detect and tracker_parts is not None:
                material = next((z for z in tracker_zones.zones if z["name"] == "物料区"), None)
                covered = bool(material and material.get("box") is not None
                               and material.get("occluded"))
                if covered and not material_occluded and stage_result is not None:
                    label = stage_result.get("label") or ""
                    if label.startswith("step") and label[4:].isdigit():
                        number = int(label[4:])
                        if number > assembled_step:
                            assembled_step = number
                            print(f"[装配] 手遮物料区 → 第 {number} 步完成", flush=True)
                material_occluded = covered

            # ---- 选要显示的帧：有同步结果就用"结果对应的那一帧"，否则用最新帧 ----
            display_stamp = frame_stamp
            # ---- 播放缓冲：攒够 dequeue_at 帧才出队，之后每帧弹最老的一帧（FIFO） ----
            # 这样屏上是"按采集顺序、平滑播放"的画面，代价是固定滞后约 dequeue_at 帧。
            # 队列低于水位就保持上一帧，等重新攒够（不跳帧、不乱序）。
            if detect and tracker_parts is not None and sync_display and sync_ring:
                if not preroll_done:
                    if len(sync_ring) >= args.dequeue_at or time.time() >= preroll_deadline:
                        preroll_done = True
                        print(f"[sync] 缓冲就绪：{len(sync_ring)} 帧，开始出队", flush=True)
                if len(sync_ring) >= args.dequeue_at:
                    seq_key, entry = sync_ring.popitem(last=False)   # 最老的一帧
                    view, detections, display_stamp = entry
                    display_seq = seq_key
                    last_display = entry
                    sync_hits += 1
                elif last_display is not None:
                    view, detections, display_stamp = last_display     # 水位不足：保持
                    sync_misses += 1

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
                # 面积判定写在右上角（不画框，不挡零件）
                if area_check is not None and area_check.judged > 0:
                    state_h = _measure_label(tracker_zones.machine_state(), label_font)[1]
                    draw_area_verdict(view, area_check.label(), ok=area_check.ok,
                                      font=label_font, top=status_height, below=state_h + 12)

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
                                          f" {stage_result['score']:.2f}")
                            # 近邻分支没有"摆角"这个概念，改报关键点数
                            method = stage_result.get("method")
                            if method == "nn":
                                stage_text += f" kp{stage_result.get('kp', 0)}"
                            elif method == "parts":
                                stage_text += ""          # 数零件的结论，没有"摆角"这回事
                            else:
                                stage_text += f" {stage_result['angle']:+.1f}°"
                        draw_stage(view, (rx, ry, rx + rw, ry + rh), stage_text,
                                   font=stage_font or label_font, top_margin=status_height)

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
                if now - last_report >= 5.0:                 # 远程调试要看主循环真实帧率
                    last_report = now
                    lag_ms = (now - display_stamp) * 1000.0 if display_stamp else 0.0
                    net = ""
                    if tracker_parts is not None and tracker_parts.runner.remote is not None:
                        remote_now = tracker_parts.runner.remote
                        net = (f" | 网络往返 {remote_now.round_trip_ms:.0f}ms"
                               f" 在飞 {remote_now.inflight}"
                               f" 迟到 {remote_now.stale_results}")
                    hit = (sync_hits / max(sync_hits + sync_misses, 1) * 100.0
                           if (sync_hits + sync_misses) else 0.0)
                    print(f"[fps] 主循环 {fps_now:.1f} fps | 画面滞后 {lag_ms:.0f}ms"
                          f"（含网络往返）{net} | 同步命中 {hit:.0f}%"
                          f" | 队列 {len(sync_ring)}帧"
                          f" | 已处理 {frame_count} 帧", flush=True)
            gray = None
            # 清晰度统计：1280x960 的 Laplacian 要 ~50ms，是主循环最尖的尖峰。
            # 改成每 60 帧、在半分辨率上算（~12ms），数值仍能反映"糊没糊"。
            if frame_count % 60 == 0 or sharpness is None:
                small = view[::2, ::2]
                gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
                sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
                brightness = float(small.mean())
            status = (f"RIGHT EYE {view.shape[1]}x{view.shape[0]} {info['fourcc']} | "
                      f"fps {fps_now:.1f} | mean {brightness:.0f} | sharp {sharpness:.1f}")
            if detect:
                # 状态栏是 cv2.putText 画的，画不了中文（会变成 ??????），所以用英文。
                # L/R 后面两个数是"暗色可见比 / 亮占比增量"；pred=预测框数量，
                # yolo=上次推理耗时（Pi 上调 --yolo-interval / --yolo-imgsz 就看这个数）
                # 阶段放在最前面：状态栏一行放不下时，后面被裁掉的只会是
                # L/R 这些调试量，阶段这个"当前装到第几步"必须始终看得见。
                if area_check is not None and area_check.calibrated:
                    status += (f" | cmp {area_check.area:.0f}"
                               f" {area_check.ratio*100:+.1f}%"
                               f" {'OK' if area_check.ok else 'NG'}")
                if stage is not None:
                    if stage_result is None:
                        status += " | st -"
                    else:
                        status += (f" | st {stage_result['label'].replace('step', '')}"
                                   f" {stage_result['score']:.2f}"
                                   f" {stage_result.get('method', '')[:1]}"
                                   f"kp{stage_result.get('kp', 0)}"
                                   f" {stage_cost:.0f}ms")
                status += (f" | trays {len(tracker_zones.drawable())}"
                           f" | parts {len(detections)}")
                if tracker_parts is not None and tracker_parts.enabled:
                    remote = tracker_parts.runner.remote
                    if remote is not None:
                        # 本机耗时不代表什么，真正要看的是"整轮往返"和"掉帧数"
                        status += (f" | net {remote.round_trip_ms:.0f}ms"
                                   f" win{remote.window} fly{remote.inflight}"
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
            # 状态栏就地画在 view 上并盖掉刚才的占位黑条；
            # draw_overlay 返回的是高度（int），图像仍是 view 本身
            draw_overlay(view, status)
            annotate = view

            if detect:
                # 工位状态提示固定在整个屏幕的右上角（状态栏正下方）
                draw_machine_state(view, tracker_zones.machine_state(), font=label_font,
                                   top=status_height)
                if assembled_step > 0:
                    # 装配进度固定在整个画面右上角、状态提示下面一行
                    step_text = f"装配到第 {assembled_step} 步"
                    patch_h = _measure_label(step_text, label_font)[1]
                    draw_text_box(view, step_text, view.shape[1] - 10,
                                  status_height + patch_h * 2 + 18,
                                  (0, 220, 0), font=label_font, right=True)

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
                            pretty = (f"近邻 距离 {stage_result.get('dist', 0.0):.3f}"
                                      if stage_result.get("method") == "nn"
                                      else f"内点 {stage_result.get('inliers', 0)}")
                            print(f"阶段识别：{stage_result['label']}"
                                  f"（{stage_result.get('method', '?')} 相似度"
                                  f" {stage_result['score']:.3f}，{pretty}，"
                                  f"关键点 {stage_result.get('kp', 0)}，"
                                  f"耗时 {stage_cost:.0f} ms）")
                    if tracker_parts is not None and not tracker_parts.enabled:
                        print(f"零件跟踪未启用：{tracker_parts.error}")
                break

            # ---- 固定节拍：采集/入队/出队同一个频率，帧率不再忽快忽慢 ----
            if tick_period > 0:
                next_tick += tick_period
                sleep_for = next_tick - time.time()
                if sleep_for > 0:
                    time.sleep(sleep_for)
                else:
                    next_tick = time.time()               # 落后太多就重新对齐，不追帧

            if fb_screen is not None:
                fb_screen.show(annotate)                  # 外接屏：整帧直接写 framebuffer
                keys_pressed = 255
            elif show_window:
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
        if grabber is not None:
            grabber.close()
        if cap is not None:                               # 图片源没有 VideoCapture
            cap.release()
        if tracker_parts is not None:
            tracker_parts.runner.close()                  # http 后端的后台线程要收干净
        if show_window:
            cv2.destroyAllWindows()
        if fb_screen is not None:
            fb_screen.close()                             # framebuffer 用完要还回去
        if capture_enabled:
            print(deck.summary())
