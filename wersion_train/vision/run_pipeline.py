"""视觉流水线入口：绿色料盘区/装配区识别（OpenCV）+ 手部实时检测跟踪。

用法（务必用 yolo26 环境，推荐直接用 run_pipeline.bat）：
    run_pipeline.bat                                              # 默认相机 0，实时预览
    run_pipeline.bat --source "Left Eye View_screenshot_13.10.2026.png" --save
    run_pipeline.bat --source test.mp4 --loop --save --log-events
    run_pipeline.bat --debug-green                                 # 存绿色掩码，调阈值用
    run_pipeline.bat --green-lo 30 60 80 --min-area 0.01           # 光照不同时调阈值
    run_pipeline.bat --no-green --region wersion_train/weights/region_best.pt    # 改用训练出的区域模型

区域识别默认走绿色分割（纯 OpenCV，零训练）；识别不到两块料盘时自动几何兜底，
并在画面顶部用红字提示，不会静默出错。

热键：Q/ESC 退出，S 保存当前帧，B 保存区域基线 JSON，空格暂停。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]   # = wersion_train/
sys.path.insert(0, str(ROOT))

# 控制台按 UTF-8 输出，避免中文提示在 cmd/批处理里变乱码
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass

from vision.common import OUTPUT, WEIGHTS, ensure_dirs, save_image  # noqa: E402
from vision.modules.green_regions import GreenConfig  # noqa: E402
from vision.modules.pipeline import VisionPipeline  # noqa: E402
from vision.stream import FpsMeter, StreamReader  # noqa: E402

WINDOW = "Version Task Pipeline"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="绿色料盘区/装配区识别 + 手部实时检测跟踪",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    source = parser.add_argument_group("视频源")
    source.add_argument("--source", default="0",
                        help="相机序号(0,1...)、视频文件或图片路径")
    source.add_argument("--backend", choices=("auto", "dshow", "msmf"), default="dshow")
    source.add_argument("--width", type=int, default=1280, help="相机请求宽度（0=不设）")
    source.add_argument("--height", type=int, default=960, help="相机请求高度（0=不设）")
    source.add_argument("--fps", type=int, default=30, help="相机请求帧率（0=不设）")
    source.add_argument("--fourcc", default="MJPG", help="相机像素格式（YUY2 会极卡）")
    source.add_argument("--layout", choices=("auto", "side_by_side", "top_bottom", "none"),
                        default="none", help="双目拼接帧的拆分方式；单目画面用 none")
    source.add_argument("--loop", action="store_true", help="视频文件循环播放")
    source.add_argument("--warmup", type=int, default=0, help="相机启动丢弃帧数")

    region = parser.add_argument_group("区域识别（绿色分割）")
    region.add_argument("--no-green", action="store_true",
                        help="关掉绿色分割，改用区域模型或几何兜底")
    region.add_argument("--green-lo", nargs=3, type=int, metavar=("H", "S", "V"),
                        default=[35, 80, 100], help="绿色 HSV 下限，光照不对先调这个")
    region.add_argument("--green-hi", nargs=3, type=int, metavar=("H", "S", "V"),
                        default=[95, 255, 255], help="绿色 HSV 上限")
    region.add_argument("--green-morph", type=int, default=7,
                        help="形态学核大小（0=不做开闭运算）")
    region.add_argument("--min-area", type=float, default=0.004,
                        help="绿色色块最小面积占整幅画面的比例")
    region.add_argument("--merge-gap", type=int, default=0,
                        help="水平间距小于该值的绿色块合并（料盘被反光切开时用）")
    region.add_argument("--debug-green", action="store_true",
                        help="保存绿色掩码与分割视图到 output/green/，用于调阈值")
    region.add_argument("--split-ratio", type=float, default=0.62,
                        help="几何兜底时料盘区占画面宽度的比例")

    model = parser.add_argument_group("模型")
    model.add_argument("--region", default=None,
                       help="区域检测权重（料盘区/装配区/hand）；给了就用模型，不用绿色分割")
    model.add_argument("--hand", default=None,
                       help="手部权重（默认 wersion_train/weights/yolo26n-pose.pt）")
    model.add_argument("--step", default=None, help="装配步骤检测权重（可选）")
    model.add_argument("--tracker",
                       default=str(ROOT / "configs" / "trackers" / "bytetrack.yaml"),
                       help="跟踪器配置；要更稳的 ID 可换 botsort.yaml")
    model.add_argument("--conf", type=float, default=0.35, help="区域/步骤置信度阈值")
    model.add_argument("--hand-conf", type=float, default=0.35, help="手部置信度阈值")
    model.add_argument("--imgsz", type=int, default=640, help="推理尺寸")
    model.add_argument("--device", default=None, help="cuda:0 / cpu；默认自动")
    model.add_argument("--baseline", default=None,
                       help="用之前保存的区域基线 JSON，跳过区域识别")
    model.add_argument("--no-keypoints", action="store_true", help="不画关键点骨架")

    out = parser.add_argument_group("输出")
    out.add_argument("--save", action="store_true", help="保存标注后的画面")
    out.add_argument("--save-dir", default=None, help="输出目录，默认 output/")
    out.add_argument("--save-baseline", action="store_true", help="保存区域基线 JSON 后继续")
    out.add_argument("--log-events", action="store_true",
                     help="把手跨越区域的事件记到 wersion_train/events/*.csv")
    out.add_argument("--no-show", action="store_true", help="不开窗口（无 GUI 环境）")
    out.add_argument("--max-frames", type=int, default=0, help="最多处理 N 帧后退出（0=不限）")
    return parser


def resolve_output_dir(args) -> Path:
    directory = Path(args.save_dir) if args.save_dir else OUTPUT
    if not directory.is_absolute():
        directory = ROOT / directory
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def make_writer(path: Path, size, fps: float):
    """视频写入器；打不开时返回 None，由调用方降级为逐帧存图。"""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, max(fps, 1.0), size)
    if not writer.isOpened():
        writer.release()
        return None
    return writer


def main() -> int:
    args = build_parser().parse_args()
    ensure_dirs()

    if args.hand is None and not (WEIGHTS / "yolo26n-pose.pt").is_file():
        print("[警告] 没找到 wersion_train/weights/yolo26n-pose.pt；"
              "先跑 wersion_train/tools/download_weights.bat --pose，或用 --hand 指定权重")

    try:
        reader = StreamReader(source=args.source, backend=args.backend, width=args.width,
                              height=args.height, fps=args.fps, fourcc=args.fourcc,
                              layout=args.layout, loop=args.loop, warmup=args.warmup)
    except RuntimeError as error:
        print(f"[错误] {error}")
        return 2
    print(f"[视频源] {reader.describe()}")

    out_dir = resolve_output_dir(args)
    green_config = GreenConfig(lo=tuple(args.green_lo), hi=tuple(args.green_hi),
                               morph=args.green_morph, min_area_ratio=args.min_area,
                               merge_gap=args.merge_gap)
    pipeline = VisionPipeline(region_weights=args.region, hand_weights=args.hand,
                              step_weights=args.step, tracker=args.tracker, conf=args.conf,
                              hand_conf=args.hand_conf, imgsz=args.imgsz, device=args.device,
                              split_ratio=args.split_ratio, baseline=args.baseline,
                              show_keypoints=not args.no_keypoints,
                              use_green=not args.no_green, green_config=green_config,
                              debug_dir=out_dir if args.debug_green else None,
                              log_events=args.log_events, verbose=True)

    detector = pipeline.detector
    if detector.use_green:
        print(f"[区域识别] 绿色分割（OpenCV）HSV {tuple(args.green_lo)}~{tuple(args.green_hi)}"
              f" | 最小面积 {args.min_area:.1%}"
              f" | 兜底中线 {args.split_ratio:.0%}")
    elif detector.using_model:
        print(f"[区域识别] 区域模型 {detector.names}")
    else:
        print(f"[区域识别] 几何兜底：左侧 {args.split_ratio:.0%} 为料盘区")
    # kpt_shape / nc 挂在 Pose26 检测头（model.model[-1]）上，不在 PoseModel 顶层
    hand_head = getattr(pipeline.tracker.model, "model", None)
    hand_head = getattr(hand_head, "model", None)
    kpt_shape = getattr(hand_head[-1], "kpt_shape", None) if hand_head is not None else None
    print(f"[手部跟踪] {pipeline.tracker.names} | 任务 "
          f"{getattr(pipeline.tracker.model, 'task', '?')} | 关键点 {kpt_shape}")

    show = not args.no_show and not reader.is_image and hasattr(cv2, "imshow")
    if show:
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)

    writer, writer_path, writer_tried = None, None, False
    fps_meter = FpsMeter()
    paused, frames, saved, baselines = False, 0, 0, 0
    last_canvas = last_frame = last_state = None
    try:
        while True:
            if not paused:
                ok, frame = reader.read()
                if not ok:
                    break
                frames += 1
                fps_now = fps_meter.tick()
                canvas, state, hands = pipeline.process(frame, frame_index=frames, fps=fps_now)
                last_canvas, last_frame, last_state = canvas, frame, state

                if args.save:
                    if reader.is_image:
                        target = out_dir / f"annotated_{Path(reader.source).stem}.png"
                        save_image(target, canvas)
                        saved += 1
                        print(f"[保存] {target}")
                    else:
                        if writer is None and not writer_tried:
                            writer_tried = True
                            writer_path = (out_dir /
                                           f"annotated_{time.strftime('%Y%m%d_%H%M%S')}.mp4")
                            writer = make_writer(writer_path,
                                                 (canvas.shape[1], canvas.shape[0]),
                                                 reader.fps_hint or 25.0)
                            if writer is None:
                                print("[警告] 视频写入器打不开，改为只存帧图")
                        if writer is not None:
                            writer.write(canvas)
                            saved += 1
                        elif frames % 30 == 0:
                            save_image(out_dir / f"frame_{frames:06d}.png", canvas)
                            saved += 1

                if args.save_baseline and not baselines:
                    path = pipeline.save_baseline(frame, state)
                    baselines += 1
                    print(f"[基线] 已保存区域基线：{path}")

                if args.max_frames and frames >= args.max_frames:
                    break

            if show:
                cv2.imshow(WINDOW, last_canvas if last_canvas is not None else frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord(" "):
                    paused = not paused
                if key == ord("s") and last_canvas is not None:
                    target = out_dir / f"snapshot_{time.strftime('%H%M%S')}.png"
                    save_image(target, last_canvas)
                    saved += 1
                    print(f"[保存] {target}")
                if key == ord("b") and last_frame is not None:
                    path = pipeline.save_baseline(last_frame, last_state, tag="manual")
                    baselines += 1
                    print(f"[基线] {path}")
            elif reader.is_image:
                break
    except KeyboardInterrupt:
        print("\n[中断] 用户停止")
    finally:
        reader.release()
        if writer is not None:
            writer.release()
        pipeline.close()
        if show:
            cv2.destroyAllWindows()

    print(f"\n处理帧数 {frames}，保存 {saved} 项"
          + (f"，视频 {writer_path}" if writer_path else ""))
    if pipeline.event_log is not None:
        print(f"事件日志：{pipeline.event_log}")
    if pipeline.tracker.last_error:
        print(f"[警告] 跟踪器最后一次报错：{pipeline.tracker.last_error}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
