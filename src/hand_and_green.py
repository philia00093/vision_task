"""手部识别 + 绿色区域框选（精简版，只做这两件事）。

画面里：
    · 绿色料盘 → 框出轮廓（青绿 = 料盘区，橙 = 装配区）
    · 手       → 框出 + 跟踪 ID

不做装配步骤识别、不判越区、不写日志、不存基线；需要那些功能用 run_pipeline.py。

用法（用 yolo26 环境，或直接用 run_hand_green.bat）：
    run_hand_green.bat                                       # 默认相机 0，实时
    run_hand_green.bat --source "Left Eye View_screenshot_13.10.2026.png" --save
    run_hand_green.bat --source 1 --width 1280 --height 720   # 换相机与分辨率
    run_hand_green.bat --green-lo 30 60 80 --min-area 0.01    # 光照不同时调绿色阈值

热键：Q/ESC 退出，S 存当前帧。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

for _stream in (sys.stdout, sys.stderr):      # 防止中文提示在 cmd 里变乱码
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.common import OUTPUT, save_image  # noqa: E402
from src.modules.green_regions import GreenConfig, detect_green_regions, draw_green  # noqa: E402
from src.modules.hand_tracker import HandTracker  # noqa: E402
from src.stream import FpsMeter, StreamReader  # noqa: E402

WINDOW = "Hand + Green Regions"
HAND_COLOR = (255, 0, 220)      # 洋红
SKELETON = [                    # 21 点骨架（MediaPipe 顺序），只画有意义的那几条
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


def draw_hands(image, hands, show_keypoints=True):
    """画手框 + 跟踪 ID（+ 关键点，若模型有输出）。"""
    for hand in hands:
        x1, y1, x2, y2 = (int(round(v)) for v in hand.box)
        cv2.rectangle(image, (x1, y1), (x2, y2), HAND_COLOR, 2)
        text = f"hand#{hand.track_id}" if hand.track_id >= 0 else "hand"
        text += f" {hand.conf:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(image, (x1, y1 - th - 8), (x1 + tw + 6, y1), HAND_COLOR, -1)
        cv2.putText(image, text, (x1 + 3, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 0), 2, cv2.LINE_AA)

        if not show_keypoints or not hand.keypoints:
            continue
        for a, b in SKELETON:
            if a >= len(hand.keypoints) or b >= len(hand.keypoints):
                continue
            xa, ya, va = hand.keypoints[a]
            xb, yb, vb = hand.keypoints[b]
            if va > 0 and vb > 0:
                cv2.line(image, (int(xa), int(ya)), (int(xb), int(yb)), HAND_COLOR, 1,
                         cv2.LINE_AA)
        for x, y, v in hand.keypoints:
            if v > 0:
                cv2.circle(image, (int(x), int(y)), 3, (255, 255, 255), -1, cv2.LINE_AA)
    return image


def draw_bar(image, text):
    cv2.rectangle(image, (0, 0), (image.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(image, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                (0, 255, 255), 1, cv2.LINE_AA)
    return image


def main() -> int:
    parser = argparse.ArgumentParser(
        description="只做手部识别 + 绿色区域框选",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--source", default="0", help="相机序号、视频文件或图片")
    parser.add_argument("--backend", choices=("auto", "dshow", "msmf"), default="dshow")
    parser.add_argument("--width", type=int, default=1280, help="相机宽度（0=不设）")
    parser.add_argument("--height", type=int, default=720, help="相机高度（0=不设）")
    parser.add_argument("--fps", type=int, default=30, help="相机帧率（0=不设）")
    parser.add_argument("--layout", choices=("auto", "side_by_side", "top_bottom", "none"),
                        default="none", help="双目拼接帧拆分；单目画面用 none")
    parser.add_argument("--hand", default=str(ROOT / "weights" / "yolo26n-pose-21kpt.pt"),
                        help="手部权重")
    parser.add_argument("--hand-conf", type=float, default=0.3, help="手部置信度阈值")
    parser.add_argument("--imgsz", type=int, default=640, help="推理尺寸")
    parser.add_argument("--device", default=None, help="cuda:0 / cpu")
    parser.add_argument("--tracker",
                        default=str(ROOT / "configs" / "trackers" / "bytetrack.yaml"),
                        help="跟踪器配置")
    parser.add_argument("--green-lo", nargs=3, type=int, metavar=("H", "S", "V"),
                        default=[35, 80, 100], help="绿色 HSV 下限")
    parser.add_argument("--green-hi", nargs=3, type=int, metavar=("H", "S", "V"),
                        default=[95, 255, 255], help="绿色 HSV 上限")
    parser.add_argument("--min-area", type=float, default=0.004,
                        help="绿色块最小面积占整幅画面的比例")
    parser.add_argument("--merge-gap", type=int, default=0, help="相邻绿色块合并间距")
    parser.add_argument("--no-keypoints", action="store_true", help="不画关键点")
    parser.add_argument("--no-show", action="store_true", help="不开窗口")
    parser.add_argument("--save", action="store_true", help="保存结果图/视频")
    parser.add_argument("--save-dir", default=None, help="输出目录，默认 output/")
    args = parser.parse_args()

    try:
        reader = StreamReader(source=args.source, backend=args.backend, width=args.width,
                              height=args.height, fps=args.fps, layout=args.layout)
    except RuntimeError as error:
        print(f"[错误] {error}")
        return 2
    print(f"[视频源] {reader.describe()}")

    green = GreenConfig(lo=tuple(args.green_lo), hi=tuple(args.green_hi),
                        min_area_ratio=args.min_area, merge_gap=args.merge_gap)
    tracker = HandTracker(weights=args.hand, tracker=args.tracker, conf=args.hand_conf,
                          imgsz=args.imgsz, device=args.device, verbose=True)
    print(f"[手部] {tracker.names} | 任务 {getattr(tracker.model, 'task', '?')}")
    print(f"[绿色] HSV {tuple(args.green_lo)}~{tuple(args.green_hi)} | 最小面积 {args.min_area:.1%}")

    out_dir = Path(args.save_dir) if args.save_dir else OUTPUT
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    show = not args.no_show and not reader.is_image and hasattr(cv2, "imshow")
    if show:
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)

    meter = FpsMeter()
    writer, frames, missing = None, 0, 0
    last = None
    try:
        while True:
            ok, frame = reader.read()
            if not ok:
                break
            frames += 1

            regions, warning = detect_green_regions(frame, green)
            if warning:
                missing += 1
            hands = tracker.update(frame)
            fps = meter.tick()

            canvas = draw_green(frame, regions)
            draw_hands(canvas, hands, show_keypoints=not args.no_keypoints)
            bar = f"fps {fps:.1f} | 绿区 {len(regions)} | 手 {len(hands)}"
            if warning:
                bar += f"  ! {warning}"
            draw_bar(canvas, bar)
            last = canvas

            if args.save:
                if reader.is_image:
                    path = out_dir / f"annotated_{Path(reader.source).stem}.png"
                    save_image(path, canvas)
                    print(f"[保存] {path}")
                else:
                    if writer is None:
                        path = out_dir / f"hand_green_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
                        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"),
                                                 reader.fps_hint or 25.0,
                                                 (canvas.shape[1], canvas.shape[0]))
                        if writer.isOpened():
                            print(f"[录制] {path}")
                        else:
                            writer.release()
                            writer = None
                            print("[警告] 视频写入器打不开，改为每 30 帧存一张图")
                    if writer is not None:
                        writer.write(canvas)
                    elif frames % 30 == 0:
                        save_image(out_dir / f"frame_{frames:06d}.png", canvas)

            if show:
                cv2.imshow(WINDOW, canvas)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("s") and last is not None:
                    path = out_dir / f"snapshot_{time.strftime('%H%M%S')}.png"
                    save_image(path, last)
                    print(f"[保存] {path}")
            elif reader.is_image:
                break
    except KeyboardInterrupt:
        print("\n[中断] 用户停止")
    finally:
        reader.release()
        if writer is not None:
            writer.release()
        if show:
            cv2.destroyAllWindows()

    print(f"\n处理 {frames} 帧；未识别到绿区的帧 {missing}"
          + (f"（请调 --green-lo/--green-hi/--min-area）" if missing == frames and frames else ""))
    if tracker.last_error:
        print(f"[警告] 跟踪器最后一次报错：{tracker.last_error}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
