"""从视频里抽帧，作为标注素材。

抽帧策略（默认组合，避免全是相近的废帧）：
    --every N        每 N 帧取一帧（最省事）
    --motion T       与上一张保留帧的平均绝对差低于 T 就丢弃（去重静止画面）
    --blur T         拉普拉斯方差低于 T 的模糊帧丢弃（去抖）
    --limit N        最多保存 N 张

用法：
    python vision_yolo_train/tools/extract_frames.py --source 视频流.mp4 --every 10
    python vision_yolo_train/tools/extract_frames.py --source 0 --every 15 --limit 300
    python vision_yolo_train/tools/extract_frames.py --source 视频流.mp4 --every 5 --split-ratio 0.5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import paths  # noqa: E402  统一路径定义（见 vision_yolo_train/paths.py）




def laplacian_var(gray) -> float:
    """清晰度指标：越小越模糊。"""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def main() -> int:
    parser = argparse.ArgumentParser(description="视频抽帧")
    parser.add_argument("--source", required=True, help="视频文件路径或相机序号")
    parser.add_argument("--out", default=None,
                        help="输出目录，默认 vision_yolo_train/raw_frames/<视频名>")
    parser.add_argument("--every", type=int, default=10, help="每 N 帧取一帧")
    parser.add_argument("--motion", type=float, default=1.5,
                        help="与上一保留帧的平均绝对差阈值（0=不去重）")
    parser.add_argument("--blur", type=float, default=30.0,
                        help="拉普拉斯方差阈值，低于此值判为模糊（0=不过滤）")
    parser.add_argument("--limit", type=int, default=0, help="最多保存张数（0=不限）")
    parser.add_argument("--resize", type=int, default=0, help="缩放到该宽度（0=原尺寸）")
    parser.add_argument("--start", type=int, default=0, help="起始帧")
    parser.add_argument("--end", type=int, default=0, help="结束帧（0=到结尾）")
    parser.add_argument("--prefix", default="frame", help="文件名前缀")
    args = parser.parse_args()

    source = str(args.source)
    out_dir = Path(args.out) if args.out else (
        paths.ROOT / "vision_yolo_train" / "raw_frames" /
        (Path(source).stem if Path(source).suffix else f"cam{source}"))
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    if not cap.isOpened():
        print(f"[错误] 打不开视频源：{source}")
        return 2
    if args.start:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.start)

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    print(f"[抽帧] {source} 总帧 {total or '未知'} fps {fps:.1f} → {out_dir}")

    previous = None
    index = kept = skipped_blur = skipped_motion = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        index += 1
        if args.end and index > args.end:
            break
        if (index - 1) % max(args.every, 1) != 0:
            continue

        if args.resize and frame.shape[1] != args.resize:
            scale = args.resize / float(frame.shape[1])
            frame = cv2.resize(frame, (args.resize, int(round(frame.shape[0] * scale))),
                               interpolation=cv2.INTER_AREA)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if args.blur and laplacian_var(gray) < args.blur:
            skipped_blur += 1
            continue
        if args.motion and previous is not None:
            if float(np.abs(gray.astype(np.int16) - previous).mean()) < args.motion:
                skipped_motion += 1
                continue
        previous = gray.astype(np.int16)

        name = f"{args.prefix}_{index:06d}.jpg"
        # 中文路径用 imencode+tofile，避免 cv2.imwrite 静默失败
        ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        if ok:
            buffer.tofile(str(out_dir / name))
            kept += 1
        if args.limit and kept >= args.limit:
            break

    cap.release()
    print(f"[完成] 读 {index} 帧，保存 {kept} 张；跳过滤糊 {skipped_blur}、静止 {skipped_motion}")
    print(f"        下一步：用 X-AnyLabeling 打开 {out_dir} 标注")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
