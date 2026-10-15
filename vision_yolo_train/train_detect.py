"""用 YOLO26n 训练检测模型（零件 / 步骤 / 区域 都用这一个入口）。

不配环境、不改全局设置，只读 vision_yolo_train/configs/data_<name>.yaml 并调用 ultralytics。

用法：
    python vision_yolo_train/train_detect.py --data wersion_parts
    python vision_yolo_train/train_detect.py --data wersion_steps --epochs 300 --imgsz 960
    python vision_yolo_train/train_detect.py --data wersion_parts --model vision_yolo_train/weights/yolo26s.pt
    python vision_yolo_train/train_detect.py --data wersion_parts --resume

默认参数按 8G 显存（RTX 5060 Laptop）给：imgsz 640、batch 8。
零件小、步骤细，识别不准时优先把 imgsz 提到 960 并补困难样本。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths  # noqa: E402  统一路径定义（见 vision_yolo_train/paths.py）



DATASETS = paths.DATASET_YAML          # 数据集名 -> data yaml（见 paths.py）
DEFAULT_WEIGHTS = paths.DEFAULT_DETECT_WEIGHTS


def main() -> int:
    parser = argparse.ArgumentParser(description="YOLO26n 检测训练")
    parser.add_argument("--data", default="wersion_parts",
                        help="数据集名（wersion_parts（零件）/ wersion_steps（步骤））")
    parser.add_argument("--yaml", default=None, help="直接指定 data yaml，覆盖 --data")
    parser.add_argument("--model", default=str(DEFAULT_WEIGHTS),
                        help="起始权重；默认 vision_yolo_train/weights/yolo26n.pt（nano）")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--imgsz", type=int, default=640, help="小零件建议 960")
    parser.add_argument("--batch", type=int, default=8, help="-1=自动")
    parser.add_argument("--device", default="0", help="0=第一块 GPU，cpu 或不填为自动")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=50, help="早停轮数")
    parser.add_argument("--name", default=None, help="本次运行名，默认 <数据集>")
    parser.add_argument("--resume", action="store_true", help="从上次中断处继续")
    parser.add_argument("--cache", action="store_true", help="缓存图片到内存，加速小数据集")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--extra", nargs=argparse.REMAINDER, default=[],
                        help="其余参数原样透传给 ultralytics，例如 --extra cos_lr=True")
    args = parser.parse_args()

    data_yaml = Path(args.yaml) if args.yaml else paths.DATASETS.get(args.data)
    if data_yaml is None or not data_yaml.is_file():
        print(f"[错误] 找不到数据配置：{data_yaml}；可选 {list(paths.DATASETS)}")
        return 2
    if not Path(args.model).is_file():
        print(f"[错误] 找不到权重 {args.model}；先跑 vision_yolo_train/tools/download_weights.bat")
        return 2

    from ultralytics import YOLO

    model = YOLO(args.model)
    print(f"[训练] 数据 {data_yaml}\n       权重 {args.model}\n"
          f"       epochs {args.epochs} imgsz {args.imgsz} batch {args.batch} device {args.device}")

    overrides = {}
    for item in args.extra:
        if "=" in item:
            key, value = item.split("=", 1)
            overrides[key] = value

    model.train(
        data=str(data_yaml), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        device=args.device, workers=args.workers, patience=args.patience,
        project=str(paths.ROOT / "vision_yolo_train" / "runs"), name=args.name or args.data,
        resume=args.resume, cache=args.cache, seed=args.seed, exist_ok=True, **overrides)

    print(f"\n[完成] 权重在 {paths.ROOT / 'vision_yolo_train' / 'runs' / (args.name or args.data)}/vision_yolo_train/weights/")
    print("       用起来：python train_detect.py --data wersion_steps  继续训步骤模型")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
