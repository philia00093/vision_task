"""建立 vision_yolo_train 的文件夹结构（幂等，可反复运行）。

只建目录，不塞多余文件：
    vision_yolo_train/
      datasets/            四个数据集（区域 / 零件 / 步骤 / 手部21点）
        <数据集>/{images,labels}/{train,val,raw}
      parts/               零件四类：base 底部 / support 支撑柱 / compass 罗盘 / top 盘
        <类别>/{images,labels}
      steps/               步骤：step_0 ~ step_3 + complete
        <步骤>/{images,labels}
      annotations/         X-AnyLabeling 导出统一放这里
      raw_frames/          视频抽帧的原始帧
      runs/                训练输出（权重、曲线）
      events/              手跨越区域的事件 CSV
      tools/               标注与训练用脚本

用法：
    vision_yolo_train\\tools\\init_datasets.bat
"""

from __future__ import annotations

from pathlib import Path

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import paths  # noqa: E402  统一路径定义（见 vision_yolo_train/paths.py）



DATASETS = ("wersion_parts", "wersion_steps")
PARTS = ("base", "support", "compass", "top")   # 与采集图片前缀一致
STEPS = ("step_0", "step_1", "step_2", "step_3", "complete")


def main() -> int:
    dirs: list[Path] = []
    for name in paths.DATASET_NAMES:
        base = paths.DATASETS / name
        dirs += [base / "images" / sub for sub in ("train", "val", "raw")]
        dirs += [base / "labels" / sub for sub in ("train", "val", "raw")]
    for name in PARTS:
        dirs += [paths.ANNOTATE / "parts" / name / sub for sub in ("images", "labels")]
    for name in STEPS:
        dirs += [paths.ANNOTATE / "steps" / name / sub for sub in ("images", "labels")]
    dirs += [paths.ANNOTATIONS, paths.RAW_FRAMES, paths.RUNS, paths.TOOLS,
             paths.CONFIGS, paths.WEIGHTS, paths.IMAGES]

    for directory in dirs:
        directory.mkdir(parents=True, exist_ok=True)

    print(f"[完成] 建立 {len(dirs)} 个目录，根目录：{paths.ROOT}")
    print("\n目录树：")
    print("vision_yolo_train/")
    print("├─ datasets/                 训练用数据集（yaml 在 vision_yolo_train/configs/ 下）")
    for name in paths.DATASET_NAMES:
        print(f"│  ├─ {name}/")
        print("│  │  ├─ images/{train,val,raw}/")
        print("│  │  └─ labels/{train,val,raw}/")
    print("├─ datasets/annotate/        按类别分开的标注工作区")
    for name in PARTS:
        print(f"│  ├─ parts/{name}/{{images,labels}}/")
    for name in STEPS:
        print(f"│  ├─ steps/{name}/{{images,labels}}/")
    print("├─ images/                   待标注图片：parts/（零件） steps/（步骤）")
    print("├─ annotations/              X-AnyLabeling 导出统一放这里")
    print("├─ raw_frames/               视频抽帧的原始帧")
    print("├─ runs/                     训练输出（权重、曲线）")
    print("├─ tools/                    抽帧 / 转换标注 / 划分 / 体检 / 下权重")
    print("├─ configs/                  数据集 yaml 与类别表")
    print("├─ weights/                  预训练权重")
    print("└─ train_detect.py           训练入口")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
