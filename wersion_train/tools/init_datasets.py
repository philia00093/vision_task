"""建立 wersion_train 的文件夹结构（幂等，可反复运行）。

只建目录，不塞多余文件：
    wersion_train/
      datasets/            四个数据集（区域 / 零件 / 步骤 / 手部21点）
        <数据集>/{images,labels}/{train,val,raw}
      parts/               零件四类：bottom 支撑柱 盘 罗盘
        <类别>/{images,labels}
      steps/               步骤：step_0 ~ step_3 + complete
        <步骤>/{images,labels}
      annotations/         X-AnyLabeling 导出统一放这里
      raw_frames/          视频抽帧的原始帧
      runs/                训练输出（权重、曲线）
      events/              手跨越区域的事件 CSV
      tools/               标注与训练用脚本

用法：
    wersion_train\\tools\\init_datasets.bat
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WERSION = ROOT / "wersion_train"

DATASETS = ("wersion_parts", "wersion_steps")
PARTS = ("bottom", "support_pillar", "tray", "compass")
STEPS = ("step_0", "step_1", "step_2", "step_3", "complete")


def main() -> int:
    dirs: list[Path] = []
    for name in DATASETS:
        base = WERSION / "datasets" / name
        dirs += [base / "images" / sub for sub in ("train", "val", "raw")]
        dirs += [base / "labels" / sub for sub in ("train", "val", "raw")]
    for name in PARTS:
        dirs += [WERSION / "parts" / name / sub for sub in ("images", "labels")]
    for name in STEPS:
        dirs += [WERSION / "steps" / name / sub for sub in ("images", "labels")]
    dirs += [WERSION / sub for sub in ("annotations", "raw_frames", "runs", "events", "tools")]

    for directory in dirs:
        directory.mkdir(parents=True, exist_ok=True)

    print(f"[完成] 建立 {len(dirs)} 个目录，根目录：{WERSION}")
    print("\n目录树：")
    print("wersion_train/")
    print("├─ datasets/                 训练用数据集（yaml 在 wersion_train/configs/ 下）")
    for name in DATASETS:
        print(f"│  ├─ {name}/")
        print("│  │  ├─ images/{train,val,raw}/")
        print("│  │  └─ labels/{train,val,raw}/")
    print("├─ parts/                    零件四类：底部 / 支撑柱 / 盘 / 罗盘")
    for name in PARTS:
        print(f"│  ├─ {name}/{{images,labels}}/")
    print("├─ steps/                    装配步骤：step_0 → step_3 → 完整体")
    for name in STEPS:
        print(f"│  ├─ {name}/{{images,labels}}/")
    print("├─ annotations/              X-AnyLabeling 导出统一放这里")
    print("├─ raw_frames/               视频抽帧的原始帧")
    print("├─ runs/                     训练输出（权重、曲线）")
    print("├─ events/                   手跨越区域的事件 CSV")
    print("└─ tools/                    标注与训练用脚本")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
