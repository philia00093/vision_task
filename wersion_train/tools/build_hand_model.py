"""生成手部 21 关键点姿态模型文件 yolo26n-pose-21kpt.pt。

背景：Ultralytics 权重表里的 yolo26n-pose.pt 是 COCO 人体 **17** 点姿态模型，
官方没有手部 21 点权重。本脚本按官方 YOLO26-pose 结构派生出一个 **21** 关键点模型：
    · 结构不做手工复制，而是读取已安装的 cfg/models/26/yolo26-pose.yaml 派生，
      只覆盖 kpt_shape / nc / scale 三处，因此不会因为模块名写错而建不出模型；
    · 把 17 点权重里形状匹配的部分（主干/颈部）迁移过来，关键点分支按 21 点新建；
    · 不改 site-packages：派生配置写在项目 wersion_train/weights/ 下，通过环境变量告诉
      ultralytics 去那里找模型 yaml，安装包本身保持原样；
    · 不装任何新包，只用现有的 ultralytics + torch。

21 点顺序与 MediaPipe 一致，见 wersion_train/configs/hand_keypoints_21.txt。

用法：
    python wersion_train/tools/build_hand_model.py
    python wersion_train/tools/build_hand_model.py --scale s      # 换 yolo26s 主干
    python wersion_train/tools/build_hand_model.py --dry-run      # 只看派生配置

产物：wersion_train/weights/yolo26n-pose-21kpt.pt，之后：
    run_pipeline.bat --hand wersion_train/weights/yolo26n-pose-21kpt.pt
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass

WEIGHTS = ROOT / "weights"
# ultralytics 会优先从 VERSION_TASK_MODELS_DIR 找模型 yaml，这样派生配置不必写进安装目录
os.environ.setdefault("VERSION_TASK_MODELS_DIR", str(WEIGHTS))

KPT_SHAPE = [21, 3]
HAND_CLASSES = 1        # 只有一个 hand 类；要分左右手就改成 2 并同步改标签
# 21 点名称顺序文件（仅用于写进权重做可读性标注，不参与计算）
KPT_NAMES_FILE = ROOT / "configs" / "hand_keypoints_21.txt"


def read_kpt_names() -> dict[int, str]:
    """读取 wersion_train/configs/hand_keypoints_21.txt 的 '序号 名称' 列表。"""
    names: dict[int, str] = {}
    if not KPT_NAMES_FILE.is_file():
        return names
    for raw in KPT_NAMES_FILE.read_text(encoding="utf-8").splitlines():
        body = raw.split("#", 1)[0].strip()
        if not body:
            continue
        parts = body.split(None, 1)
        if len(parts) == 2 and parts[0].isdigit():
            names[int(parts[0])] = parts[1].strip()
    return names


def attach_pose_metadata(model, verbose: bool = True) -> None:
    """给手搓出来的 PoseModel 补上 kpt_shape / kpt_names。

    Ultralytics 的 PoseModel 不在 __init__ 里设置这两个属性，而是由训练器设置，
    save() 时随模型一起 pickle。手工构建的模型缺了它们，推理时
    pose/predict.py 会去读 self.model.kpt_shape 并抛
    AttributeError: 'AutoBackend' object has no attribute 'kpt_shape'。
    官方 yolo26n-pose.pt 里就有这两个属性，这里照做。
    """
    inner = getattr(model, "model", model)          # YOLO 包装 → PoseModel
    shape = inner.yaml.get("kpt_shape") or list(KPT_SHAPE)
    inner.kpt_shape = list(shape)
    names = read_kpt_names()
    if names:
        inner.kpt_names = {i: names[i] for i in sorted(names)}
    # 类别名：派生配置只有 nc=1 没有名字，补成 hand，否则后面日志里显示成 '0'
    inner.names = {0: "hand"} if int(getattr(inner, "nc", 1) or 1) == 1 else dict(inner.names)
    if verbose:
        print(f"[元数据] kpt_shape = {inner.kpt_shape}"
              f"，kpt_names = {len(getattr(inner, 'kpt_names', {}) or {})} 个"
              f"，names = {inner.names}")


def verify_saved(path: Path) -> bool:
    """回读权重文件确认元数据与关键点形状都在，避免又出一个跑不动的权重。"""
    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    inner = checkpoint.get("model")
    head = inner.model[-1] if inner is not None else None
    ok_shape = getattr(head, "kpt_shape", None) == list(KPT_SHAPE)
    ok_meta = isinstance(getattr(inner, "kpt_shape", None), (list, tuple))
    print(f"[校验] 权重内 kpt_shape {getattr(inner, 'kpt_shape', None)}"
          f" | 检测头 {getattr(head, 'kpt_shape', None)}"
          f" | kpt_names {len(getattr(inner, 'kpt_names', {}) or {})}"
          f" → {'通过' if (ok_shape and ok_meta) else '不通过'}")
    return ok_shape and ok_meta


def derive_config(scale: str, nc: int) -> tuple[str, Path]:
    """读取官方 yolo26-pose.yaml，覆盖 scale/nc/kpt_shape 后写回项目内。"""
    from ultralytics.nn.tasks import check_yaml
    from ultralytics.utils import YAML
    from ultralytics import __file__ as ul_init

    base_yaml = Path(ul_init).parent / "cfg" / "models" / "26" / "yolo26-pose.yaml"
    if not base_yaml.is_file():
        raise FileNotFoundError(f"找不到官方姿态模型配置：{base_yaml}")

    config = YAML.load(check_yaml(base_yaml))     # 已解析成 dict，不需要保留注释
    original_shape = config.get("kpt_shape")
    original_nc = config.get("nc")
    config["kpt_shape"] = list(KPT_SHAPE)
    config["nc"] = int(nc)
    config["scale"] = scale

    print(f"[派生] 源配置 {base_yaml.name}：kpt_shape {original_shape} → {config['kpt_shape']}"
          f"，nc {original_nc} → {config['nc']}，scale {scale}")
    if "scales" in config and scale not in config["scales"]:
        raise ValueError(f"源配置不支持 scale={scale}，可选 {list(config['scales'])}")

    out_path = WEIGHTS / f"yolo26{scale}-pose-21kpt.yaml"
    YAML.save(out_path, config, header=(
        "# 手部 21 关键点 YOLO26 姿态模型配置（由 wersion_train/tools/build_hand_model.py 生成）\n"
        f"# 派生自官方 {base_yaml.name}，仅覆盖 scale / nc / kpt_shape\n"
        "# kpt_shape: [21, 3] —— 21 点顺序见 wersion_train/configs/hand_keypoints_21.txt\n"))
    return str(out_path), out_path


def main() -> int:
    parser = argparse.ArgumentParser(description="生成 21 关键点手部姿态模型文件")
    parser.add_argument("--scale", default="n", choices=("n", "s", "m", "l", "x"),
                        help="模型尺寸，默认 n（和 yolo26n 一致）")
    parser.add_argument("--nc", type=int, default=HAND_CLASSES,
                        help="类别数，默认 1（只有 hand）")
    parser.add_argument("--base", default=None,
                        help="迁移权重的源权重，默认 wersion_train/weights/yolo26<scale>-pose.pt")
    parser.add_argument("--out", default=None,
                        help="输出权重，默认 wersion_train/weights/yolo26<scale>-pose-21kpt.pt")
    parser.add_argument("--dry-run", action="store_true", help="只派生配置不建模型")
    args = parser.parse_args()

    WEIGHTS.mkdir(parents=True, exist_ok=True)
    config_path, _ = derive_config(args.scale, args.nc)
    print(f"[派生] 已写入 {config_path}")

    if args.dry_run:
        print(Path(config_path).read_text(encoding="utf-8"))
        return 0

    base_path = Path(args.base) if args.base else WEIGHTS / f"yolo26{args.scale}-pose.pt"
    out_path = Path(args.out) if args.out else WEIGHTS / f"yolo26{args.scale}-pose-21kpt.pt"

    from ultralytics import YOLO

    print(f"[建模型] 用派生配置构建 21 点模型")
    model = YOLO(config_path)
    # kpt_shape / nc 挂在 Pose26 检测头上，不在 PoseModel 顶层
    try:
        head = model.model.model[-1]
        print(f"[建模型] kpt_shape = {getattr(head, 'kpt_shape', None)}"
              f"，nc = {getattr(head, 'nc', None)}，检测头 = {type(head).__name__}")
    except Exception as error:
        print(f"[建模型] 读取检测头信息失败（不影响建模型）：{error}")

    if base_path.is_file():
        model.load(str(base_path))
        print(f"[迁移] 已从 {base_path.name} 迁移形状匹配的权重（主干/颈部）")
    else:
        print(f"[迁移] 源权重不存在（{base_path}），本次为全新初始化；"
              "可先跑 wersion_train/tools/download_weights.bat --pose")

    attach_pose_metadata(model)          # 关键一步：补 kpt_shape / kpt_names
    model.save(str(out_path))
    if not verify_saved(out_path):
        print("[失败] 权重元数据不完整，推理会报 kpt_shape 错误；请把上面的输出发我")
        return 1

    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"\n[完成] {out_path}（{size_mb:.1f} MB）")
    print(f"       用法：run_pipeline.bat --hand {out_path.name} --source "
          f"\"Left Eye View_screenshot_13.10.2026.png\" --save")
    print("       注意：关键点分支是按 21 点新建的、尚未训练，点位不准；"
          "检测框与跟踪 ID 现在可用。")
    print("             要得到准确的 21 点：标 wersion_train/configs/data_hand_pose.yaml → "
          "wersion_train/train_pose.py --data wersion_hand_pose")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
