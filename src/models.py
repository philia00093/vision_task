"""模型装配：权重路径解析 + 统一的 ultralytics 模型包装。

为什么单独一层：ultralytics 的 YOLO(...) 接受裸权重名并会自己去 GitHub 找；
本项目要求权重必须来自 weights/ 目录，避免训练时莫名其妙下了另一个模型。
所以这里统一解析路径、校验存在性，并把"未安装 ultralytics/torch"报成可读的错误。
"""

from __future__ import annotations

from pathlib import Path

from .common import DEFAULT_DETECT_WEIGHTS, DEFAULT_POSE_WEIGHTS, WEIGHTS


class ModelNotReady(RuntimeError):
    pass


def resolve_weights(name: str | Path | None, kind: str = "detect") -> Path:
    """把权重参数解析成真实文件路径。

    允许的写法：
        None / "auto"     → 按 kind 取 weights/yolo26n.pt 或 weights/yolo26n-pose.pt
        "yolo26n.pt"      → weights/yolo26n.pt（找不到再按裸名交给 ultralytics）
        "runs/xxx/best.pt"→ 相对项目根或绝对路径
    """
    fallback = DEFAULT_POSE_WEIGHTS if kind == "pose" else DEFAULT_DETECT_WEIGHTS
    if name is None or str(name).lower() in ("", "auto"):
        return fallback

    candidate = Path(str(name)).expanduser()
    if candidate.is_file():
        return candidate
    if not candidate.is_absolute():
        for base in (WEIGHTS, Path.cwd()):
            probe = base / candidate
            if probe.is_file():
                return probe
    # 到这一步说明文件不在，但可能是 ultralytics 认识的官方权重名，交给它下载
    if candidate.suffix == ".pt" and candidate.parent == Path("."):
        return candidate
    raise ModelNotReady(
        f"权重文件不存在：{name}\n"
        f"  已查找：{WEIGHTS}、{Path.cwd()}\n"
        f"  请先运行 scripts/download_weights.bat，或把 .pt 放进 weights/ 目录"
    )


def default_weights(kind: str = "detect") -> Path:
    return DEFAULT_POSE_WEIGHTS if kind == "pose" else DEFAULT_DETECT_WEIGHTS


def load_yolo(weights, task: str | None = None, verbose: bool = False):
    """加载 ultralytics 模型；把环境缺失报成中文可读错误。

    task 传 None 时交给 ultralytics 自己从权重推断任务，
    这样自训练的手部姿态权重（best.pt）也能被正确识别为 pose。
    """
    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise ModelNotReady(
            f"当前 Python 环境缺少 ultralytics/torch（{error}）。\n"
            "  本项目请用 yolo26 环境：D:\\anaconda3\\envs\\yolo26\\python.exe\n"
            "  即运行 run_pipeline.bat，而不是直接 python src/run_pipeline.py"
        ) from error

    path = resolve_weights(weights if weights is not None else default_weights(
        "pose" if task == "pose" else "detect"), kind=task or "detect")
    if verbose:
        print(f"[模型] 载入 {path}")
    return YOLO(str(path), task=task) if task else YOLO(str(path))
