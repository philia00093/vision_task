"""下载 YOLO26 预训练权重到 weights/。

用法（必须用 yolo26 环境，别用 base）：
    使用 download_weights.bat
    或  D:\\anaconda3\\envs\\yolo26\\python.exe scripts\\download_weights.py

默认只拉两个检测权重：yolo26n.pt（训练主力）和 yolo26.pt（大模型，精度上限参考）。
加 --pose 额外拉 yolo26n-pose.pt，用于手部 21 关键点训练（COCO 人体姿态预训练权重，
关键点分支会被重建为 21 点，但主干特征仍是有效的预训练起点）。

已存在的文件默认跳过，--force 可强制重下。
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "weights"

# ultralytics 会从 GitHub release 拉取；已存在同名文件时也会跳过，这里再兜一层显式控制
DEFAULT_MODELS = ["yolo26n.pt", "yolo26.pt"]
POSE_MODELS = ["yolo26n-pose.pt"]


def human(size: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{size} B"
        size /= 1024.0
    return f"{size:.1f} GB"


def find_weight(name: str) -> Path | None:
    """权重可能被 ultralytics 落在 weights/、项目根或 cwd，逐处查找。

    attempt_download_asset 在部分版本里返回裸文件名而不是真实路径，
    所以下载后必须以磁盘实际情况为准，不能相信返回值。
    """
    for candidate in (WEIGHTS / name, ROOT / name, Path.cwd() / name):
        if candidate.is_file():
            return candidate
    return None


def download(name: str, force: bool = False, tag: str = "v8.4.0") -> Path | None:
    existing = find_weight(name)
    if existing and not force:
        print(f"[跳过] {name} 已存在：{existing}（{human(existing.stat().st_size)}）")
        return existing

    from ultralytics.utils.downloads import download as ul_download

    WEIGHTS.mkdir(parents=True, exist_ok=True)
    url = f"https://github.com/ultralytics/assets/releases/download/{tag}/{name}"
    print(f"[下载] {name} <- {url}")
    before = existing
    try:
        ul_download(url, dir=WEIGHTS)
    except Exception as error:  # 网络/代理/404 都落到这里，给出可操作提示
        print(f"[失败] {name}：{error}")
        print("       可手动下载后放入 weights/ ："
              f"https://github.com/ultralytics/assets/releases 里的 {name}")
        return None

    path = find_weight(name)
    if path is None:
        print(f"[失败] {name}：下载调用未报错，但 weights/ 下没有找到该文件")
        return None
    if path != WEIGHTS / name:  # 落错位置就搬回 weights/
        target = WEIGHTS / name
        shutil.move(str(path), str(target))
        path = target
    if before is not None and path.stat().st_size != before.stat().st_size:
        print(f"[警告] {name} 大小与已有文件不一致，已用新下载的覆盖")
    print(f"[完成] {path}（{human(path.stat().st_size)}）")
    return path


def verify_hand_keypoint_support() -> None:
    """确认当前 ultralytics 认识 yolo26 系列资源，避免版本过旧白跑一次训练。"""
    try:
        from ultralytics.utils.downloads import GITHUB_ASSETS_NAMES as names
    except Exception as error:
        print(f"[警告] 无法读取 ultralytics 资源表：{error}")
        return
    found = sorted(n for n in names if n.startswith("yolo26"))
    if not found:
        print("[警告] 当前 ultralytics 资源表里没有 yolo26，请升级："
              "pip install -U ultralytics")
        return
    print(f"[信息] 当前 ultralytics 提供 {len(found)} 个 yolo26 权重，"
          f"例如：{', '.join(found[:4])} ...")


def main() -> int:
    parser = argparse.ArgumentParser(description="下载 YOLO26 预训练权重")
    parser.add_argument("--models", nargs="*", default=None,
                        help=f"自定义权重名，默认 {DEFAULT_MODELS}")
    parser.add_argument("--pose", action="store_true",
                        help=f"额外下载姿态权重 {POSE_MODELS}（手部关键点训练用）")
    parser.add_argument("--force", action="store_true", help="已存在也重新下载")
    args = parser.parse_args()

    import ultralytics
    print(f"[信息] ultralytics {ultralytics.__version__} / python {sys.version.split()[0]}")
    verify_hand_keypoint_support()

    names = list(args.models) if args.models else list(DEFAULT_MODELS)
    if args.pose:
        names += POSE_MODELS

    ok, failed = [], []
    for name in names:
        path = download(name, force=args.force)
        (ok if path else failed).append(name)

    print(f"\n成功 {len(ok)}：{', '.join(ok) if ok else '无'}")
    if failed:
        print(f"失败 {len(failed)}：{', '.join(failed)}")
    print(f"权重目录：{WEIGHTS}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
