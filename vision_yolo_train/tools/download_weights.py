"""下载 YOLO26 预训练权重到 vision_yolo_train/weights/。

用法（用 yolo26 环境）：
    python vision_yolo_train/tools/download_weights.py
    或双击 vision_yolo_train/tools/download_weights.bat

默认只拉 yolo26n.pt（训练主力，nano）。
注意：Ultralytics 只发布带尺寸后缀的权重（yolo26n/s/m/l/x.pt），
      没有名为 yolo26.pt 的文件，所以别照抄这个名字。
要换更大的主干：--models yolo26s.pt（或 m/l/x）。

已存在的文件默认跳过，--force 可强制重下。
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import paths  # noqa: E402  统一路径定义（见 vision_yolo_train/paths.py）



# ultralytics 会从 GitHub release 拉取；已存在同名文件时也会跳过，这里再兜一层显式控制
DEFAULT_MODELS = ["yolo26n.pt"]


def human(size: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{size} B"
        size /= 1024.0
    return f"{size:.1f} GB"


def find_weight(name: str) -> Path | None:
    """权重可能被 ultralytics 落在 vision_yolo_train/weights/、项目根或 cwd，逐处查找。

    attempt_download_asset 在部分版本里返回裸文件名而不是真实路径，
    所以下载后必须以磁盘实际情况为准，不能相信返回值。
    """
    for candidate in (paths.WEIGHTS / name, paths.ROOT / name, Path.cwd() / name):
        if candidate.is_file():
            return candidate
    return None


def download(name: str, force: bool = False, tag: str = "v8.4.0") -> Path | None:
    existing = find_weight(name)
    if existing and not force:
        print(f"[跳过] {name} 已存在：{existing}（{human(existing.stat().st_size)}）")
        return existing

    from ultralytics.utils.downloads import download as ul_download

    paths.WEIGHTS.mkdir(parents=True, exist_ok=True)
    url = f"https://github.com/ultralytics/assets/releases/download/{tag}/{name}"
    print(f"[下载] {name} <- {url}")
    before = existing
    try:
        ul_download(url, dir=paths.WEIGHTS)
    except Exception as error:  # 网络/代理/404 都落到这里，给出可操作提示
        print(f"[失败] {name}：{error}")
        print("       可手动下载后放入 vision_yolo_train/weights/ ："
              f"https://github.com/ultralytics/assets/releases 里的 {name}")
        return None

    path = find_weight(name)
    if path is None:
        print(f"[失败] {name}：下载调用未报错，但 vision_yolo_train/weights/ 下没有找到该文件")
        return None
    if path != paths.WEIGHTS / name:  # 落错位置就搬回 vision_yolo_train/weights/
        target = paths.WEIGHTS / name
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
    parser.add_argument("--force", action="store_true", help="已存在也重新下载")
    args = parser.parse_args()

    import ultralytics
    print(f"[信息] ultralytics {ultralytics.__version__} / python {sys.version.split()[0]}")
    verify_hand_keypoint_support()

    names = list(args.models) if args.models else list(DEFAULT_MODELS)

    ok, failed = [], []
    for name in names:
        path = download(name, force=args.force)
        (ok if path else failed).append(name)

    print(f"\n成功 {len(ok)}：{', '.join(ok) if ok else '无'}")
    if failed:
        print(f"失败 {len(failed)}：{', '.join(failed)}")
    print(f"权重目录：{paths.WEIGHTS}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
