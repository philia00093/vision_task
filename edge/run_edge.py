"""末端（树莓派）入口 —— 相机 + 料盘判定 + 零件跟踪 + 显示/采集。

真正跑起来的命令：

    python edge/run_edge.py                                  # 本机推理（开发机）
    python edge/run_edge.py --infer-backend http \
           --server-url http://192.168.1.10:8080             # 算力放电脑（树莓派）

参数说明看 `python edge/run_edge.py --help`，全部定义在 edge/args.py。
"""

from __future__ import annotations

import sys
from pathlib import Path

# 让 `python edge/run_edge.py` 这种直接跑的方式也能 import 到 edge 包
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from edge.app import main  # noqa: E402  必须放在 sys.path 调整之后

if __name__ == "__main__":
    raise SystemExit(main())
