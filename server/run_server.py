"""服务器端入口 —— 模型推理 + HTTP 服务 + 网页监控台。

    python server/run_server.py                                  # 监听 0.0.0.0:8080
    python server/run_server.py --port 9000 --imgsz 960          # 换端口/换推理尺寸
    python server/run_server.py --token 你的口令                  # 简单口令，防误连

启动后会打印真实可访问的网址（含本机局域网 IP），树莓派照那个地址配 --server-url。
"""

from __future__ import annotations

import sys
from pathlib import Path

# 让 `python server/run_server.py` 这种直接跑的方式也能 import 到 server 包
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from server.service import main  # noqa: E402  必须放在 sys.path 调整之后

if __name__ == "__main__":
    raise SystemExit(main())
