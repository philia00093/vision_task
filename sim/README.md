# sim/ — 发包模拟（两端都能伪装）

模型还没训出来、或者手上没有树莓派的时候，用这两个脚本把**协议链路**先跑通。
它们只依赖 `protocol.py` 和标准库，**不需要 opencv、不需要 torch、不需要权重**。

| 文件 | 伪装成 | 干什么 |
|---|---|---|
| `mock_server.py` | 电脑端识别服务 | 不加载模型，对每一帧回**伪造的检测框**（3 个动框 + 手部框 + 步数/布尔 + 耗时） |
| `mock_edge.py` | 树莓派 | 不用摄像头，自己造图（或读一张图）按二进制帧发给服务端，打印回包 |

## 1. 假服务 → 树莓派（没有模型时先看效果）

```bash
# 电脑上
python3 sim/mock_server.py --host 0.0.0.0 --port 8080

# 树莓派上（和连真服务完全一样的命令）
python3 edge/run_edge.py --infer-backend http \
    --server-url http://<电脑IP>:8080 --wire binary --fb /dev/fb0
```

屏幕上会出现：3 个沿圆周运动的框（base/support/compass）、每 4 秒露一次洋红手部框、
状态栏 `srv stepN b101 12ms`（步数 1~4 每 3 秒循环）。

调试开关：

```bash
python3 sim/mock_server.py --boxes 4 --no-hand          # 4 个框、不发手
python3 sim/mock_server.py --jitter-ms 80                # 每次推理慢 80ms
python3 sim/mock_server.py --drop-ratio 0.3              # 30% 的包直接不回（模拟丢包）
```

## 2. 假树莓派 → 真服务（没有树莓派时测服务端）

```bash
# 电脑上起真服务
python3 server/run_server.py --host 0.0.0.0 --port 8080

# 另一个终端：伪造树莓派发包
python3 sim/mock_edge.py --server-url http://127.0.0.1:8080 --frames 10 --fps 2
python3 sim/mock_edge.py --corrupt-every 3      # 每 3 帧弄坏一次 CRC，验证服务端拒收
python3 sim/mock_edge.py --wire json            # 走老的 JSON 路
python3 sim/mock_edge.py --image 某张照片.png    # 用真图当帧（纯读字节，不需要 opencv）
```

## 3. 自检

```bash
python3 sim/mock_server.py --self-test   # 起临时实例，发一帧验证往返 + 坏 CRC 拒收
python3 protocol.py                      # 协议本身的往返/边界/拆包自检
```

## 注意

- `mock_server.py` **不检查** `X-Token`（真服务才有 `--token`），只用于本地联调。
- 两个脚本都只实现 HTTP + 二进制帧这一条路；UDP/TCP 裸流没做（`protocol.py` 里的
  `split_frames` 已经具备拆包能力，需要时可以自己接）。
