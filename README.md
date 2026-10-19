# vision_task — 视觉识别（末端 + 服务器端）

相机画面里识别**物料区**（右侧，红框）与**装配区**（左侧，蓝框），
按物料区是否被手遮住显示工位状态（**取料中** / **装配中**）；
用训练好的 YOLO 检测**零件**（base / support / compass / top）并跟踪它们；
再用**基础视觉**认装配**阶段**（把装配区裁剪图旋转 1° 遍历一圈，跟几张参考图比相似度）。

代码按职责拆成三个包 + 一个训练工程，一个文件只干一件事，方便逐个阅读：

```
vision_task/
├─ edge/              ★ 末端（树莓派）：相机、料盘判定、跟踪、阶段识别、显示、采集
│                       入口：edge/run_edge.py
├─ server/            ★ 服务器端（电脑）：模型推理 + HTTP 服务 + 网页监控台
│                       入口：server/run_server.py
├─ test/              ★ 无通讯版（一台机器跑完，自己加逻辑的沙盒）
│                       入口：test/run_test.py
├─ protocol.py          二进制帧格式（上行图像 / 下行 160 字节结果帧），两侧共用这一份
├─ vision_yolo_train/   训练工程（数据集、标注转换、训练、体检、阶段参考图）—— 独立的第三块
├─ Dockerfile           树莓派镜像（arm64，装 opencv/numpy/PIL，不装 torch）
├─ docker-compose.pi.yml 树莓派一条命令起末端（设备/网络/卷都配好）→ 见第 9 节
├─ .dockerignore        把 800MB 训练图 / 1.6GB 数据集挡在镜像外
├─ annotate_parts.bat   拉起 X-AnyLabeling 标注（训练流程用）
├─ train_parts.bat      拉起训练（训练流程用）
└─ output/               运行输出（已 gitignore）
```

---

## 1. 两块怎么配合

```
        ┌──────────────────── edge/（树莓派）────────────────────┐
相机 ──▶│ 取右目 ─┬─ 料盘判定(OpenCV) ──▶ 装配区/物料区 + 工位状态 │
        │         ├─ 零件检测 ──▶ 框 + 卡尔曼跟踪 ─────────────┤──▶ 绘制/显示/采集
        │         └─ 阶段识别(OpenCV) ──▶ step_N + 相似度/摆角 ──┘
        │                        ▲                              │
        └────────────────────────┼──────────────────────────────┘
                                 │ 上行：一帧 JPEG（~100KB）
                                 │ 下行：只回框 JSON（~300B）
        ┌────────────────────────┴──────────────────────┐
        │               server/（电脑）                 │
        │  推理(GPU) → 结果 + 最近一帧 → 网页监控台      │
        └───────────────────────────────────────────────┘
```

三条路（料盘 / 零件 / 阶段）各自节流、各画各的：料盘每 `--zone-interval` 帧、
零件每 `--yolo-interval` 帧、阶段每 `--stage-interval` 帧，互不等待。

三种推理后端（`--infer-backend`），末端代码完全一样，只换这一层：

| 后端 | 在哪算 | 需要什么 |
|---|---|---|
| `ultralytics` | 本机（开发机） | torch + ultralytics |
| `onnx` | 本机（树莓派裸跑） | 只要 opencv + 一个 .onnx |
| `http` | **电脑** | 电脑跑 `server/run_server.py`，末端什么都不用装 |

下行永远只有框（几百字节），没有任何视频流。

---

## 2. 推荐阅读顺序（从底往上，逐个文件）

先读 `edge/`，按依赖顺序读，每个文件都能独立看懂：

| # | 文件 | 行数 | 一句话职责 |
|---|---|---|---|
| 1 | `edge/constants.py` | 134 | **所有可调常量**：料盘阈值、遮挡判据、卡尔曼参数、网络参数 |
| 2 | `edge/project.py` | 58 | 项目根目录怎么定位、目录创建 |
| 3 | `edge/imaging.py` | 100 | 帧/框基础：拆双目、取右目、缩放、掩膜→框、存图 |
| 4 | `edge/camera.py` | 78 | 开相机、协商 MJPG/分辨率、实测帧率、探测模式 |
| 5 | `edge/drawing.py` | 174 | 所有画面叠加：框、轨迹、中文标签、状态栏、阶段标签 |
| 6 | `edge/zones.py` | 146 | **料盘图像算法**：暗色掩膜、两个盘的形状判据、粘连切分 |
| 7 | `edge/zone_tracker.py` | 287 | **区域状态机**：锁定两个料盘、判遮挡、取料中/装配中、取装配区窗口 |
| 8 | `edge/tracking.py` | 142 | 卡尔曼滤波（只跟踪中心点） |
| 9 | `edge/detection.py` | 288 | **三种推理后端**：本机 torch / 本机 ONNX / 远程 HTTP |
| 10 | `edge/parts.py` | 238 | 检测 + 跟踪的整合：关联、补帧、**一个零件一个框** |
| 11 | `edge/capture.py` | 186 | 采集按钮（存训练图，默认存干净帧） |
| 12 | `edge/stage_match.py` | 267 | **阶段识别（基础视觉）**：规格化 → 旋转 1° 遍历 → 取近似最大相似度 |
| 13 | `edge/args.py` | 188 | 命令行参数全部在这里（`--help` 的内容） |
| 14 | `edge/app.py` | 495 | **主循环**：把上面所有工具按节奏串起来（料盘/零件/阶段三路并行） |
| 15 | `edge/run_edge.py` | 23 | 入口（调 `edge/app.py` 的 main） |

再读 `server/`（只有 4 个文件）：

| # | 文件 | 行数 | 一句话职责 |
|---|---|---|---|
| 16 | `server/model.py` | 64 | 加载模型、跑推理、留最近一帧/结果 |
| 17 | `server/web.py` | 197 | 网页监控台（HTML+JS 单文件，无外部依赖） |
| 18 | `server/service.py` | 186 | HTTP 路由：`/`、`/health`、`/state`、`/frame.jpg`、`POST /detect` |
| 19 | `server/run_server.py` | 21 | 入口（打印真实可访问网址） |

最后是 `test/`（无通讯版，读起来最快）：

| # | 文件 | 行数 | 一句话职责 |
|---|---|---|---|
| 20 | `test/run_test.py` | 460 | 主循环（去掉全部通讯参数，阶段识别两条路可选） |
| 21 | `test/step_recognizer.py` | 125 | **规则版阶段识别**（按装配区零件组合推断），改规则就改这个文件 |

最后是根目录的 `protocol.py`（529 行），**两侧共用的通信帧格式**：
上行一帧图像、下行 160 字节结果帧（4 组四点 + 预留手部框 + 步数 + 3 个布尔 + CRC32），
文件头就是完整的字节表，`python protocol.py` 会跑一遍往返自检。现在跑的通路还是 HTTP+JSON，
这个模块是给"换成二进制"准备的（接法见第 8 节）。

训练那套（`vision_yolo_train/`）跟上面两块**没有代码耦合**，只有两处接口：
末端的 `--yolo-weights` 指向它产出的 `runs/*/weights/best.pt`；
`server/` 的默认权重也用同一个文件。训练细节看 `vision_yolo_train/README.md`。

---

## 3. 怎么跑

**A. 一台电脑上全跑**（开发调试最省事）

```bash
python edge/run_edge.py                                  # 实时预览 + 本机推理
python edge/run_edge.py --source 某张图.png --snapshot --no-capture   # 离线跑一张图
```

**B. 树莓派 + 电脑（正式部署）**

```powershell
# 电脑：起识别服务 + 网页监控台（启动会打印真实网址）
D:\anaconda3\envs\yolo26\python.exe server\run_server.py --host 0.0.0.0 --port 8080 --imgsz 960

# 浏览器：看相机画面 + 实时框
#   http://<电脑局域网IP>:8080/

# 树莓派：只采集 + 画框 + 推帧
python edge/run_edge.py --infer-backend http --server-url http://<电脑局域网IP>:8080 \
       --yolo-interval 10 --no-window
```

**C. 树莓派裸跑**（不联网、不用电脑）

```bash
python edge/run_edge.py --infer-backend onnx --yolo-weights best.onnx --yolo-imgsz 640 --detect-width 0
```

**D. 阶段识别**（默认就开着，参考图在 `vision_yolo_train/images/steps/`）

```bash
python edge/run_edge.py                       # 阶段标签贴「装配区」框右上角
python edge/run_edge.py --no-stage            # 关掉阶段识别（只留料盘+零件）
python edge/run_edge.py --stage-region 0 477 228 396   # 固定裁剪窗口，不必等区域锁定
python edge/run_edge.py --angle-step 2 --stage-interval 20   # Pi 上省算力
python test/run_test.py --stage-mode both     # 两条路一起跑：模板 + 零件组合
```

热键：`S` 保存当前帧，`Q` / `ESC` 退出，`1~5` 采集训练数据。

---

## 4. 几个已经踩过的坑（读代码时会看到对应注释）

| 现象 | 原因 / 处理 | 在哪 |
|---|---|---|
| 区域框横跨两个料盘、左右区乱跳 | 两盘挨着时暗色掩膜连成一块，必须先"找缝切开" | `edge/zones.py` |
| 区域框每帧都在动 | 锁定前框会跟着轮廓走；稳定若干帧后**锁定并落盘** | `edge/zone_tracker.py` |
| 一直显示"装配中"不显示"取料" | 单看暗色可见量会漏判（手只盖一角时只掉到 0.8），加了"亮占比增量"第二路 | `edge/zone_tracker.py` |
| 零件框"预测全部跑飞" | 把宽高也放进卡尔曼状态 → 框每帧线性长大；尺寸不能参与滤波 | `edge/tracking.py` |
| 同一零件出现两个框 | 丢检的旧轨迹 + 新检出的新轨迹；现在按重叠度合并 | `edge/parts.py` |
| 采集的图带着误识别的框 | 默认存**画之前**的干净帧，要带叠加用 `--capture-overlay` | `edge/capture.py` |
| 阶段标签和区域名叠在一起 | 料盘框只有一两百像素宽，两个标签同一行必然撞 → 阶段标签**再抬一行** | `edge/drawing.py` |
| 阶段相似度只有 0.5 上下、来回跳 | 每阶段只有 1 张参考图，换个摆向就认不像 → 每阶段放 2~3 张样本（见下） | `edge/stage_match.py` |
| 服务首帧要 1.3 秒（客户端超时） | CUDA 冷启动；服务端启动时先预热一遍 | `server/model.py` |

---

## 5. 数据采集按钮 & 阶段参考图

实时预览时画面底部 5 个按钮，点按钮或按**数字键**就把画面存进
`vision_yolo_train/images/parts/`：

| 按钮 | 键 | 文件 | 类别 |
|---|---|---|---|
| base | `1` | `base_N.png` | 底部 |
| support | `2` | `support_N.png` | 支撑柱 |
| compass | `3` | `compass_N.png` | 罗盘 |
| top | `4` | `top_N.png` | 盘/顶盖 |
| 全部 | `5` | `test_N.png` | 通用（可能含多件） |

编号自动接续（扫描已有 `<前缀>_<数字>`，从最大 +1 开始，不覆盖、不回收缺号）。
关掉按钮：`--no-capture`；换目录：`--capture-root <目录>`。

**阶段参考图**放在 `vision_yolo_train/images/steps/`，两种放法都认（`--stage-dir` 可换目录）：

```
steps/step1.png  step_2.png  step_3.png  step_4.png      # 平铺：文件名 = 阶段名（每阶段 1 张）
steps/step_3/sample_1.png  steps/step_3/sample_2.png     # 分目录：目录名 = 阶段名，可放多张
```

同一阶段**建议 2~3 张**（不同摆向 / 不同光照）。只有 1 张时它能认出"正对着拍"的那一种摆法，
零件转个角度相似度就会掉下来；样本多了，旋转遍历才有东西可对。
参考图是**整幅画面**（不是裁出来的料盘），程序会按当前装配区窗口去裁，所以拍的时候
镜头位置别动。

---

## 6. 环境

| 用途 | 需要 |
|---|---|
| `edge/` 的相机/料盘/绘制/采集 | Python 3 + numpy + opencv + PIL（画中文） |
| `edge/` 的 `--infer-backend ultralytics` | 再加 torch + ultralytics（本机 `yolo26` 环境） |
| `edge/` 的 `--infer-backend onnx` | 只要 opencv（+ 导出的 .onnx） |
| `edge/` 的 `--infer-backend http` | **什么都不用装**（纯标准库） |
| `server/` | torch + ultralytics（本机 `yolo26` 环境） |

本机 `yolo26` 环境：Python 3.11 / torch 2.8.0+cu128 / ultralytics 8.4.157。
两个 `.bat` 已指向 `D:\anaconda3\envs\yolo26\python.exe`。

---

## 7. 注意

- **入口只有三个**：`edge/run_edge.py`（末端）、`server/run_server.py`（服务器）、
  `test/run_test.py`（无通讯版沙盒）。老的 `test_run.py` / `infer_server.py` 已经删掉了。
- 仓库根目录如果存在 `zones.json`，启动时会**直接读它**当区域框；怀疑框不对就删掉
  （会重新自动识别），或直接 `--zones "x1,y1,x2,y2;x1,y1,x2,y2"` 手工给（前=左=装配区）。
- 阶段识别和零件跟踪是**两条独立的路**：阶段只看装配区那块裁剪图，零件框乱了/关掉了都不影响它；
  反过来阶段识别也不碰零件框。想只要一个就 `--no-stage` 或 `--no-yolo`。
- 阶段识别默认每 10 帧跑一次（`--stage-interval`）：一次匹配约 20 ms（开发机），
  树莓派上会慢 5~10 倍，嫌卡就 `--angle-step 2`（180 个角度）或 `--stage-size 64`。

---

## 8. 帧格式（树莓派 ⇄ 电脑，`protocol.py`）

**两端都接好了**，默认就走二进制帧（`--wire binary`）；老的 JSON 还在，`--wire json` 一行切回去。
两条路共用同一个路由：服务端看请求体头两个字节，`"VP"` → 二进制帧，`FF D8` → 裸 JPEG。
完整字节表在 `protocol.py` 文件头，这里只列要点：

| 帧 | 长度 | 内容 |
|---|---|---|
| 通用帧头 | 16 B | magic `"VP"` + 版本 + 类型 + 标志 + 保留 + seq(u32) + 载荷长(u32) |
| 上行 图像帧 | 16+12+JPEG+4 | 宽高(u16×2) + 格式(1=JPEG) + 质量 + 时间戳(u32 ms) + JPEG 原始字节 |
| 下行 结果帧 | **恒 160 B** | `ref_seq` + 4 个检测槽 + 手部槽（预留）+ 步数 + 3 个布尔 + 16 B 尾部预留 |
| CRC | 4 B | CRC32，覆盖 magic 到载荷结尾（不含自己） |
| 一个检测槽 | 20 B | 类别 + valid + 置信度(千分比 u16) + **4 个点**×(x u16,y u16) |

几个刻意的选择：

- **4 个点而不是 2 个**：现在的轴对齐框按顺时针写 `左上→右上→右下→左下`；将来换旋转框/轮廓
  帧格式不用改。
- **定长 160 字节**：不用 TLV，一次 `struct.unpack` 解析完，树莓派上几乎不花算力；
  `det_count` 说明真正用了几组。
- **手部框先占 20 字节**（`valid=0` = 没检测到手），格式和检测槽完全一样，解析代码共用。
- **尾 16 字节 + 帧头保留位**：加字段先吃预留，不够才升 `VERSION`，老解析器会明确拒收而不是读出乱码。
- **`ref_seq` 回显上行序号**：结果乱序回来时，比上次小的直接丢，旧结果不会覆盖新结果。
- **`cost_us`**：把原来空着的 4 字节定成"服务端推理耗时"，树莓派状态栏直接显示，不然算力信息丢了。

### 两端各在哪（想改就从这些地方下手）

| 位置 | 做什么 |
|---|---|
| `protocol.py` | 帧头/图像帧/结果帧的编解码、CRC、`split_frames` 拆包、`describe()` 摘要 |
| `edge/detection.py:186` `_post_binary` | 树莓派**上行**：`encode_image_frame` → POST → `parse_result_frame`，把结果转回内部的 `[x1,y1,x2,y2,cls,conf]` |
| `edge/detection.py:54` | `wire_format` 开关（`binary`/`json`），到 `edge/args.py:42` 的 `--wire` |
| `edge/detection.py:110/122` | `runner.remote.step` / `.flags` / `.hand`：下行帧里那几个预留字段 |
| `edge/app.py:415` | 状态栏显示 `srv step3 b101`（电脑端报的步数 + 3 个布尔） |
| `edge/app.py:385` | 手部框：服务端发 `valid=1` 就自动画成洋红色框 |
| `server/service.py:120` | 电脑端**判帧**：`wire.looks_like_frame(raw)` → 走二进制分支 |
| `server/service.py:139` `_handle_binary` | 电脑端收帧 → 推理 → `encode_result_frame` 回 160 字节 |
| `server/model.py:27` | `STATE["step"/"flags"/"hand"]`：电脑端说了算的三个字段的来源 |
| `server/service.py:174` `POST /stage` | 联调用：`curl -X POST .../stage -d '{"step":3,"flags":[1,0,1]}'` 把预留字段填上 |

```python
import protocol as P

# 树莓派（edge/detection.py 里就是这么写的）
frame = P.encode_image_frame(jpeg, 1280, 960, seq=seq, quality=80, stamp_ms=t)
result = P.parse_result_frame(响应体)        # detections/hand/step/flags/cost_ms

# 电脑（server/service.py 里就是这么写的）
image = P.parse_image_frame(请求体)
boxes = infer(cv2.imdecode(image["jpeg"], cv2.IMREAD_COLOR), ...)   # server/model.py
回包 = P.encode_result_frame(image["seq"], boxes, step=3, flags=(1, 0, 1),
                             hand={"conf": 0.7, "box": (x1, y1, x2, y2)}, cost_us=12345)
```

`python protocol.py` 跑自检（往返、边界、坏 CRC 拒收、TCP 拆包）。实测两端联调：
`edge/run_edge.py --wire binary` 对 `server/run_server.py`，一帧一个来回，下行恒定 160 字节，
状态栏出现 `srv step3 b101`、手部框画在洋红框里；`--wire json` 走老路，结果一样只是没有那三个字段。

---

## 9. 树莓派部署（6 种方式 + Docker 全流程）

先说清概念：**SD 卡里烧的永远是"系统"**（.img），程序是部署进去的。所以"烧录程序"按你的真实需求分成六条路，
先挑一条再往下看对应小节：

| # | 方式 | 卡里烧什么 | 程序怎么进去 | 首次耗时 | 适合 |
|---|---|---|---|---|---|
| A | 裸拷贝直接跑 | Raspberry Pi OS 64 位 | `scp`/`git clone` + `apt install python3-opencv` | 10 分钟 | 先跑通、调试期（**最快验证**） |
| B | 裸拷贝 + systemd | 同上 | A + 一个 service 文件 | 20 分钟 | 单台长期跑，不想用容器 |
| C | **Docker 在派上 build**（推荐） | 同上（另装 Docker） | `docker compose up -d --build` | 30 分钟 | 绝大多数情况 |
| D | Docker 离线镜像 | 同上（另装 Docker） | 别处 build → `docker save/scp/load` | 30 分钟 + 构建机 | 派不联网 / 要装多台 |
| E | Docker + 局域网 registry | 同上（另装 Docker） | `push` / `pull` | 同 D | 多台频繁更新 |
| F | 母卡克隆 | 已配好的整卡 `.img` | `dd` 备份母卡再写别的卡 | 20 分钟/张 → 5 分钟/张 | 同型号批量 5 张以上 |
| G | 自制系统镜像 | 自己生成的 `.img` | pi-gen 构建时把 Docker/代码/服务全预装 | 1 小时+ | 出厂级批量、完全离线 |

> 单台设备：直接 C。派不联网：D。要装很多张一样的卡：F（比 G 省一个数量级的力气）。
> **只有 G 才算"把程序烧进卡里"**，其余都是"卡里烧系统、程序部署上去"。

各方式的具体命令见 9.1~9.10；A/B（不用 Docker 的裸跑）见第 9.11 节。

### 9.1 烧系统（在电脑上做）

1. 装 **Raspberry Pi Imager**，选 `Raspberry Pi OS (64-bit)`（Bookworm，**必须 64 位**：
   arm64 的 opencv wheel 只有 64 位系统能装），选你的 SD 卡。
2. 点齿轮/编辑设置：设主机名、用户名密码、Wi-Fi、**打开 SSH**（用密码或公钥都行）。
3. 写入，插卡上电，`ssh 用户名@树莓派IP` 能进去就成了。
4. 进系统先更新一次：
   ```bash
   sudo apt update && sudo apt full-upgrade -y
   sudo reboot
   ```

### 9.2 装 Docker（在树莓派上做）

> 你的 Windows 电脑**不用装 Docker**：`server/` 用原生 Python + CUDA 跑，Docker 里反而用不了显卡。
> 只有树莓派要装。前提是烧的 **64 位**系统（`uname -m` 应该是 `aarch64`，不是 `armv7l`）。
>
> 注意：`curl -fsSL https://get.docker.com | sudo sh` 是**Linux 的命令**，在 Windows 的 cmd/PowerShell
> 里跑不通（没有 `sh`，管道是文本流；Windows 的 `curl` 在 PowerShell 5.1 里还是 `Invoke-WebRequest` 的别名）。
> 非要在 Windows 上装 Docker 只有一条正路：**Docker Desktop + WSL2**（见下面最后一段）。

**方案 A：一键脚本（最省事，arm64/armhf 都支持）**

```bash
curl -fsSL https://get.docker.com | sudo sh
```

脚本会自动配好 Docker 官方 apt 仓库、装 `docker-ce / containerd / buildx / compose 插件`，
并 `systemctl enable --now docker`（开机自启）。

**方案 B：官方 apt 仓库手工装（想自己控制版本、以后好升级）**

```bash
# 1) 装依赖 + 放 GPG 公钥
sudo apt update
sudo apt install -y ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/debian/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

# 2) 加源（树莓派 64 位系统就是 Debian bookworm，VERSION_CODENAME 会自动取 bookworm）
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] \
https://download.docker.com/linux/debian $(. /etc/os-release && echo $VERSION_CODENAME) stable" \
| sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# 3) 装
sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io \
                    docker-buildx-plugin docker-compose-plugin
```

**装完必须做的两步**

```bash
# 1) 免 sudo：把自己加进 docker 组（之后重新登录 / 或执行 newgrp docker 立刻生效）
sudo usermod -aG docker $USER
newgrp docker

# 2) 验证：能打出版本号 + 能跑容器，才算装好
docker --version
docker compose version          # 有这一条就说明 compose 插件在（不用再单独 apt 装）
docker run --rm hello-world     # 拉镜像 + 跑容器，整条链路打通
```

> `sudo apt install docker-compose-plugin` 是**错的**：Docker 的 apt 源自带这个包，
> Debian 官方源里没有。用方案 A/B 装完就有了；`docker-compose`（老的一代 Python 版）也不要装。

**国内网络：拉镜像慢/超时怎么办**

```bash
# 阿里云个人镜像加速器（控制台 → 容器镜像服务 → 镜像加速器，会给你一个专属地址，最稳）
sudo mkdir -p /etc/docker
sudo tee /etc/docker/daemon.json > /dev/null <<'EOF'
{
  "registry-mirrors": ["https://<你的ID>.mirror.aliyuncs.com"]
}
EOF
sudo systemctl restart docker
docker info | grep -A2 "Registry Mirrors"
```

`apt` 慢就换国内源（清华/中科大，`/etc/apt/sources.list.d/raspi.sources` 或 `sources.list`）；
我们 Dockerfile 里的 `pip install` 也可以加 `-i https://pypi.tuna.tsinghua.edu.cn/simple`，见第 9.8 节。

**装不上时的自查**

| 现象 | 原因 / 处理 |
|---|---|
| `systemctl status docker` 里全是 `failed`，日志提 `iptables` | 内核/`iptables` 不是 nft 版：`sudo update-alternatives --set iptables /usr/sbin/iptables-legacy` 后 `sudo systemctl restart docker` |
| `docker run` 报 `permission denied ... /var/run/docker.sock` | 没进 docker 组：`sudo usermod -aG docker $USER` 再**重新登录**（`newgrp docker` 只在当前 shell 生效） |
| `apt-get update` 报 `NO_PUBKEY` | 公钥没放对：重做方案 B 的第 1 步，确认 `docker.asc` 存在且 `chmod a+r` |
| `docker compose` 提示 `is not a docker command` | 只装了 `docker-ce` 没装 `docker-compose-plugin`；方案 B 补上它，或重跑方案 A |
| 装了 `docker.io`（Debian 自带的旧版）想换官方版 | `sudo apt remove docker docker-engine docker.io containerd runc` 然后再走方案 A/B |

> **Windows 上非要装 Docker**（本项目不需要，仅备查）：先管理员 PowerShell 执行 `wsl --install`
> （装 WSL2 + Ubuntu，需重启、需 BIOS 开虚拟化），再 `winget install -e --id Docker.DockerDesktop`，
> 装完在 Docker Desktop 设置里打开 *Resources → WSL integration*，最后 `docker run --rm hello-world` 验证。
> Docker Desktop 里跑 CUDA 镜像要 Windows 端装好 NVIDIA 驱动并加 `--gpus all`，镜像好几个 GB，
> 比直接用现有 `yolo26` 的 conda 环境麻烦得多 —— 所以 `server/` 就在 Windows 原生跑。

### 9.3 把代码放到树莓派上

```bash
# A. 有 git 仓库
git clone <你的仓库地址> ~/vision_task && cd ~/vision_task

# B. 没有仓库：在电脑上 scp 整个目录（会自动跳过没权限/不需要的）
scp -r D:\wang_robot\3_1\version_task\vision_task pi@树莓派IP:~/vision_task
```

`images/parts`（800MB）和 `datasets`（1.6GB）**不用传**，树莓派上只跑末端；
但 `vision_yolo_train/images/steps`（阶段参考图，4.7MB）要留着。

### 9.4 先确认摄像头认得

```bash
sudo apt install -y v4l-utils
v4l2-ctl --list-devices                    # 应该看到你的 USB 摄像头 → /dev/video0
v4l2-ctl -d /dev/video0 --list-formats-ext | head -30    # 看支持的最大分辨率/MJPG
```

不是 `/dev/video0` 就改 `docker-compose.pi.yml` 里的 `devices:`。

### 9.5 构建 + 跑

```bash
# 1. 改电脑地址：docker-compose.pi.yml 里 --server-url=http://192.168.1.10:8080
#    先在电脑上起服务：python server/run_server.py --host 0.0.0.0 --port 8080
nano docker-compose.pi.yml

# 2. 构建镜像（arm64 本机编译，第一次 5~10 分钟；之后改代码不用重建）
docker compose -f docker-compose.pi.yml build

# 3. 起（-d 后台）
docker compose -f docker-compose.pi.yml up -d

# 4. 看日志（启动横幅、"帧格式 binary"、连接结果、识别统计都在这里）
docker compose -f docker-compose.pi.yml logs -f
```

电脑浏览器打开 `http://电脑IP:8080/` 就能看到树莓派画面 + 实时框。

### 9.6 日常维护

```bash
docker compose -f docker-compose.pi.yml restart      # 改了代码：restart 就生效（代码是挂进去的）
docker compose -f docker-compose.pi.yml up -d --build   # 改了 Dockerfile / 依赖才需要重建
docker compose -f docker-compose.pi.yml down         # 停掉
docker compose -f docker-compose.pi.yml logs --tail 50
docker exec -it vision-edge python -c "import cv2;print(cv2.__version__)"   # 进容器看一眼
```

开机自启：`restart: unless-stopped` + Docker 服务本身开机启动（默认就是），
树莓派断电重启后会自动把末端拉起来。

### 9.7 想在树莓派桌面上看预览窗口

```bash
# 宿主机（桌面登录的同一个用户）
xhost +local:
# compose 里已经有 /tmp/.X11-unix 挂载 + DISPLAY=:0；command 里去掉 --no-window
# 还需要带 Qt 的 opencv：重建镜像时换掉那个 build-arg
docker compose -f docker-compose.pi.yml build --build-arg OPENCV_PKG=opencv-python
```

无显示器（大多数情况）就用默认的 `--no-window`：树莓派只负责采集 + 推帧，
要看画面看电脑网页，省一份 X11 的折腾。

### 9.8 踩坑对照

| 现象 | 原因 / 处理 |
|---|---|
| `error: no such device /dev/video0` | 摄像头没插好，或节点不是 video0；`v4l2-ctl --list-devices` 查；实在不行临时 `privileged: true` |
| 容器起来就退出，日志 `读取画面失败` | 设备节点传进去了但容器没权限；把用户加入 `video` 组，或 `privileged: true` 兜底 |
| `ImportError: libGL.so.1` | 用了非 headless 的 opencv 但没装 libgl1；用默认的 `opencv-python-headless` |
| 画面全黑 / 只有一半 | 双目 2560x960 的源要 `--layout auto`（默认），单目源要 `--layout none` |
| 连不上电脑 | 先 `ping 电脑IP`；电脑防火墙放行 8080；`--server-url` 不能写 `127.0.0.1`（那指的是树莓派自己） |
| 状态栏 `net 300ms` 且框很飘 | 网络/带宽不够：`--jpeg-quality 60 --yolo-interval 15`，或把 5GHz Wi-Fi 换有线 |
| 帧格式对不上（状态栏没有 `srv step`） | 电脑端还是旧代码（只认 JSON）：两端一起更新；或先 `--wire json` 顶上 |
| 构建卡在下载 wheel | 树莓派走国内镜像：`RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple ...` |

### 9.9 在 Ubuntu / x86 电脑上构建好再给树莓派

先看这台 Ubuntu 是什么架构：`uname -m`

- `aarch64` → **本身就是 arm64**（树莓派上装的 Ubuntu、或 ARM 服务器）：直接 `docker build`，
  跟 9.5 一样，不用下面的 qemu。
- `x86_64` → 是普通电脑：要模拟 arm64，慢（10~30 分钟），但一次构建能反复用。

```bash
# 0) Ubuntu 上装 Docker（有 docker 就跳过）
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER && newgrp docker

# 1) x86 上让 Docker 能执行 arm64 指令（装一次，重启后仍在）
sudo docker run --privileged --rm tonistiigi/binfmt --install all

# 2) 建个专用 builder 并交叉构建
docker buildx create --driver docker-container --name pi-builder --use
docker buildx build --platform linux/arm64 -t vision-edge:pi --load .

# 3) 顺手验证这个 arm64 镜像真的能跑（x86 上靠 qemu 执行，慢但能看出 import 有没有问题）
docker run --rm --platform linux/arm64 vision-edge:pi --help | head -5

# 4) 导出 + 拷到树莓派 + 导入
docker save vision-edge:pi -o vision-edge.tar
scp vision-edge.tar pi@树莓派IP:~/
ssh pi@树莓派IP 'docker load -i ~/vision-edge.tar'
```

注意 `--load` 不能省：不加的话多平台构建结果只留在 build cache 里，`docker save` 会找不到镜像。
`scp`/`ssh` 也可以换成局域网 registry（`docker run -d -p 5000:5000 --name reg registry:2`，
然后 `push localhost:5000/vision-edge:pi` / 派上 `pull`）。

比在树莓派上直接 `build` 慢得多，第一次最容易在 qemu 上卡住；正常情况下就在树莓派上编。

### 9.10 「装好 Docker / 构建好镜像」再交给树莓派

方向先摆正：**SD 卡里烧的是操作系统镜像，Docker 镜像不能烧进卡里**。想省掉树莓派上的构建，
是把**容器镜像**拷过去，不是重烧系统。三种做法按省事程度排：

| 做法 | 怎么做 | 适合 |
|---|---|---|
| **在树莓派上 build**（推荐） | `docker compose -f docker-compose.pi.yml build` | 树莓派能联网、就一台设备 —— 一次 5~10 分钟，之后改代码只 restart |
| **别处构建 → 离线导入** | 在 arm64 机器上 build（或 x86 上 `buildx --platform linux/arm64`），`docker save -o vision-edge.tar vision-edge:pi`，`scp` 到树莓派后 `docker load -i vision-edge.tar` | 树莓派不联网 / 要一次装多台 |
| **做自定义系统镜像** | pi-gen / rpi-image-gen 把 Docker 和镜像预装进 .img，再烧卡 | 批量出厂、几十台起 —— 工程量最大，本项目用不到 |

离线导入的完整命令：

```bash
# 构建机（arm64 原生最快；x86 用上面 9.9 的 buildx）
docker build -t vision-edge:pi .
docker save vision-edge:pi -o vision-edge.tar        # 约几十 MB（只有 opencv/numpy/PIL）
scp vision-edge.tar pi@树莓派IP:~/

# 树莓派
docker load -i vision-edge.tar
cd ~/vision_task && docker compose -f docker-compose.pi.yml up -d   # 镜像已存在，不会重新 build
```

**关于「先把系统换成 Ubuntu」**：技术上完全可以（Raspberry Pi Imager 里就直接能选 Ubuntu
Server 22.04/24.04 LTS 的 arm64 镜像），但**换系统不是装 Docker 的前提** ——
Raspberry Pi OS 64 位跑 `curl -fsSL https://get.docker.com | sudo sh` 一样一条命令装完。
而且我们的镜像是 `python:3.11-slim-bookworm`（Debian），**容器里跟宿主机是什么系统无关**，
Ubuntu 上照样构建、照样跑。真要换之前先想清楚这几点代价：

- Ubuntu on Pi 上 `rpi-eeprom`/bootloader 固件由 Raspberry Pi OS 独家管理，Ubuntu 里更新不了；
- 用的是 **USB 双目摄像头**（UVC）就没问题，`/dev/video0` 一样；但要是 CSI 排线摄像头
  （libcamera），Ubuntu 上支持明显差、坑多；
- Ubuntu Server 没桌面，想在树莓派本地看预览窗口还得自己装桌面/转发 X；
- 你现在这套 `docker-compose.pi.yml` / `devices: /dev/video0` / `v4l2-ctl` 步骤两边通用，
  唯一要补的是 Ubuntu 上先 `sudo apt install -y v4l-utils libgl1`（headless 用不到 libgl1）。

结论：**能换，但没收益**。想省事就在 Raspberry Pi OS 64 位上装 Docker；想免去树莓派上的构建，
就用上面的 `docker save / load` 离线导入，而不是重烧系统。

### 9.11 不用 Docker 的两条路（A / B，最快验证 + 长期自启）

**A. 裸拷贝直接跑**（连 Docker 都不装，10 分钟能出画面）

```bash
# 电脑上：只传要用的东西（训练图/数据集不用传）
scp -r vision_task pi@树莓派IP:~/

# 树莓派上：系统源里就有 opencv/numpy/PIL，一个 pip 都不用装
sudo apt update
sudo apt install -y python3-opencv python3-numpy python3-pil v4l-utils
v4l2-ctl --list-devices                       # 确认 /dev/video0
cd ~/vision_task

# 先抓一帧看通不通（不连电脑也能测相机/料盘/阶段）
python3 edge/run_edge.py --no-window --snapshot --no-yolo --save-dir ~/out

# 正式跑：算力在电脑上，二进制帧
python3 edge/run_edge.py --infer-backend http --server-url http://电脑IP:8080 \
        --wire binary --no-window
```

Debian bookworm 自带的是 `python3-opencv` 4.5.x，本项目用到的 `imencode/imdecode/cv2.dnn/warpAffine`
都有，够用。要更新版 opencv 才用 venv：

```bash
sudo apt install -y python3-venv python3-pip
python3 -m venv ~/venv
~/venv/bin/pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
    opencv-python-headless numpy pillow
~/venv/bin/python edge/run_edge.py --infer-backend http --server-url http://电脑IP:8080
```

**B. 加个 systemd 服务**（开机自启、崩了自动重启，比 Docker 更"轻"）

```bash
sudo tee /etc/systemd/system/vision-edge.service > /dev/null <<'EOF'
[Unit]
Description=vision edge (camera + zones + stage -> remote yolo)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/vision_task
ExecStart=/usr/bin/python3 edge/run_edge.py --infer-backend http \
          --server-url http://电脑IP:8080 --wire binary --no-window
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable --now vision-edge
systemctl status vision-edge --no-pager     # 起来就是 active (running)
journalctl -u vision-edge -f                # 看日志（等价于 docker logs -f）
```

改了代码不用重启服务：`sudo systemctl restart vision-edge`。

**F. 母卡克隆**（要装多张卡时最省事，比自制镜像简单得多）

```bash
# 在一张已经配好的"母卡"上（接读卡器时设备名可能不是 mmcblk0，用 lsblk 确认）
sudo dd if=/dev/mmcblk0 of=~/master.img bs=4M status=progress
# 写回其它卡（Linux）
sudo dd if=~/master.img of=/dev/sdX bs=4M status=progress
# Windows：Win32DiskImager 选 master.img 写入
```

注意：母卡镜像比目标卡大就写不进去 —— 批量就买同型号同容量的卡；
`dd` 拿到的是**整卡**（含系统、Docker、镜像、代码、zones.json），这才叫"把程序烧进卡里"。
