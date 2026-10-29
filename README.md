# vision_task — 装配工位视觉识别（树莓派末端 + 电脑推理）

树莓派拍画面、只做采集/绘制/跟踪；**推理全在电脑上**（GPU），结果按帧号回传；
屏幕上显示装配区/物料区、零件框、当前步骤、罗盘正反、支撑杆圈数。

```
        ┌──────────────── edge/（树莓派 4B）────────────────┐
相机 ──▶│ 取帧 ─┬─ 料盘判定(OpenCV) ──▶ 装配区/物料区 + 取料中/装配中 │
        │       ├─ 上传裁剪帧 ──────────┐                    │
        │       └─ 卡尔曼 + 帧↔框对齐 ──┘ 绘制/显示(/dev/fb0)  │
        └───────────────┬──────────────────────────────────┘
                        │ 上行：一帧 JPEG（约 30 KB，900×500 裁剪）
                        │ 下行：160 字节定长结果帧
        ┌───────────────┴──────────────────────────────────┐
        │              server/（电脑 + RTX 显卡）             │
        │  零件模型 / 阶段模型 / MediaPipe 手部 → 结果帧       │
        │  网页监控台 2000  ←  最近一帧 + 检测框 + 统计        │
        └──────────────────────────────────────────────────┘
```

## 目录

```
vision_task/
├─ edge/          树莓派末端：采集、跟踪、绘制、计圈、显示
├─ server/        电脑端：推理服务(8080)、监控台(2000)
├─ deploy/        启动/部署脚本（run_all.sh、deploy_pi.sh、systemd 单元）
├─ tools/         调试工具（compass_view.py 看罗盘正反、inspect_steps.py 看阶段样本）
├─ protocol.py    两端共用的二进制帧格式（自检：python protocol.py）
└─ vision_yolo_train/
   ├─ runs/wersion_parts/weights/best_v2.pt   零件模型 base/support/compass/top
   ├─ runs/steps_stage_best.pt                阶段模型 step_0..step_3/complete
   └─ weights/hand_landmarker.task            MediaPipe 手部 21 点
```

## 怎么起

```bash
bash deploy/run_all.sh           # 起识别服务(8080) + 监控台(2000) + 重启树莓派末端
bash deploy/run_all.sh --status  # 只看状态（会打印树莓派实时 fps 行）
bash deploy/run_all.sh --stop    # 全停
```

网页：

| 用途 | 电脑上 | 局域网 |
|---|---|---|
| 识别服务（状态 + 采集按钮） | http://127.0.0.1:8080 | http://<电脑IP>:8080 |
| 监控台（实时画面 + 框 + 曲线） | http://127.0.0.1:2000 | http://<电脑IP>:2000 |

树莓派上的末端参数在 `/usr/local/sbin/vision-edge-start.sh`，改完 `sudo systemctl restart vision-edge`。

## 关键机制

| 机制 | 做什么 | 在哪 |
|---|---|---|
| 帧↔框对齐 | 屏幕上放的是 22 帧前的画面（播放缓冲），框就取**那一帧自己的结果**（按上行 seq）；没结果时才用卡尔曼预测，且限制离上一帧实测位置 1.2 个框对角线以内 | `edge/detection.py` `history`、`edge/parts.py` `frame_detections()` |
| 独立发送进程 | JPEG 编码 + HTTP 收发都在子进程（共享内存传帧），主循环不再和它抢 GIL；识别帧 14 → 25 帧/秒 | `edge/sender_proc.py` |
| 红点偏心计圈 | 杆子上贴红点，点绕轴心的角度逐帧解缠累加 → 圈数；轴心取装配区里 base 框心 | `edge/rod_dot.py` |
| 罗盘正反 | 中心毂黑区占比（正 ≈0.42 / 反 ≈0.30），Otsu 自动阈值，只在中心 60% 裁剪上算 | `edge/defects.py` `compass_side()` |
| 阶段识别 | 电脑端阶段模型（step_0..3/complete）每 10 帧一次，结果走下行帧预留槽；零件组合规则兜底 | `server/service.py` `--stage-every`、`edge/stage_parts.py` |
| 装配区不追踪 | 进了装配区的零件不建轨迹、不画框（放进去就装好了） | `edge/parts.py` `exclude_rect` |
| 料盘状态机 | 暗色掩膜 + 形状判据锁两个盘；手遮物料区 → "装配中"，遮挡解除累计步数 | `edge/zones.py`、`edge/zone_tracker.py` |
| 手部 21 点 | 电脑端 MediaPipe 每帧检测，树莓派 5 Hz 轮询 `/hand` 取关键点，屏上只画点不画框 | `server/hand_mp.py`、`edge/hand.py` |

## 帧格式（`protocol.py`）

| 帧 | 长度 | 内容 |
|---|---|---|
| 通用帧头 | 16 B | magic `"VP"` + 版本 + 类型 + 标志 + 保留 + seq(u32) + 载荷长(u32) |
| 上行 图像帧 | 16+12+JPEG+4 | 宽高(u16×2) + 格式 + 质量 + 时间戳 + JPEG |
| 下行 结果帧 | **恒 160 B** | `ref_seq` + 4 个检测槽 + 预留槽(阶段) + 步数 + 3 布尔 + 16 B 尾部 |
| 尾部 16 B | — | ack_seq + rcv_nxt + win + 4×u8 压缩手框 + 2 预留 |
| CRC | 4 B | CRC32，覆盖 magic 到载荷结尾 |

要点：定长帧一次 `struct.unpack` 解完；4 个点而不是 2 个（将来换旋转框不改协议）；
`ref_seq` 回显上行序号，乱序/迟到结果按帧号入库、显示只用最新；`cost_us` 是服务端推理耗时。

## 踩坑表

| 现象 | 原因 / 处理 | 在哪 |
|---|---|---|
| 框跑到零件前面 / 跟不上 | 框按"现在"外推，而画面是 0.7 秒前的 → 改成按帧号取结果 | `edge/parts.py` |
| 单帧漏检框就闪掉 | 结果回来了但这一帧模型没检出 → 按类别短时兜底 0.4 秒 | `edge/parts.py` `_hold` |
| 识别帧只有 14 | 主循环和发送线程抢同一把 GIL → 发送拆进程 | `edge/sender_proc.py` |
| 手部每两帧报 TypeError，半边帧收不到回包 | MediaPipe 返回的是列表不是字典 | `server/service.py` |
| 服务器容量只有 34 帧/秒 | 阶段模型每帧都跑 → 改每 10 帧，跳过的帧沿用上次结果 | `server/service.py` `--stage-every` |
| 红点圈和屏上的点对不上 | 跟踪跑在最新帧、画在缓冲帧 → 跟踪挪到出队之后 | `edge/app.py` |
| 红点轴心偏 40px | 原来用 ROI 中心 → 改用装配区 base 框心，圆拟合偏离 60px 就丢弃 | `edge/rod_dot.py` |
| 零件框"预测全部跑飞" | 宽高进了卡尔曼状态 → 尺寸不参与滤波 | `edge/tracking.py` |
| 同一零件两个框 | 丢检旧轨迹 + 新轨迹 → 按重叠度合并 | `edge/parts.py` |
| 区域框横跨两个料盘 | 两盘挨着时暗色掩膜连成一块 → 先找缝切开 | `edge/zones.py` |
| 一直显示"装配中" | 只看暗色可见量会漏判 → 加"亮占比增量"第二路 | `edge/zone_tracker.py` |

## 环境

| 用途 | 需要 |
|---|---|
| 树莓派 `edge/` | Python 3 + numpy + opencv + PIL（画中文），仅标准库做网络 |
| 电脑 `server/` | torch + ultralytics + mediapipe（conda 环境 `yolo26`） |

本机：Python 3.12 / torch 2.13 / ultralytics 8.4 / RTX 5060。
树莓派：Raspberry Pi 4B、Bookworm、1280×960 MJPEG、/dev/fb0 直写。

## 注意

- 区域框存在 `zones.json`（仓库根目录），删掉会重新自动识别；也可以 `--zones "x1,y1,x2,y2;x1,y1,x2,y2"` 手给。
- 树莓派上 `--dequeue-at` 是播放缓冲深度：越大框/画面越稳，但屏幕越滞后（现在 22 帧 ≈ 0.75 秒）。
- 链路是瓶颈时先看树莓派日志里的 `[对齐]` 行：`无seq` 多 = 帧没提交出去（发送窗口/网络），`无结果` 多 = 回包没赶上播放缓冲。
