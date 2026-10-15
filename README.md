# vision_task — 视觉识别 + YOLO 训练

相机画面：识别**物料区**（画面右侧，红框）与**装配区**（画面左侧，蓝框），
并按物料区是否被遮挡显示工位状态（**取料中** / **装配中**）。
另含用 YOLO26n 训练**零件识别**与**装配步骤识别**的完整数据与代码。


---

## 1. 目录结构

```
vision_task/
├─ test_run.py                 ★ 主程序：预览 + 绿色区域识别 + 工位状态
├─ stereo_fusion.py            双目拆帧与图像读写（中文路径安全）
├─ output/                     运行输出（已 gitignore）
└─ vision_yolo_train/          ★ YOLO 视觉模型训练工程
   ├─ train_detect.py            训练入口（YOLO26n 检测，最常用）
   ├─ paths.py                   ★ 统一路径定义，所有脚本都从这里取路径
   ├─ tools/                     抽帧 / 转换标注 / 划分 / 体检 / 下权重 / 建骨架
   ├─ configs/                   数据集 yaml 与类别表
   ├─ weights/                   预训练权重：yolo26n.pt（已随仓库提供，开箱即用）
   ├─ images/                    ★ 待标注图片总目录
   │  ├─ parts/                  零件识别：底部 / 支撑柱 / 盘 / 罗盘
   │  └─ steps/                  步骤识别：step_0~3 / complete
   ├─ datasets/                  YOLO 数据集（images/{train,val,raw} + labels/…）
   │  ├─ wersion_parts/          零件识别数据集
   │  ├─ wersion_steps/          步骤识别数据集
   │  └─ annotate/               按类别分开的标注工作区
   │     ├─ parts/{bottom,support_pillar,tray,compass}/{images,labels}
   │     └─ steps/{step_0..step_3,complete}/{images,labels}
   ├─ annotations/               X-AnyLabeling 导出统一放这
   ├─ raw_frames/                视频抽帧原始帧
   └─ runs/                      训练输出（权重、曲线）
```

**路径只有一个来源**：`vision_yolo_train/paths.py`。
其余脚本不再各写 `parents[N]`，脚本挪位置也不会断路径。
想看解析结果：`python vision_yolo_train/paths.py`

## 2. 预训练权重（已随仓库提供）

`vision_yolo_train/weights/yolo26n.pt`（5.3 MB，YOLO26 nano）**已经在仓库里，克隆即可训练**，
不需要额外下载。训练入口默认就用它：

```bash
python vision_yolo_train/train_detect.py --data wersion_parts     # 默认 --model = weights/yolo26n.pt
```

权重丢了或想重新拉取时：

```bash
python vision_yolo_train/tools/download_weights.py
```

> 说明：Ultralytics 只发布带尺寸后缀的权重（`yolo26n/s/m/l/x.pt`），
> **没有名为 `yolo26.pt` 的文件**。要更大的主干用
> `download_weights.py --models yolo26s.pt`，再 `train_detect.py --model .../yolo26s.pt`。

## 3. 直接运行（无需训练）

```bash
python test_run.py                                            # 实时相机
python test_run.py --source vision_yolo_train\images\parts\test_1.png --save
```

热键：`S` 保存当前帧（含识别结果），`Q` / `ESC` 退出。

绿色阈值现场可调（光照不同时）：

```bash
python test_run.py --green-lo 30 60 80 --min-area 0.01 --zone-min-area 0.012
```

| 参数 | 作用 |
|---|---|
| `--green-lo / --green-hi` | 绿色 HSV 范围（实测绿板 H≈75 S≈120 V≈224-255） |
| `--min-area` | 连通域最小面积，滤噪点 |
| `--zone-min-area` | 被当作"区域"的最小面积，滤掉桌面小绿杂物 |
| `--occlude-ratio` | 可见绿量低于「首次检测量×该值」判为被遮挡（默认 0.75） |

## 4. 标注 → 训练流程

```bash
# 1) 抽帧
python vision_yolo_train/tools/extract_frames.py --source 视频.mp4 --every 10
# 2) 用 X-AnyLabeling 标注（零件/步骤选 YOLO Detection），导出到 annotations/
# 3) 转成 YOLO 标签
python vision_yolo_train/tools/convert_anylabeling.py --data wersion_parts
# 4) 划分 train/val（同时把 yaml 的 path 改写成当前机器的绝对路径）
python vision_yolo_train/tools/split_dataset.py --data wersion_parts
# 5) 体检（格式、类别分布、越界坐标）
python vision_yolo_train/tools/check_labels.py --data wersion_parts
# 6) 训练（YOLO26n）
python vision_yolo_train/train_detect.py --data wersion_parts
#    步骤识别同理：--data wersion_steps
```

训练产物在 `vision_yolo_train/runs/<数据集>/weights/best.pt`。

## 5. 类别语义

| 数据集 | 类别 |
|---|---|
| 零件（wersion_parts） | bottom 底部 / support_pillar 支撑柱 / tray 盘 / compass 罗盘 |
| 步骤（wersion_steps） | step_0 空盘 → step_1 只有底部 → step_2 +支撑柱 → step_3 +罗盘 → complete 完整体 |

## 6. 运行环境

本机 `yolo26` 环境：Python 3.11 / torch 2.8.0+cu128 / ultralytics 8.4.157。
两个 `.bat` 已指向 `D:\anaconda3\envs\yolo26\python.exe`，换机器改里面的 `PY` 变量即可。

`test_run.py` 只用标准库 + numpy + opencv + PIL（画中文标签），**不加载任何模型**；
只有训练脚本（`vision_yolo_train/train_detect.py`、`vision_yolo_train/tools/`）需要 ultralytics。
