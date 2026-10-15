# vesion_task — 视觉识别 + YOLO 训练

相机画面：识别**物料区**（画面右侧，红框）与**装配区**（画面左侧，蓝框），
并按物料区是否被遮挡显示工位状态（**取料中** / **装配中**）。
另含用 YOLO26n 训练**零件识别**与**装配步骤识别**的完整数据与代码。

> 目录名 `vesion_task` 是创建时的拼写（少了一个 `r`），与 GitHub 仓库名一致，未改名以免打断远端关联。

---

## 1. 目录结构

```
vesion_task/
├─ test_run.py             ★ 主程序：右目预览 + 绿色区域识别 + 工位状态
├─ stereo_fusion.py        双目拆帧与图像读写（中文路径安全）
├─ output/                 运行输出（已 gitignore）
└─ wersion_train/          ★ 全部 YOLO 训练相关
   ├─ src/train_detect.py      训练入口（YOLO26n 检测）
   ├─ tools/                   抽帧 / 转换标注 / 划分 / 体检 / 下权重
   ├─ configs/                 数据集 yaml、类别表、跟踪器配置
   ├─ weights/                 预训练权重（.pt 已 gitignore，用脚本重下）
   ├─ datasets/                YOLO 数据集（images/{train,val,raw} + labels/...）
   │  ├─ wersion_parts/        零件识别数据集
   │  ├─ wersion_steps/        步骤识别数据集
   ├─ images/                  ★ 训练图片总目录
   │  ├─ parts/                零件识别用图（底部 / 支撑柱 / 盘 / 罗盘）
   │  └─ steps/                步骤识别用图（step_0~3 / complete）
   ├─ parts/{bottom,support_pillar,tray,compass}/{images,labels}   按类别分开标注
   ├─ steps/{step_0..step_3,complete}/{images,labels}              按步骤分开标注
   ├─ annotations/             X-AnyLabeling 导出统一放这
   ├─ raw_frames/              视频抽帧原始帧
   ├─ runs/                    训练输出
   └─ events/                  手越区事件 CSV
```

## 2. 克隆后先补权重（仓库不含 .pt）

```bash
python wersion_train/tools/download_weights.py
```

会下到 `wersion_train/weights/yolo26n.pt`（检测训练用的预训练权重）。

## 3. 直接运行（无需训练）

```bash
python test_run.py                                            # 实时相机
python test_run.py --source wersion_train\images\parts\test_1.png --save
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
python wersion_train/tools/extract_frames.py --source 视频.mp4 --every 10
# 2) 用 X-AnyLabeling 标注（零件/步骤选 YOLO Detection），导出到 annotations/
# 3) 转成 YOLO 标签
python wersion_train/tools/convert_anylabeling.py --data wersion_parts
# 4) 划分 train/val（同时把 yaml 的 path 改写成当前机器的绝对路径）
python wersion_train/tools/split_dataset.py --data wersion_parts
# 5) 体检（格式、类别分布、越界坐标）
python wersion_train/tools/check_labels.py --data wersion_parts
# 6) 训练（YOLO26n）
python wersion_train/src/train_detect.py --data wersion_parts
#    步骤识别同理：--data wersion_steps
```

训练产物在 `wersion_train/runs/<数据集>/weights/best.pt`。

## 5. 类别语义

| 数据集 | 类别 |
|---|---|
| 零件（wersion_parts） | bottom 底部 / support_pillar 支撑柱 / tray 盘 / compass 罗盘 |
| 步骤（wersion_steps） | step_0 空盘 → step_1 只有底部 → step_2 +支撑柱 → step_3 +罗盘 → complete 完整体 |

## 6. 运行环境

本机 `yolo26` 环境：Python 3.11 / torch 2.8.0+cu128 / ultralytics 8.4.157。
两个 `.bat` 已指向 `D:\anaconda3\envs\yolo26\python.exe`，换机器改里面的 `PY` 变量即可。

`test_run.py` 只用标准库 + numpy + opencv + PIL（画中文标签），**不加载任何模型**；
只有训练脚本（`wersion_train/src`、`wersion_train/tools`）需要 ultralytics。
