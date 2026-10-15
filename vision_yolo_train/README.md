# vision_yolo_train — YOLO 视觉模型训练工程

用 **YOLO26n** 训练两个检测任务：

| 数据集 | 识别对象 | 类别 |
|---|---|---|
| `wersion_parts` | **零件识别** | bottom 底部 / support_pillar 支撑柱 / tray 盘 / compass 罗盘 |
| `wersion_steps` | **步骤识别** | step_0 空盘 → step_1 只有底部 → step_2 +支撑柱 → step_3 +罗盘 → complete 完整体 |

预训练权重 **`weights/yolo26n.pt` 已随仓库提供**，克隆即可开训，无需额外下载。

---

## 目录

```
vision_yolo_train/
├─ train_detect.py        ★ 训练入口
├─ paths.py               ★ 统一路径定义（所有脚本都从这里取路径）
├─ tools/                 建骨架 / 抽帧 / 转标注 / 划分 / 体检 / 下权重
├─ configs/               data_*.yaml 数据集配置、classes_*.txt 类别表
├─ weights/               yolo26n.pt（已提供）
├─ images/                ★ 待标注图片：parts/（零件）、steps/（步骤）
├─ datasets/              YOLO 数据集
│  ├─ wersion_parts/        零件识别数据集
│  ├─ wersion_steps/        步骤识别数据集
│  └─ annotate/             按类别分开的标注工作区
│     ├─ parts/<类别>/{images,labels}/
│     └─ steps/<步骤>/{images,labels}/
├─ annotations/           X-AnyLabeling 导出统一放这里
├─ raw_frames/            视频抽帧的原始帧
└─ runs/                  训练输出（权重、曲线）
```

**路径只有一个来源**：`paths.py`。其余脚本不再各写 `parents[N]`，脚本挪位置也不会断路径。
查看解析结果：

```bash
python vision_yolo_train/paths.py
```

## 标注 → 训练

```bash
# 0) 建目录骨架（幂等，可反复跑）
python vision_yolo_train/tools/init_datasets.py

# 1) 准备图片：相机截图或从视频抽帧，放进 images/parts/ 或 images/steps/
python vision_yolo_train/tools/extract_frames.py --source 视频.mp4 --every 10

# 2) 用 X-AnyLabeling 标注，导出格式选 YOLO Detection，导出到 annotations/

# 3) 转成 YOLO 标签（收集到 datasets/<数据集>/labels/raw）
python vision_yolo_train/tools/convert_anylabeling.py --data wersion_parts

# 4) 划分 train/val（按类别分层；并把 yaml 的 path 改写成当前机器的绝对路径）
python vision_yolo_train/tools/split_dataset.py --data wersion_parts

# 5) 体检：字段数、坐标越界、类别分布、图/标签配对
python vision_yolo_train/tools/check_labels.py --data wersion_parts

# 6) 训练（默认用 weights/yolo26n.pt）
python vision_yolo_train/train_detect.py --data wersion_parts
#    步骤识别同理
python vision_yolo_train/train_detect.py --data wersion_steps
```

训练产物：`runs/<数据集>/weights/best.pt`。

## 常用训练参数

```bash
# 小零件/相邻步骤容易混时，提高分辨率并延长时间
python vision_yolo_train/train_detect.py --data wersion_parts --imgsz 960 --epochs 300

# 换更大主干（先下载）
python vision_yolo_train/tools/download_weights.py --models yolo26s.pt
python vision_yolo_train/train_detect.py --data wersion_parts --model vision_yolo_train/weights/yolo26s.pt

# 中断后继续
python vision_yolo_train/train_detect.py --data wersion_parts --resume

# 其余 ultralytics 参数透传
python vision_yolo_train/train_detect.py --data wersion_parts --extra cos_lr=True
```

默认按 8G 显存给：`imgsz 640`、`batch 8`。显存不够就降 `--batch`。

## 标注注意点

**零件识别（wersion_parts）**
- 每类建议 ≥ 300 个实例，必须包含**被手遮住一半**、**只露一角**的困难样本，否则实时识别会漏检。
- 零件小，训练时优先把 `--imgsz` 提到 960。
- 同一零件在不同工位、不同光照下都要采。

**步骤识别（wersion_steps）**
- 每个步骤只标**一个框**（框住整个装配体/料盘），不要标多个。
- 阶段之间样本量要均衡，否则会偏向样本最多的那一类。
- 最易混的是**相邻步骤**（如 step_1 与 step_2），"正装到一半"的中间态要单独采。
- 若最终决定不要 complete 类、把完整体并入 step_3，删掉该行并同步改
  `configs/classes_wersion_steps.txt` 和标注。

## 常见问题

| 现象 | 原因 / 处理 |
|---|---|
| `check_labels.py` 报"一张图片都没有" | 正常：还没标数据。先抽帧 → 标注 → `convert_anylabeling.py` |
| 训练报 `images not found` | 跑一次 `split_dataset.py --data <数据集>`，它会把 yaml 的 `path` 改写成当前机器的绝对路径 |
| `download_weights.py` 报 404 | 权重名写错。只有 `yolo26n/s/m/l/x.pt`，**没有 `yolo26.pt`** |
| 训练很慢 | 降低 `--imgsz` 或 `--batch`；确认用的是 GPU（`--device 0`） |
