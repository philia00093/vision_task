# vision_yolo_train — YOLO 视觉模型训练工程

用 **YOLO26n** 训练两个检测任务：

| 数据集 | 识别对象 | 类别 |
|---|---|---|
| `wersion_parts` | **零件识别** | base 底部 / support 支撑柱 / compass 罗盘 / top 盘 |
| `wersion_steps` | **步骤识别** | step_0 空盘 → step_1 只有底部 → step_2 +支撑柱 → step_3 +罗盘 → complete 完整体 |

> 类别名与采集图片前缀一致（`base_*.png` 等），以图片文件名为准。

## 标注操作（X-AnyLabeling）

图片已按类别归档好，**打开一个类别目录就只标那一类**，不会标错类别：

```
datasets/annotate/parts/
├─ base/images/     base_1.png … base_60.png        ← 底部
├─ support/images/  support_1.png … support_60.png  ← 支撑柱
├─ compass/images/  compass_1.png … compass_60.png  ← 罗盘
└─ top/images/      top_1.png … top_60.png          ← 盘
```

**步骤 1**：X-AnyLabeling 打开某个类别的 `images/` 目录（例如 `parts/base/images/`）。
**步骤 2**：每张图给目标物体画**一个矩形框**，类别名**必须**填该类别的名字：

| 目录 | 标注时填的类别名 |
|---|---|
| `parts/base/` | `base` |
| `parts/support/` | `support` |
| `parts/compass/` | `compass` |
| `parts/top/` | `top` |

> 类别名写错（比如在 base 目录里填 `bottom`）会导致转换时报"未知类别"，所以照上表填。

**步骤 3**：导出格式选 **YOLO Detection**，导出/保存到 `vision_yolo_train/annotations/`。
（也可以直接导出 YOLO `.txt`，转换脚本两种都认。）

**步骤 4**：标注完跑转换 + 划分：

```bash
python vision_yolo_train/tools/convert_anylabeling.py --data wersion_parts
python vision_yolo_train/tools/split_dataset.py --data wersion_parts --val-ratio 0.2 --test-prefix test --test-count 20 --clean
python vision_yolo_train/tools/check_labels.py --data wersion_parts --splits train val test
python vision_yolo_train/train_detect.py --data wersion_parts
```

> `--test-prefix test --test-count 20`：从 `test_*.png` 里随机抽 20 张单独放 `images/test`，
> 并在 yaml 里写 `test: images/test`。抽签用 `--seed`（默认 0），同一 seed 每次都抽到同一批图，
> 想换一批就改 `--seed`。不加这两个参数就只分 train/val。
> 训练完用 `yolo detect val model=<权重> data=<yaml> split=test` 在测试集上评估。

> 标注文件放哪都行：用 `annotate_parts.bat` 导出到 `annotations/` 可以，直接打开
> `images/parts/` 标（json 会存在图片旁边）也可以，`convert_anylabeling.py` 两个位置都认，
> 并且会去 `datasets/annotate/`、`images/` 里找同名图片一起收进数据集。

> **补标/重标之后要重跑划分，而且必须加 `--clean`**：
> `split_dataset.py` 只往里复制、不清旧文件，不加 `--clean` 会让上一轮的图
> 同时留在 train 和 val 里（等于验证集泄题）。
> ```bash
> python vision_yolo_train/tools/split_dataset.py --data wersion_parts --clean
> ```

`split_dataset.py` 会按类别分层划分训练集/验证集（每个类别在 val 里都有代表），
并把 yaml 的 `path` 改写成当前机器的绝对路径。

## 重新归档图片（如果又采集了新数据）

新采集的图片都在 `images/parts/` 下按前缀命名，重新归档即可（复制模式，原图保留）：

```bash
python vision_yolo_train/tools/prepare_annotation.py          # 预览
python vision_yolo_train/tools/prepare_annotation.py --apply  # 执行
```

编号会自动接续，已归档的图片不会重复复制。`test_*.png` 是通用样本（可能含多个类别），
默认跳过归档，需要时用 `--skip` 调整。

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
| 转换后 train/val 里混进上一轮的图 | 重跑划分时加 `--clean`，否则旧 split 的文件会残留（可能同一张图同时在 train 和 val） |
| 训练报 `images not found` | 跑一次 `split_dataset.py --data <数据集>`，它会把 yaml 的 `path` 改写成当前机器的绝对路径 |
| `download_weights.py` 报 404 | 权重名写错。只有 `yolo26n/s/m/l/x.pt`，**没有 `yolo26.pt`** |
| 训练很慢 | 降低 `--imgsz` 或 `--batch`；确认用的是 GPU（`--device 0`） |
