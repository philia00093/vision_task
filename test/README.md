# test/ — 无通讯版（自己写逻辑的沙盒）

这个目录里**没有任何网络代码**：不开端口、不发请求、不连电脑。
相机、料盘判定、零件检测、卡尔曼跟踪、绘制、采集全部直接复用 `edge/` 里已经验证过的实现，
所以这里只剩「主循环 + 你的逻辑」，好读、好改。

| | `edge/run_edge.py`（正式部署） | `test/run_test.py`（本文件所在目录） |
|---|---|---|
| 推理 | 本机 torch / 本机 ONNX / **远程 HTTP** | **只有本机**（torch 或 ONNX） |
| 通讯参数 | 有（`--server-url` 等） | **一个都没有** |
| 用途 | 树莓派 + 电脑 | 单机开发、加自己的逻辑 |

## 文件

| 文件 | 作用 |
|---|---|
| `run_test.py` | 主循环。和 `edge/app.py` 同构，但去掉全部通讯；阶段识别两条路都在这里调度 |
| `step_recognizer.py` | **规则版阶段识别 —— 要改判定规则就改这个文件**。数据契约 + 默认实现（很粗，按你的工艺替换） |

模板版（基础视觉）阶段识别不在这里，在 `edge/stage_match.py`：它只吃装配区那块裁剪图，
旋转 1° 遍历一圈跟参考图比相似度，跟零件框完全无关。

## 怎么跑

```powershell
# 相机 + 本机模型（阶段走模板匹配，默认）
D:\anaconda3\envs\yolo26\python.exe test\run_test.py

# 离线跑一张图（不碰相机）
D:\anaconda3\envs\yolo26\python.exe test\run_test.py --source 某张图.png --layout none --snapshot --no-capture

# 只跑料盘判定 + 阶段（不加载模型，纯 opencv）
python test\run_test.py --no-yolo

# 阶段：两条路都跑，画面上对照看
python test\run_test.py --stage-mode both

# 阶段：只要规则版（按零件组合）
python test\run_test.py --stage-mode parts

# 阶段：关掉模板匹配（--no-stage 是 edge/args.py 里的统一开关）
python test\run_test.py --no-stage
```

热键：`S` 存图，`Q`/`ESC` 退出，`1~5` 采集训练数据。

无显示器（树莓派/Docker）加 `--no-window`，配合 `--save` 定期存图。

## 两条阶段路线对照

| `--stage-mode` | 走哪条路 | 显示在哪 | 依赖 |
|---|---|---|---|
| `template`（默认） | 装配区裁剪图旋转遍历 + 参考图相似度 | 装配区框右上角绿色「阶段 step_3 0.87 +12.4°」 | 只要 opencv + `vision_yolo_train/images/steps/` |
| `parts` | 按装配区里检出的零件组合推断 | 状态栏 `| step step_2` | 要 YOLO 检出零件 |
| `both` | 两条都跑 | 两处都有，方便对照 | 两样都要 |

两条路**互不干扰**：模板不看零件框（YOLO 关掉、认错都不影响它），规则版不看图像。
要哪条开哪条，或同时开。

## 你的逻辑写在哪

每帧都会执行 `run_test.py` 里「阶段识别：两条路互不干扰」那一段：

```python
# template 路线（隔 --stage-interval 帧才真跑一次）
if stage is not None:
    stage_result = stage.match(view, region=tracker_zones.region("装配区"))

# parts 路线
if step_recognizer is not None:
    step_text = step_recognizer.update(tracker_zones.zones, detections)
```

- **输入**：`zones`（两个区域，带 `occluded`/`visible_ratio` 等）、
  `detections`（零件框、类别、置信度、track_id、轨迹、是否是卡尔曼补的）
  —— 字段含义在 `step_recognizer.py` 文件开头的「数据契约」里写全了。
- **输出**：一个字符串，会显示在**状态栏**（`| step complete`），变化时也打印到控制台。
- 想输出更多信息（进度、置信度、缺哪件），把返回值改成 dict，再改 `run_test.py` 里显示它的两行即可。

## 默认的阶段判定（可整体换掉）

只看**装配区里出现了哪些类别**，取满足条件的最高阶段：

| 装配区里的零件 | 阶段 |
|---|---|
| （空） | `step_0` |
| base | `step_1` |
| base + support | `step_2` |
| base + support + compass | `step_3` |
| 四件都有 | `complete` |

明显不够严谨的地方（都留给你按工艺补）：

- **没有顺序约束**：只要凑齐就算，不管先后；
- **没有时间约束**：手抖一下多检出/漏检出就跳步（可以考虑"连续 N 帧一致才算"）；
- **没用区域信息**：零件是在物料区还是装配区，默认实现只用"中心点落在装配区内"；
- **没用置信度**：`conf` 0.35 的低分检出也算数；
- **没用轨迹**：`track_id`/`trail` 没用上，其实可以用来判"同一个零件被拿起来走动"。

## 和 edge/ 的关系

`test/` 只是 `edge/` 的**用法示例 + 逻辑沙盒**，不改 `edge/` 一行代码。
如果哪天你觉得某段需要改在 `edge/` 里（比如相机协商、掩膜阈值），改 `edge/` 会同时影响正式部署，
改 `test/` 只影响这里 —— 这是刻意的分工。
