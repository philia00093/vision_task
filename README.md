# version_task — 料盘/装配区视觉 + 手部跟踪 + 零件与装配步骤训练

针对左目相机画面：**左侧料盘区**、**右侧装配区**，实时跟踪**手**，
并用 YOLO26n 训练**四个零件**（底部 / 支撑柱 / 盘 / 罗盘）和**装配步骤**（step_0 ~ step_3 + 完整体）。

---

## 1. 目录

```
version_task/
├─ left_eye_view.py         左目相机预览/抓帧（原有，保留）
├─ stereo_fusion.py         双目拆帧与图像读写（原有，保留）
├─ run_pipeline.bat         ★ 跑视觉流水线（区域检测 + 手部跟踪）
├─ src/
│  ├─ run_pipeline.py       流水线入口（命令行参数见 --help）
│  ├─ stream.py             视频流读取（相机 MJPEG 协商 / 视频文件 / 图片）
│  ├─ models.py             权重路径解析与模型加载
│  ├─ common.py             路径、类别表、中文路径读写
│  └─ modules/
│     ├─ regions.py         料盘区/装配区检测 + 手越区判断 + 基线 JSON
│     ├─ hand_tracker.py    手部跟踪（21 关键点，ByteTrack/BoT-SORT 给 ID）
│     ├─ draw.py            区域框、手框+ID、骨架、状态栏、告警
│     └─ pipeline.py        三者串成一条链
├─ configs/
│  ├─ data_regions.yaml         区域数据集（tray_area / assembly_area / hand）
│  ├─ data_wersion_parts.yaml   零件数据集（bottom / support_pillar / tray / compass）
│  ├─ data_wersion_steps.yaml   步骤数据集（step_0 ~ step_3 / complete）
│  ├─ data_hand_pose.yaml       手部 21 关键点数据集（YOLO pose）
│  ├─ classes_*.txt             类别表
│  ├─ hand_keypoints_21.txt     21 点顺序（MediaPipe 顺序）
│  └─ trackers/{bytetrack,botsort}.yaml
├─ weights/                 yolo26n.pt、yolo26n-pose.pt
├─ wersion_train/           ★ 训练与标注工作区（见下）
├─ data/                    区域与手部姿态数据集
├─ output/                  标注结果图、区域基线 JSON
└─ scripts/download_weights.py  再下/补下权重
```

## 2. wersion_train 文件夹

```
wersion_train/
├─ datasets/                      训练用数据集（yaml 在 configs/ 下）
│  ├─ wersion_regions/            区域：料盘区 / 装配区 / 手
│  ├─ wersion_parts/              零件：4 类
│  ├─ wersion_steps/              步骤：5 类
│  └─ wersion_hand_pose/          手部 21 关键点
│     每个数据集统一：images/{train,val,raw}  labels/{train,val,raw}
├─ parts/                         零件分开标，避免串类
│  ├─ bottom/{images,labels}/          底部
│  ├─ support_pillar/{images,labels}/  支撑柱
│  ├─ tray/{images,labels}/            盘
│  └─ compass/{images,labels}/         罗盘
├─ steps/                         步骤分开标
│  ├─ step_0/   空盘初始态
│  ├─ step_1/   盘里只有底部
│  ├─ step_2/   底部 + 支撑柱
│  ├─ step_3/   底部 + 支撑柱 + 罗盘
│  └─ complete/ 完整体（盘 + 底部 + 支撑柱 + 罗盘）
├─ annotations/                   X-AnyLabeling 导出统一放这里
├─ raw_frames/                    视频抽帧的原始帧
├─ runs/                          训练输出（权重、曲线）
├─ events/                        手跨越区域的事件 CSV
└─ tools/                         抽帧 / 转换 / 划分 / 体检 / 建目录
```

## 3. 运行方式

```
run_pipeline.bat --source "Left Eye View_screenshot_13.10.2026.png" --save
run_pipeline.bat --source 0                                   # 实时相机预览，Q 退出
run_pipeline.bat --source test.mp4 --loop --save --log-events
run_pipeline.bat --debug-green                                # 存绿色掩码到 output/green/
```

### 区域识别：绿色分割（默认）

料盘是纯色工装，直接 OpenCV 在 HSV 空间分割，**零训练**：左块 = 料盘区，右块 = 装配区。

现场光照变了就调阈值，不用重新标数据：

```
run_pipeline.bat --green-lo 30 60 80 --green-hi 100 255 255   # 放宽绿色范围
run_pipeline.bat --min-area 0.01                              # 提高面积门槛，滤小色块
run_pipeline.bat --merge-gap 40                               # 料盘被反光切成两块时合并
```

识别不到两块时自动几何兜底（`--split-ratio`）并在画面顶部红字提示，不会静默出错。
要换回训练模型：`--no-green --region weights/region_best.pt`。


## 5. 类别语义（重要）

| 数据集 | 类别 | 含义 |
|---|---|---|
| regions | tray_area / assembly_area / hand | 左侧料盘区 / 右侧装配区 / 手 |
| parts | bottom / support_pillar / tray / compass | 底部 / 支撑柱 / 盘 / 罗盘 |
| steps | step_0 | 空盘初始态（盘里什么都没有） |
| | step_1 | 盘里只有**底部** |
| | step_2 | 底部 + 支撑柱 |
| | step_3 | 底部 + 支撑柱 + 罗盘 |
| | complete | 完整体（盘 + 底部 + 支撑柱 + 罗盘） |