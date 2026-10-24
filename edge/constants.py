"""所有可调常量集中在这里 —— 改行为先看这个文件。

分四组：料盘区域识别、遮挡判据、零件跟踪/卡尔曼、树莓派与网络。"""

from __future__ import annotations

import cv2



WINDOW = "Right Eye View"

BACKENDS = {"auto": cv2.CAP_ANY, "dshow": cv2.CAP_DSHOW,
            "msmf": getattr(cv2, "CAP_MSMF", cv2.CAP_ANY)}

CANDIDATE_MODES = [(2560, 960, 30), (2560, 720, 60), (1280, 960, 60), (1280, 720, 60),
                   (960, 960, 60), (1280, 480, 60), (640, 480, 60)]

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

DEFAULT_DARK_V_MAX = 100            # HSV 明度上限：V 低于它才算"暗"（料盘）

DEFAULT_ZONE_SCAN_TOP = 0.36        # 只在这一比例以下找料盘（上面是黑色机器臂）

DEFAULT_TRAY_MIN_SHARE = 0.012      # 单块料盘至少占整幅的比例（滤掉桌面杂物）

DEFAULT_TRAY_MAX_SHARE = 0.30       # 单块料盘最多占整幅的比例（滤掉整片阴影）

DEFAULT_TRAY_SHAPE_LIMIT = 2.8      # 宽高比上限，超过算细长（机器臂/桌边），不是料盘

DEFAULT_TRAY_SPLIT_RATIO = 1.15     # 宽/高超过它就怀疑是两盘并排 → 找缝切开

DEFAULT_TRAY_SEAM_DROP = 0.75       # 缝那列的暗像素要低于中间带中位数的这个比例才认

DEFAULT_TRAY_FILL_KERNEL = 25        # 料盘找框前先做一次大核闭运算：料盘上半部分反光（V 很高），
                                     # 暗色掩膜会挖掉一块；边缘一圈仍是暗的，闭运算能把中间补回来

DEFAULT_TRAY_MIN_FILL = 0.30        # 料盘中间有大圆凹槽，填充率偏低，门槛要放宽

DEFAULT_ZONE_LOCK_FRAMES = 3        # 连续这么多帧都认出两个盘就锁定，之后不再重认

DEFAULT_ZONES_FILE = "zones.json"   # 锁定的两个区域存这里，下次启动直接读

DEFAULT_COVER_MARGIN = 45           # 亮阈值 = 料盘自身亮度的中位数 + 这个余量（跟着现场光照走）

DEFAULT_COVER_DELTA = 0.12          # 亮占比比静止水平高这么多就认为有东西伸进去

DEFAULT_COVER_WINDOW = 120          # 静止水平取最近这么多帧的最小值（放久了的零件不算遮挡）

DEFAULT_GREEN_LO = (35, 80, 100)

DEFAULT_GREEN_HI = (95, 255, 255)

GREEN_MIN_AREA_RATIO = 0.004        # 连通域最小面积（占整幅画面的比例）

GREEN_ZONE_MIN_AREA_RATIO = 0.012   # 被当作"区域"的最小面积；比这小的色块算杂物，

GREEN_MIN_FILL_RATIO = 0.55         # 外接矩形填充率下限，滤掉细长杂散色块

ZONE_COLORS = [(255, 0, 0),      # 左侧：装配区（蓝）
               (0, 0, 255)]      # 右侧：物料区（红）

STATE_COLOR = (0, 255, 255)      # 状态文字（取料中/装配中）：黄底黑字

STAGE_COLOR = (0, 220, 0)        # 阶段标签（阶段 step_3 0.87）：绿底黑字，与蓝/红区域框区分开

PART_COLOR = (255, 176, 0)       # 零件框

HAND_COLOR = (255, 0, 255)       # 手部框（电脑端下发的手部识别预留位，现在通常没有）

TRAIL_COLOR = (0, 165, 255)      # 轨迹（越旧越暗）

DEFAULT_LINE_WIDTH = 3           # 框线粗细：源图 1px 到 1024x600 屏上只剩 0.6px，
                                 # 太小看不见；3px 缩完约 1.9px，正好

DEFAULT_YOLO_WEIGHTS = "vision_yolo_train/runs/wersion_parts/weights/best.pt"

DEFAULT_YOLO_CONF = 0.35

DEFAULT_YOLO_IOU = 0.5

DEFAULT_YOLO_IMGSZ = 960         # 与训练一致；降 640 可换帧率

DEFAULT_TRAIL_LEN = 20           # 轨迹保留点数（约 0.7 秒的尾巴，太长会像残影）

DEFAULT_TRACK_MAX_AGE = 1        # 只要有一轮 YOLO 没检出就丢掉轨迹 —— 不留残影

DEFAULT_INFER_BACKEND = "ultralytics"   # ultralytics（开发机）/ onnx（Pi 本机）/ http（算力放电脑）

DEFAULT_SERVER_URL = "http://127.0.0.1:8080"   # http 后端：电脑上 infer_server.py 的地址

DEFAULT_JPEG_QUALITY = 80        # 发给电脑的 JPEG 质量（80 时 1280x960 约 100~150KB）

DEFAULT_WIRE = "binary"          # 和电脑通信的帧格式：binary=protocol.py 的二进制帧 / json=老的 JSON

DEFAULT_HTTP_WINDOW = 6        # 允许同时在飞的帧数（滑动窗口），吞吐 ≈ window / RTT

DEFAULT_SERVER_TIMEOUT = 3.0     # 单次请求超时（秒）：留够首帧冷启动的余量；

DEFAULT_YOLO_INTERVAL = 10       # 每多少帧跑一次 YOLO（10 帧 ≈ 30fps 下每秒 3 次）

DEFAULT_ZONE_INTERVAL = 10       # 每多少帧量一次料盘遮挡（状态判定不需要每帧）

DEFAULT_ZONE_WIDTH = 0           # 料盘掩码的宽度（0=原分辨率；Pi 上建议 640，省 CPU）

DEFAULT_PART_NAMES = ("base", "support", "compass", "top")   # ONNX 里没有类别名，用这个表

DEFAULT_KF_ACC = 900.0           # 加速度噪声：手改变速度有多快（调大=更信检测、速度收敛更快）

DEFAULT_KF_MEAS = 10.0           # 检测噪声：模型框的抖动幅度（px，调大=更信预测、更平滑）

DEFAULT_KF_VMAX = 2000.0         # 速度上限 px/s（防飞）

DEFAULT_KF_DT_MAX = 0.5          # 单步时间上限 s（卡顿一下也不会瞬移）

DEFAULT_KF_LOST_DAMP = 0.85      # 丢检后每帧速度衰减系数（滑停，不沿直线飞出去）

DEFAULT_KF_LOST_AFTER = 2        # 连续漏检这么多次后才开始衰减（短暂遮挡别把速度打没）

DEFAULT_KF_SIZE_ALPHA = 0.35     # 尺寸平滑系数（尺寸不进状态，只做轻量平滑）

DEFAULT_KF_ASSOC = 0.6           # 关联距离上限 = 框短边 × 该系数

CJK_FONTS = ("C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf",
             "C:/Windows/Fonts/simsun.ttc", "C:/Windows/Fonts/Deng.ttf")

LABEL_FONT_SIZE = 14             # 框上标签字号（原来 22 太大，会盖住零件；--label-size 可调）

CAPTURE_SPECS = (
    # 按钮名        文件名前缀   输出目录（相对项目根）                     快捷键
    ("base",       "base",      "vision_yolo_train/images/parts",        "1"),
    ("support",    "support",   "vision_yolo_train/images/parts",        "2"),
    ("compass",    "compass",   "vision_yolo_train/images/parts",        "3"),
    ("top",        "top",       "vision_yolo_train/images/parts",        "4"),
    ("全部",        "test",      "vision_yolo_train/images/parts",        "5"),
)

BUTTON_HEIGHT = 34

BUTTON_GAP = 8

BUTTON_MARGIN = 10

BUTTON_RIGHT_RESERVE = 0        # 按钮从左边排起，右侧留给状态提示
