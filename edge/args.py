"""命令行参数：`--help` 里看到的一切都在这里。

参数按功能分组（相机/料盘区域/零件模型/树莓派节流），想加开关只改这个文件。"""

from __future__ import annotations

import cv2
import argparse

from edge.constants import BACKENDS, DEFAULT_HTTP_WINDOW, DEFAULT_COVER_DELTA, DEFAULT_COVER_MARGIN, DEFAULT_COVER_WINDOW, DEFAULT_DARK_V_MAX, DEFAULT_GREEN_HI, DEFAULT_GREEN_LO, DEFAULT_INFER_BACKEND, DEFAULT_JPEG_QUALITY, DEFAULT_KF_ACC, DEFAULT_KF_MEAS, DEFAULT_LINE_WIDTH, DEFAULT_SERVER_TIMEOUT, DEFAULT_SERVER_URL, DEFAULT_TRACK_MAX_AGE, DEFAULT_TRAIL_LEN, DEFAULT_WIRE, DEFAULT_YOLO_CONF, DEFAULT_YOLO_IMGSZ, DEFAULT_YOLO_INTERVAL, DEFAULT_YOLO_IOU, DEFAULT_YOLO_WEIGHTS, DEFAULT_ZONES_FILE, DEFAULT_ZONE_INTERVAL, DEFAULT_ZONE_LOCK_FRAMES, DEFAULT_ZONE_SCAN_TOP, DEFAULT_ZONE_WIDTH, GREEN_MIN_AREA_RATIO, GREEN_ZONE_MIN_AREA_RATIO, LABEL_FONT_SIZE, STAGE_FONT_SIZE
from edge.stage_match import (DEFAULT_ANGLE_RANGE, DEFAULT_ANGLE_STEP,
                              DEFAULT_KP_GATE, DEFAULT_MIN_INLIERS,
                              DEFAULT_REFINE_STEP, DEFAULT_STAGE_INTERVAL,
                              DEFAULT_STAGE_MARGIN, DEFAULT_STAGE_SIZE,
                              MIN_STAGE_SCORE, STAGE_DEFAULT_DIR)


DETECT_ARGUMENTS = (
    (("--detect-width",), dict(type=int, default=960,
                               help="零件模型的输入宽度（0=用原分辨率）；"
                                    "料盘区域识别始终在原分辨率上做")),
    (("--detect-interval",), dict(type=int, default=1,
                                  help="每 N 帧做一次识别（>1 可提帧率，中间帧沿用上次结果）")),
    # ---- 抽帧节流：按帧数（--yolo-interval 10 = 每 10 帧跑一次 YOLO）----
    (("--yolo-interval",), dict(type=int, default=DEFAULT_YOLO_INTERVAL,
                                help="每多少帧跑一次 YOLO（默认 10；中间帧用卡尔曼预测补框）")),
    (("--zone-interval",), dict(type=int, default=DEFAULT_ZONE_INTERVAL,
                                help="每多少帧量一次料盘遮挡（取料/装配状态不需要每帧判）")),
    (("--zone-width",), dict(type=int, default=DEFAULT_ZONE_WIDTH,
                             help="料盘掩码的计算宽度（0=原分辨率；Pi 上建议 640，省 CPU）")),
    (("--infer-backend",), dict(choices=("ultralytics", "onnx", "http"),
                                default=DEFAULT_INFER_BACKEND,
                                help="推理后端：ultralytics=本机 torch；onnx=本机 cv2.dnn；"
                                     "http=**把算力放到局域网里的电脑**（跑 infer_server.py）")),
    (("--server-url",), dict(default=DEFAULT_SERVER_URL,
                             help="http 后端：电脑上识别服务的地址，如 http://192.168.1.10:8080")),
    (("--server-token",), dict(default="",
                               help="http 后端：和服务端 --token 一致的口令（可留空）")),
    (("--jpeg-quality",), dict(type=int, default=DEFAULT_JPEG_QUALITY,
                               help="http 后端：发出去的 JPEG 质量（越小越省带宽、越糊）")),
    (("--server-timeout",), dict(type=float, default=DEFAULT_SERVER_TIMEOUT,
                                 help="http 后端：单次请求超时秒数，超时这一轮就没结果")),
    (("--http-window",), dict(type=int, default=DEFAULT_HTTP_WINDOW,
                               help="http 后端：滑动窗口大小（同时在飞的帧数）。"
                                    "吞吐 ≈ 窗口 / 往返，Wi-Fi 下 6 帧约能到 30fps；"
                                    "实际以服务端 /health 回的 window 为准")),
    (("--wire",), dict(choices=("binary", "json"), default=DEFAULT_WIRE,
                       help="http 后端的帧格式：binary=protocol.py 的二进制帧"
                            "（上行图像帧、下行恒定 160 字节，带 step/布尔/手部框/耗时）"
                            " | json=老的 JSON（调试或对接旧服务端时用）")),
    (("--upload-crop",), dict(type=int, nargs=4, metavar=("X", "Y", "W", "H"),
                              default=None,
                              help="只把这块「工作区」发去识别（原画面像素坐标）。"
                                   "面积小了 JPEG 更小、树莓派编码更快，"
                                   "而且模型在同样 imgsz 下看到的零件更大更准")),
    # ---- 料盘区域：默认按"黑色料盘"分割，绿色是备选 ----
    (("--zone-mode",), dict(choices=("dark", "green"), default="dark",
                            help="料盘分割方式：dark=黑色料盘（现在），green=老版绿色料盘")),
    (("--dark-v-max",), dict(type=int, default=DEFAULT_DARK_V_MAX,
                             help="dark 模式：HSV 明度上限，V 低于它才算料盘")),
    (("--zone-scan-top",), dict(type=float, default=DEFAULT_ZONE_SCAN_TOP,
                                help="dark 模式：只在该比例以下找料盘"
                                     "（排除顶部同样是黑色的机器臂）")),
    (("--zones",), dict(default=None, metavar="X1,Y1,X2,Y2;X1,Y1,X2,Y2",
                        help="手工指定两个区域（前=左=装配区，后=右=物料区），"
                             "给了就不再自动识别")),
    (("--zones-file",), dict(default=None,
                             help=f"区域文件，默认 <项目根>/{DEFAULT_ZONES_FILE}；"
                                  "自动锁定后会写入，下次启动直接读")),
    (("--no-zone-save",), dict(action="store_true",
                               help="自动锁定后不写区域文件")),
    (("--zone-lock-frames",), dict(type=int, default=DEFAULT_ZONE_LOCK_FRAMES,
                                   help="连续多少帧都认出两个料盘就锁定区域")),
    (("--green-lo",), dict(type=int, nargs=3, metavar=("H", "S", "V"),
                           default=list(DEFAULT_GREEN_LO), help="green 模式：绿色 HSV 下限")),
    (("--green-hi",), dict(type=int, nargs=3, metavar=("H", "S", "V"),
                           default=list(DEFAULT_GREEN_HI), help="green 模式：绿色 HSV 上限")),
    (("--min-area",), dict(type=float, default=GREEN_MIN_AREA_RATIO,
                           help="色块最小面积占整幅画面的比例（连通域过滤）")),
    (("--zone-min-area",), dict(type=float, default=GREEN_ZONE_MIN_AREA_RATIO,
                                help="被当作「区域」的最小面积占整幅画面的比例；小于它的"
                                     "杂物会被忽略，避免它抢走物料区/装配区的框")),
    (("--merge-gap",), dict(type=int, default=0,
                            help="水平间距小于该值的色块合并（料盘被反光切开时用）")),
    (("--occlude-ratio",), dict(type=float, default=0.75,
                                help="区域内当前可见量低于「基准×该比例」判为被遮挡"
                                     "（显示取料中/装配中）；实测手遮住料盘一部分时约 0.66")),
    (("--cover-delta",), dict(type=float, default=DEFAULT_COVER_DELTA,
                              help="框内「亮像素占比」比最近静止水平高这么多，也判为"
                                   "被遮挡（补 --occlude-ratio 的漏判）")),
    (("--cover-margin",), dict(type=int, default=DEFAULT_COVER_MARGIN,
                               help="「亮」阈值 = 框内料盘亮度中位数 + 这个余量"
                                    "（跟着现场光照走，偏暗的现场也不会把手判成暗的）")),
    (("--cover-window",), dict(type=int, default=DEFAULT_COVER_WINDOW,
                               help="静止水平取最近多少帧的最小值（装好的零件过一会儿"
                                    "就不再算遮挡）")),
    # ---- 零件检测+跟踪（训练好的 YOLO 权重）----
    (("--no-yolo",), dict(action="store_true",
                          help="不加载零件模型，只做料盘区域识别")),
    (("--no-track",), dict(action="store_true",
                            help="不做卡尔曼跟踪（零件框只按检测更新，不预测、不画轨迹）")),
    (("--no-trail",), dict(action="store_true",
                           help="不画运动轨迹（框照常）")),
    (("--line-width",), dict(type=int, default=DEFAULT_LINE_WIDTH,
                             help="框线粗细（源图像素）。外接屏会把 1280x960 缩到 800x600，"
                                     "所以 1px 在屏上只有 0.6px；默认 3")),
    (("--label-size",), dict(type=int, default=LABEL_FONT_SIZE,
                             help="框上标签的字号（默认 14；调大能看清但会盖住零件）")),
    (("--yolo-weights",), dict(default=None,
                               help=f"零件模型权重；默认 <项目根>/{DEFAULT_YOLO_WEIGHTS}")),
    (("--yolo-conf",), dict(type=float, default=DEFAULT_YOLO_CONF,
                            help="零件检测置信度阈值")),
    (("--yolo-iou",), dict(type=float, default=DEFAULT_YOLO_IOU, help="NMS 的 IoU 阈值")),
    (("--yolo-imgsz",), dict(type=int, default=DEFAULT_YOLO_IMGSZ,
                             help="模型输入尺寸（训练用的 960；降 640 换帧率）")),
    (("--yolo-device",), dict(default="0", help="推理设备：0=第一块 GPU，cpu=CPU，空=自动")),
    (("--trail-len",), dict(type=int, default=DEFAULT_TRAIL_LEN,
                            help="轨迹保留的点数（越多尾巴越长）")),
    (("--track-max-age",), dict(type=int, default=DEFAULT_TRACK_MAX_AGE,
                                help="连续这么多次 YOLO 都没检出就丢掉该轨迹"
                                     "（中间靠卡尔曼预测补框）")),
    (("--kf-acc",), dict(type=float, default=DEFAULT_KF_ACC,
                         help="卡尔曼加速度噪声 px/s^2：调大=速度收敛快、跟得紧但抖；"
                              "调小=更平滑但滞后")),
    (("--kf-meas",), dict(type=float, default=DEFAULT_KF_MEAS,
                          help="卡尔曼测量噪声 px：调大=更信预测（更平滑），"
                               "调小=更信检测（更跟手）")),
    # ---- 阶段识别（基础视觉：旋转遍历 + 模板匹配，和"零件跟踪"并列共存）----
    (("--no-stage",), dict(action="store_true", help="关掉阶段识别")),
    (("--stage-method",), dict(choices=("cascade", "sift", "grad"), default="cascade",
                               help="阶段匹配算法：cascade=按关键点数分流"
                                    "（少→稠密块近邻，多→SIFT，推荐）；"
                                    "sift=只用特征点；grad=旧的梯度模板法")),
    (("--stage-kp-gate",), dict(type=int, default=DEFAULT_KP_GATE,
                                help="关键点数门槛：低于它走近邻（光面阶段），"
                                     "高于它走 SIFT（有齿/有孔阶段）")),
    (("--stage-min-inliers",), dict(type=int, default=DEFAULT_MIN_INLIERS,
                                    help="SIFT 分支内点数低于它就记 unknown")),
    (("--stage-dir",), dict(default=STAGE_DEFAULT_DIR,
                            help="阶段参考图目录（平铺=文件名，子目录=目录名+多样本）")),
    (("--stage-region",), dict(type=int, nargs=4, metavar=("X", "Y", "W", "H"),
                               default=None,
                               help="强制指定装配区裁剪窗口；不给就用锁定后的装配区框")),
    (("--stage-label-size",), dict(type=int, default=STAGE_FONT_SIZE,
                                   help="「阶段」标签在画面上的字号（默认比普通标签大一号）")),
    (("--stage-period",), dict(type=float, default=0.5,
                               help="每隔多少秒识别一次阶段（按时间，不按帧数）")),
    (("--stage-window",), dict(type=float, default=1.0,
                               help="判定窗口（秒）：窗口内多数一致才采纳，单帧抖动不翻牌")),
    (("--stage-edges",), dict(action="store_true",
                              help="先偏导求边缘再匹配（边界图），抑制托盘/背景纹理；"
                                   "纯 SIFT 模式下 0/1 步明显变好，级联模式下会让关键点门槛失效")),
    (("--stage-interval",), dict(type=int, default=DEFAULT_STAGE_INTERVAL,
                                 help="[旧] 每多少帧识别一次；已被 --stage-period 取代")),
    (("--angle-step",), dict(type=float, default=DEFAULT_ANGLE_STEP,
                             help="旋转遍历步长（度）；默认 1° 走一整圈")),
    (("--angle-min",), dict(type=float, default=DEFAULT_ANGLE_RANGE[0],
                            help="旋转遍历起始角（度）")),
    (("--angle-max",), dict(type=float, default=DEFAULT_ANGLE_RANGE[1],
                            help="旋转遍历结束角（度）")),
    (("--refine-step",), dict(type=float, default=DEFAULT_REFINE_STEP,
                              help="命中角度附近细搜步长（度），用来逼近近似最大")),
    (("--no-rotate",), dict(action="store_true",
                            help="不做旋转遍历（最快，但零件摆向一变就容易认不出）")),
    (("--stage-size",), dict(type=int, default=DEFAULT_STAGE_SIZE,
                             help="阶段图规格化边长（px）；越小越快越糙")),
    (("--stage-margin",), dict(type=float, default=DEFAULT_STAGE_MARGIN,
                               help="裁剪窗口外扩比例（把料盘边缘也框进去）")),
    (("--stage-min-score",), dict(type=float, default=MIN_STAGE_SCORE,
                                  help="相似度低于它就把阶段显示成 unknown")),
)

def build_parser():
    parser = argparse.ArgumentParser(
        description="右目视图 + 料盘区域识别 + 零件跟踪",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--camera", type=int, default=0, help="相机序号（见 list_cameras.py）")
    parser.add_argument("--source", default=None,
                        help="直接指定视频文件或图片路径；给了就不用相机（离线验证用）")
    parser.add_argument("--backend", choices=tuple(BACKENDS), default="dshow")
    parser.add_argument("--mode", default="2560x960",
                        help="请求分辨率，形如 2560x960（默认；若卡顿可试 2560x720 或 1280x960）")
    parser.add_argument("--width", type=int, help="直接指定帧宽（优先于 --mode）")
    parser.add_argument("--height", type=int, help="直接指定帧高")
    parser.add_argument("--fps", type=int, default=30, help="请求帧率（默认 30）")
    parser.add_argument("--fourcc", default="MJPG", help="像素格式，默认 MJPG（YUY2 会极卡）")
    parser.add_argument("--buffersize", type=int, default=2,
                        help="驱动缓冲帧数：1 最省延迟但只有 24fps（实测），2 是 40fps 的平衡点")
    parser.add_argument("--layout", choices=("auto", "side_by_side", "top_bottom", "none"),
                        default="auto", help="auto 对半切；none 表示整帧即单目（右目）")
    parser.add_argument("--warmup", type=int, default=30, help="启动丢弃帧数")
    parser.add_argument("--resize", type=int, default=0, help="缩放到该宽度（0 不缩放）")
    parser.add_argument("--loop-fps", type=float, default=30.0,
                        help="主循环固定节拍（0=不限制，跑满 CPU）。锁到 30 后采集/入队/出队"
                             "同一个节拍，帧率不再忽快忽慢")
    parser.add_argument("--roi-px", type=int, nargs=4, metavar=("X", "Y", "W", "H"),
                        help="标出识别区域（像素，缩放前坐标）")
    parser.add_argument("--no-window", action="store_true",
                        help="不开预览窗口（树莓派/Docker 无显示器时用；配合 --save 定期存图）")
    parser.add_argument("--fb", default=None, metavar="DEV",
                        help="把画面直接写进 framebuffer（无图形界面的外接屏），例如 /dev/fb0；"
                             "给了它就不再开窗口、不响应按键，Ctrl+C 退出")
    parser.add_argument("--fb-mode", choices=("fit", "crop", "stretch"), default="fit",
                        help="framebuffer 缩放方式：fit=留黑边等比 / crop=裁切铺满 / stretch=拉伸")
    parser.add_argument("--fb-rotate", type=int, choices=(0, 90, 180, 270), default=0,
                        help="framebuffer 输出旋转角度（屏幕竖装时用）")
    parser.add_argument("--snapshot", action="store_true", help="抓一帧保存后退出")
    parser.add_argument("--no-sync-display", action="store_true",
                        help="画面不跟结果走：显示最新帧（框会比画面慢一个网络往返）")
    parser.add_argument("--dequeue-at", type=int, default=20,
                        help="接收队列攒够多少帧才开始出队（之后每帧弹最老的一帧）")
    parser.add_argument("--preroll", type=int, default=0,
                        help="接收缓冲区**首次**攒够多少帧才开始出图（一次性，之后不再判断）")
    parser.add_argument("--preroll-timeout", type=float, default=3.0,
                        help="预充最多等多少秒，超时就先出图（防止一直不出）")
    parser.add_argument("--sync-seconds", type=float, default=8.0,
                        help="画面/结果配对的环形缓冲保留多少秒（默认 1.0，"
                             "要覆盖网络往返；RTT 0.5s 就得留 0.5s 以上）")
    parser.add_argument("--sync-max-frames", type=int, default=40,
                        help="环形缓冲最多存几帧（防内存爆掉，1.8G 的派约 40 帧=150MB）")
    parser.add_argument("--save", action="store_true",
                        help="把带识别结果的画面存到 --save-dir（图片源存单张结果图）")
    parser.add_argument("--probe", action="store_true", help="逐模式实测帧率后退出")
    parser.add_argument("--fps-test", type=int, default=0, metavar="N",
                        help="读 N 帧并报告端到端实测帧率后退出（只测相机+绘制，"
                             "不含零件模型推理；带模型的实际帧率看窗口状态栏）")
    parser.add_argument("--save-dir", default="output")
    parser.add_argument("--no-capture", action="store_true",
                        help="不显示数据采集按钮（默认显示 base/support/compass/top/全部）")
    parser.add_argument("--capture-overlay", action="store_true",
                        help="采集时把识别结果也存进图片（默认**不存**：存干净的原始画面，"
                             "免得误识别的框被烧进训练数据）")
    parser.add_argument("--capture-root", default=None,
                        help="采集图片的保存根目录，默认项目根；图片落在 "
                             "<根>/vision_yolo_train/images/parts/ 下")

    group = parser.add_argument_group("料盘区域识别 + 零件跟踪")
    group.add_argument("--no-detect", action="store_true",
                       help="关掉识别，只做右目预览")
    for flags, options in DETECT_ARGUMENTS:
        group.add_argument(*flags, **options)
    return parser
