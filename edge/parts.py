"""零件检测 + 跟踪的整合层：把"检测结果"变成"每帧可画的框"。

负责：检测↔轨迹关联（按预测中心距离）、卡尔曼补帧、**一个零件一个框**（重叠去重）、轨迹历史。"""

from __future__ import annotations

import time
from collections import deque
from pathlib import Path

from edge.constants import DEFAULT_INFER_BACKEND, DEFAULT_JPEG_QUALITY, DEFAULT_KF_ACC, DEFAULT_KF_ASSOC, DEFAULT_KF_LOST_AFTER, DEFAULT_KF_MEAS, DEFAULT_SERVER_TIMEOUT, DEFAULT_SERVER_URL, DEFAULT_TRACK_MAX_AGE, DEFAULT_TRAIL_LEN, DEFAULT_WIRE, DEFAULT_YOLO_CONF, DEFAULT_YOLO_IMGSZ, DEFAULT_YOLO_IOU
from edge.detection import YoloRunner
from edge.tracking import KalmanTrack, box_iou


class PartTracker:
    """YOLO 检测 + 卡尔曼跟踪，**每一帧**都能给出可画的框（含预测框）。

    这是按树莓派 4B 的算力设计的：
      · YOLO 按 --yolo-interval 抽帧跑（默认 10 帧一次），CPU 才吃得消；
      · 其余帧用卡尔曼预测把框补出来，显示端仍是相机帧率的平滑框；
      · 某件零件这一帧没检出（被手挡/漏检），继续预测并画成橙色虚线；
        连续 --track-max-age 次都没检出就丢掉；
      · **一个零件只保留一个框**：丢检的旧轨迹和新检出的同一零件会合并，
        不会出现"幽灵框 + 新框"两个框叠在一件零件上（见 _merge_duplicates）。

    不做"手拿的零件"高亮：框只有一种颜色，手在哪不影响判定。
    模型/后端加载失败不抛异常，只把原因记在 self.error 里，让调用方降级成"只识别料盘"。
    """

    def __init__(self, weights, backend=DEFAULT_INFER_BACKEND, conf=DEFAULT_YOLO_CONF,
                 iou=DEFAULT_YOLO_IOU, imgsz=DEFAULT_YOLO_IMGSZ, device=None,
                 trail_len=DEFAULT_TRAIL_LEN,
                 track_max_age=DEFAULT_TRACK_MAX_AGE,
                 kf_acc=DEFAULT_KF_ACC, kf_meas=DEFAULT_KF_MEAS,
                 server_url=DEFAULT_SERVER_URL, server_token="",
                 jpeg_quality=DEFAULT_JPEG_QUALITY, server_timeout=DEFAULT_SERVER_TIMEOUT,
                 wire=DEFAULT_WIRE):
        self.runner = YoloRunner(weights, backend=backend, imgsz=imgsz, conf=conf,
                                 iou=iou, device=device, server_url=server_url,
                                 server_token=server_token, jpeg_quality=jpeg_quality,
                                 server_timeout=server_timeout, wire_format=wire)
        self.trail_len = max(int(trail_len), 2)
        self.track_max_age = max(int(track_max_age), 1)
        self.kf_acc = float(kf_acc)
        self.kf_meas = float(kf_meas)
        self.tracks: dict[int, KalmanTrack] = {}
        self.trails: dict[int, deque] = {}
        self.next_id = 1
        self.error = ""
        self.last_cost_ms = 0.0                # 上次 YOLO 的耗时，用来在状态栏显示
        self.frame_size: tuple | None = None   # 最近一次检测帧的尺寸（给卡尔曼做夹取用）

    @property
    def enabled(self) -> bool:
        return self.runner.ready

    @property
    def names(self) -> dict:
        return self.runner.names

    @property
    def weights(self) -> str:
        return self.runner.weights

    def describe(self) -> str:
        """一行式状态，供启动时打印。"""
        if not self.enabled:
            return f"零件跟踪：未启用（{self.runner.error}）"
        labels = ",".join(self.names[i] for i in sorted(self.names))
        where = (f"服务 {self.runner.server_url}"
                 f"（算力在电脑上，帧格式 {self.runner.remote.wire_format}）"
                 if self.runner.remote is not None
                 else f"设备 {self.runner.device or 'auto'}")
        return (f"零件跟踪：{Path(self.weights).name} | 后端 {self.runner.backend} | "
                f"类别 {labels} | conf {self.runner.conf} | imgsz {self.runner.imgsz} | {where}")

    # ------------------------------------------------------------ 稀疏推理
    def detect(self, frame, scale=1.0, offset=(0.0, 0.0), view_size=None) -> None:
        """跑一次 YOLO，并把结果关联到轨迹上（按 --yolo-interval 稀疏调用）。

        frame/scale 是模型输入图及其缩放比；offset 是这张图在原画面里的左上角偏移。
        **只上传工作区裁剪**时（--upload-crop），框要加回这个偏移才是原画面坐标——
        这是裁剪优化唯一的坑，漏加就会整体偏移、框全错位。

        关联用**"检测中心 ↔ 卡尔曼预测中心"的距离**（不是 IoU）：YOLO 是稀疏跑的，
        两次推理之间零件可能移动很多、和上一帧完全没有重叠，按 IoU 会全部断链、
        轨迹号一路涨（实测就是这个现象）。先把各条轨迹外推到"现在"拿到预测位置，
        再按距离从小到大贪心配对即可。
        """
        if not self.enabled:
            return
        now = time.time()
        start = now
        try:
            raw = self.runner.detect(frame)
        except Exception as error:
            # 推理报错不能把预览搞崩，也不能每帧刷同样的报错：打印一次后关掉跟踪
            self.error = f"推理失败（{error}）"
            print(f"[警告] {self.error}；已关闭零件跟踪，只保留料盘识别")
            self.runner.model = None
            self.runner.net = None
            return
        self.last_cost_ms = (time.time() - start) * 1000.0
        # 卡尔曼夹取用的是**整幅画面**的尺寸，不是裁剪后的模型输入尺寸
        if view_size is not None:
            self.frame_size = (int(view_size[0]), int(view_size[1]))
        elif self.frame_size is None:
            self.frame_size = (frame.shape[1], frame.shape[0])

        inverse = 1.0 / scale if scale else 1.0
        dx, dy = float(offset[0]), float(offset[1])
        detections = [((x1 * inverse + dx, y1 * inverse + dy,
                        x2 * inverse + dx, y2 * inverse + dy), cls, conf)
                      for x1, y1, x2, y2, cls, conf in raw]

        # 1) 所有轨迹先外推到"现在"，拿到预测中心
        predicted: dict[int, tuple] = {}
        for track_id, track in self.tracks.items():
            track.width, track.height = self.frame_size
            predicted[track_id] = track.predict(
                now, damp_velocity=track.miss >= DEFAULT_KF_LOST_AFTER)

        # 2) 按"检测中心 ↔ 预测中心"的距离从小到大贪心配对
        pairs: list[tuple] = []
        for det_index, (box, cls_id, _conf) in enumerate(detections):
            cx, cy = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
            size = max(min(box[2] - box[0], box[3] - box[1]), 1.0)
            limit = max(DEFAULT_KF_ASSOC * size, 25.0)
            for track_id, (px, py) in predicted.items():
                if self.tracks[track_id].cls_id != int(cls_id):
                    continue                      # 类别不同的不配，免得两件零件串号
                distance = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5
                if distance <= limit:
                    pairs.append((distance, det_index, track_id))
        pairs.sort(key=lambda item: item[0])

        matched_det: set[int] = set()
        matched_track: set[int] = set()
        for _distance, det_index, track_id in pairs:
            if det_index in matched_det or track_id in matched_track:
                continue
            box, cls_id, conf = detections[det_index]
            self.tracks[track_id].update(box, cls_id,
                                         self.names.get(int(cls_id), f"cls{int(cls_id)}"),
                                         conf, now)
            matched_det.add(det_index)
            matched_track.add(track_id)

        # 3) 没配上的检测 → 新建轨迹；但**先看它是不是叠在已有轨迹上**
        #    （同一零件因为丢检被新建了一条轨迹 → 就会出现两个框，必须并过去）
        for det_index, (box, cls_id, conf) in enumerate(detections):
            if det_index in matched_det:
                continue
            track_id = self._overlapping_track(box)
            if track_id is not None:
                self.tracks[track_id].update(box, cls_id,
                                             self.names.get(int(cls_id), f"cls{int(cls_id)}"),
                                             conf, now)
                matched_track.add(track_id)
                continue
            track_id = self.next_id
            self.next_id += 1
            self.tracks[track_id] = KalmanTrack(
                track_id, box, cls_id,
                self.names.get(int(cls_id), f"cls{int(cls_id)}"),
                conf, self.track_max_age, self.frame_size or (1280, 960),
                std_acc=self.kf_acc, meas_std=self.kf_meas)
            matched_track.add(track_id)

        # 4) 这一轮没配上的轨迹 → 记漏检，超过寿命就丢掉
        for track_id in [k for k in self.tracks if k not in matched_track]:
            track = self.tracks[track_id]
            track.miss += 1
            if not track.alive:
                self.tracks.pop(track_id, None)
                self.trails.pop(track_id, None)

    def _overlapping_track(self, box, iou_limit=0.25):
        """找一个和这个框明显重叠的已有轨迹（类别不限）——用来避免"同一零件两个框"。

        场景：零件被挡了一下 → 上一轮判成丢检、旧轨迹还在预测；这一轮又检出同一零件，
        按"距离 + 同类别"没配上（比如模型这一帧类别给错了），就会新建一条轨迹，
        于是同一件零件上出现"旧预测框 + 新框"两个框。这里按重叠度并回去。
        """
        best_id, best_iou = None, iou_limit
        for track_id, track in self.tracks.items():
            value = box_iou(track.box, box)
            if value > best_iou:
                best_id, best_iou = track_id, value
        return best_id

    # ------------------------------------------------------------ 每帧输出
    def tick(self) -> list[dict]:
        """每一帧都调用：把轨迹推进到当前时刻，返回可以立刻绘制的列表。

        中间帧没有检测，就靠卡尔曼外推补框；**丢检的轨迹会一边外推一边把速度衰减掉**，
        所以它会在原地滑停，而不是沿着最后的方向飞出画面（之前"预测全部跑飞"就是这个）。
        返回前会做一次去重，保证**一个零件只有一个框**。
        """
        now = time.time()
        detections: list[dict] = []
        for track_id, track in self.tracks.items():
            track.predict(now, damp_velocity=track.miss >= DEFAULT_KF_LOST_AFTER)
            box = track.box
            trail = self.trails.setdefault(track_id, deque(maxlen=self.trail_len))
            trail.append(((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0))
            detections.append({
                "box": (int(box[0]), int(box[1]), int(box[2]), int(box[3])),
                "name": track.name,
                "conf": track.conf,
                "track_id": track_id,
                "trail": trail,
                "predicted": track.miss > 0,          # 这一帧没检出，框是预测的
            })
        self._merge_duplicates(detections)
        return detections

    def _merge_duplicates(self, detections: list[dict], iou_limit=0.30) -> None:
        """把明显重叠的多条轨迹合成一条，保证"一个零件只有一个框"。

        保留"这一轮真检出过（predicted=False）、置信度更高"的那条，把另一条直接丢掉
        （连轨迹一起丢，免得轨迹还挂在那里画线）。被丢掉的那条在下一轮会被重新识别，
        但因为这里的合并是按重叠度判的，不会反复分裂。
        """
        ranked = sorted(detections,
                        key=lambda d: (d["predicted"], -d["conf"]))   # 检出优先、高 conf 优先
        keep: list[dict] = []
        for det in ranked:
            duplicate = None
            for other in keep:
                if box_iou(det["box"], other["box"]) > iou_limit:
                    duplicate = other
                    break
            if duplicate is None:
                keep.append(det)
            else:
                self.tracks.pop(det["track_id"], None)       # 丢掉重复轨迹
                self.trails.pop(det["track_id"], None)
        detections[:] = sorted(keep, key=lambda d: d["track_id"])
