"""手部跟踪模块。

两种模式，按权重是检测模型还是姿态模型自动选：

    pose 模式（推荐，符合"21 关键点"要求）
        权重 yolo26n-pose.pt 或自训练的 21 点手部模型 →
        Ultralytics track() 同时给出框、跟踪 ID 和 21 个关键点。

    detect 模式（还没训练好手部关键点时的过渡方案）
        权重是普通检测模型 → 只拿框 + 跟踪 ID；
        当前检测模型里若有 hand 类，就只保留 hand 类，其余类交给区域检测去处理。

注意：Ultralytics 没有官方手部关键点预训练权重。要拿到稳定的 21 点，
必须用 X-AnyLabeling 标手部 21 点 → configs/data_hand_pose.yaml → yolo pose train
训练出自有权重（见 wersion_train/README.md）。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..models import load_yolo


@dataclass
class HandObservation:
    track_id: int
    box: tuple[float, float, float, float]        # x1, y1, x2, y2
    conf: float
    keypoints: list[tuple[float, float, float]] = field(default_factory=list)  # x, y, v

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.box
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0

    def tip(self, index: int = 8) -> tuple[float, float] | None:
        """取某个关键点坐标，默认食指尖（8），用于判断"手在料盘区还是在装配区"。"""
        if len(self.keypoints) <= index:
            return None
        x, y, v = self.keypoints[index]
        return (x, y) if v > 0 else None


class HandTracker:
    """封装 model.track(persist=True)，把结果转成 HandObservation 列表。"""

    def __init__(self, weights=None, tracker: str = "configs/trackers/bytetrack.yaml",
                 conf: float = 0.35, iou: float = 0.5, imgsz: int = 640,
                 device: str | None = None, verbose: bool = False):
        # 不指定权重时用 21 点姿态模型；自训练权重（best.pt）在名字上看不出任务，
        # 所以只有名字里明确带 pose 时才强制 task=pose，其余交给 ultralytics 推断。
        probe = str(weights) if weights is not None else "yolo26n-pose.pt"
        self.task = "pose" if "pose" in probe.lower() else None
        self.model = load_yolo(weights, task=self.task, verbose=verbose)
        self.is_pose = str(getattr(self.model, "task", "") or "").lower() == "pose"
        self.tracker = str(tracker)
        self.conf, self.iou, self.imgsz, self.device = conf, iou, imgsz, device
        self.names = dict(getattr(self.model, "names", {}) or {})
        # 只有权重类别里存在 hand 类时才做类别过滤：
        # 自训练手部权重通常只有一个 hand 类，而 COCO 人体姿态权重没有 hand 类，
        # 这种情况下不过滤，让所有检测结果都进来（person 框里就包含手）。
        self._hand_ids = {i for i, n in self.names.items()
                          if str(n).strip().lower() in ("hand", "hands", "手")}
        self.last_error: str | None = None

    def update(self, image) -> list[HandObservation]:
        try:
            results = self.model.track(
                source=image, persist=True, tracker=self.tracker, conf=self.conf,
                iou=self.iou, imgsz=self.imgsz, device=self.device, verbose=False)
        except Exception as error:  # 单帧失败不该打断整条流水线
            self.last_error = f"{type(error).__name__}: {error}"
            return []

        if not results:
            return []
        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None or boxes.xyxy is None or len(boxes.xyxy) == 0:
            return []

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy() if boxes.conf is not None else [1.0] * len(xyxy)
        classes = (boxes.cls.cpu().numpy().astype(int) if boxes.cls is not None
                   else [0] * len(xyxy))
        ids = (boxes.id.cpu().numpy().astype(int) if getattr(boxes, "id", None) is not None
               else [-1] * len(xyxy))
        kpts = None
        if getattr(result, "keypoints", None) is not None and result.keypoints.xy is not None:
            kpts = result.keypoints.data.cpu().numpy()  # (N, K, 3)

        observations: list[HandObservation] = []
        for i in range(len(xyxy)):
            if self._hand_ids and int(classes[i]) not in self._hand_ids:
                continue  # 检测模型里混了别的类，这里只要手
            points = []
            if kpts is not None and i < len(kpts):
                points = [(float(x), float(y), float(v)) for x, y, v in kpts[i]]
            observations.append(HandObservation(
                track_id=int(ids[i]),
                box=tuple(float(v) for v in xyxy[i]),
                conf=float(confs[i]),
                keypoints=points))
        return observations
