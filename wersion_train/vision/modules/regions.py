"""区域检测与越区判断：左侧料盘区、右侧装配区，以及手当前落在哪个区。

三条路径，按可靠性排序：
    1. 绿色分割（默认，推荐）：料盘是纯色工装，用 OpenCV 在 HSV 空间直接分割，
       零样本零训练，左块=料盘区、右块=装配区。见 green_regions.py；
    2. 模型路径：用 wersion_train/configs/data_regions.yaml 训练出的检测模型给出
       tray_area / assembly_area / hand 三类框；
    3. 几何兜底：上面都不用时，用 --split-ratio 在中线附近切两半，
       保证流水线不会因为找不到料盘而停。

越区判断同时给出两种口径，因为两者含义不同：
    · by_box    —— 手框中心落在哪个区（稳，但手大进框一半时就会提前判成进入）
    · by_tip    —— 食指尖关键点落在哪个区（准，但依赖 21 点模型的精度）
界面上两个都显示，便于你按实际装配节拍选择用哪个作为机器人触发信号。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path

from ..common import OUTPUT
from ..models import load_yolo
from .green_regions import GreenConfig, detect_green_regions, save_debug


@dataclass
class Region:
    name: str                                  # tray_area / assembly_area
    box: tuple[float, float, float, float]     # x1, y1, x2, y2
    conf: float
    source: str = "model"                      # model / geometry

    def contains(self, x: float, y: float) -> bool:
        x1, y1, x2, y2 = self.box
        return x1 <= x <= x2 and y1 <= y <= y2

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.box
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0


@dataclass
class RegionState:
    regions: list[Region] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    hand_zone: str = "none"          # tray_area / assembly_area / outside / none
    hand_zone_tip: str = "none"
    step_label: str = ""             # 装配步骤模型的输出（可选）
    step_conf: float = 0.0

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["regions"] = [dict(r, box=list(r["box"])) for r in payload["regions"]]
        return payload


class RegionDetector:
    """区域检测器：绿色分割优先，可切模型，最后几何兜底。"""

    def __init__(self, weights=None, conf: float = 0.35, imgsz: int = 640,
                 device=None, split_ratio: float = 0.5, verbose: bool = False,
                 step_weights=None, step_conf: float = 0.35,
                 use_green: bool = True, green_config: GreenConfig | None = None,
                 debug_dir: str | Path | None = None):
        self.split_ratio = float(split_ratio)
        self.conf, self.imgsz, self.device = conf, imgsz, device
        self.model = None
        self.names: dict[int, str] = {}
        self.step_model = None
        self.step_names: dict[int, str] = {}
        self.step_conf = step_conf
        self.use_green = bool(use_green)
        self.green_config = green_config or GreenConfig()
        self.debug_dir = Path(debug_dir) if debug_dir else None
        self.last_source = "geometry"
        self._debug_saved = 0

        if weights is not None:
            try:
                self.model = load_yolo(weights, task="detect", verbose=verbose)
                self.names = {int(k): str(v) for k, v in (self.model.names or {}).items()}
            except Exception as error:
                print(f"[警告] 区域模型载入失败，改用绿色分割/几何兜底：{error}")
                self.model = None

        if step_weights is not None:
            try:
                self.step_model = load_yolo(step_weights, task="detect", verbose=verbose)
                self.step_names = {int(k): str(v) for k, v in (self.step_model.names or {}).items()}
            except Exception as error:
                print(f"[警告] 步骤模型载入失败，本次不判步骤：{error}")
                self.step_model = None

    @property
    def using_model(self) -> bool:
        return self.model is not None

    @property
    def mode_text(self) -> str:
        if self.use_green:
            return "绿色分割(OpenCV)"
        return "区域模型" if self.using_model else "几何兜底"

    def _green_regions(self, frame) -> tuple[list[Region], list[str]]:
        """绿色分割结果 → 统一 Region 结构（xywh → xyxy）。"""
        greens, warning = detect_green_regions(frame, self.green_config)
        regions = []
        for item in greens:
            x, y, w, h = item.box
            regions.append(Region(item.name, (float(x), float(y), float(x + w), float(y + h)),
                                  1.0, "green"))
        warnings = [warning] if warning else []
        if self.debug_dir is not None and regions and self._debug_saved < 3:
            from .green_regions import green_mask
            path = save_debug(frame, greens, green_mask(frame, self.green_config),
                              self.debug_dir / "green", tag=str(self._debug_saved))
            self._debug_saved += 1
            if self._debug_saved == 1:
                print(f"[调试] 绿色分割视图已保存：{path}")
        return regions, warnings

    def _geometry_regions(self, width: int, height: int) -> list[Region]:
        cut = int(round(width * self.split_ratio))
        return [Region("tray_area", (0.0, 0.0, float(cut), float(height)), 1.0, "geometry"),
                Region("assembly_area", (float(cut), 0.0, float(width), float(height)), 1.0,
                       "geometry")]

    def _classified(self, index: int) -> str:
        name = self.names.get(int(index), str(index))
        low = name.strip().lower()
        # 兼容中英文类名与常见写法
        if low in ("tray", "tray_area", "料盘区", "料盘", "feeding_area", "pallet"):
            return "tray_area"
        if low in ("assembly", "assembly_area", "装配区", "装配"):
            return "assembly_area"
        if low in ("hand", "hands", "手"):
            return "hand"
        return low

    def update(self, frame) -> RegionState:
        height, width = frame.shape[:2]
        state = RegionState()

        if self.use_green:
            regions, warnings = self._green_regions(frame)
            state.regions += regions
            state.warnings += warnings
            if regions:
                self.last_source = "green"
        elif self.model is not None:
            result = self.model.predict(source=frame, conf=self.conf, imgsz=self.imgsz,
                                        device=self.device, verbose=False)[0]
            boxes = getattr(result, "boxes", None)
            if boxes is not None and boxes.xyxy is not None:
                xyxy = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy() if boxes.conf is not None else [1.0] * len(xyxy)
                classes = (boxes.cls.cpu().numpy().astype(int) if boxes.cls is not None
                           else [0] * len(xyxy))
                grouped: dict[str, tuple[tuple[float, float, float, float], float]] = {}
                for i in range(len(xyxy)):
                    label = self._classified(int(classes[i]))
                    if label not in ("tray_area", "assembly_area"):
                        continue
                    box = tuple(float(v) for v in xyxy[i])
                    # 同类多框时取面积最大的那个，避免小误检干扰
                    area = (box[2] - box[0]) * (box[3] - box[1])
                    if label not in grouped:
                        grouped[label] = (box, float(confs[i]))
                    else:
                        old = grouped[label][0]
                        if area > (old[2] - old[0]) * (old[3] - old[1]):
                            grouped[label] = (box, float(confs[i]))
                for label, (box, conf) in grouped.items():
                    state.regions.append(Region(label, box, conf, "model"))
                if grouped:
                    self.last_source = "model"

        found = {r.name for r in state.regions}
        if not {"tray_area", "assembly_area"} <= found:
            missing = {"tray_area", "assembly_area"} - found
            state.warnings.append("未识别到：" + "、".join(sorted(missing)) + "（已用几何兜底补齐）")
            for region in self._geometry_regions(width, height):
                if region.name in missing:
                    state.regions.append(region)

        state.regions.sort(key=lambda r: 0 if r.name == "tray_area" else 1)
        self._sanity_check(state, width)
        if self.step_model is not None:
            self._update_step(frame, state)
        return state

    def _sanity_check(self, state: RegionState, width: int) -> None:
        """料盘区必须在装配区左侧——这是本工位的硬约束，违反就报警。"""
        trays = [r for r in state.regions if r.name == "tray_area"]
        assembles = [r for r in state.regions if r.name == "assembly_area"]
        if not trays or not assembles:
            return
        tray_x = trays[0].center[0]
        assembly_x = assembles[0].center[0]
        if tray_x >= assembly_x:
            state.warnings.append(
                f"区域左右颠倒：料盘区中心 x={tray_x:.0f} 未在装配区 x={assembly_x:.0f} 左侧")
        # 区域面积占比异常（例如模型把整幅画面框成一块）也提示一下
        tray_w = trays[0].box[2] - trays[0].box[0]
        if tray_w > width * 0.95:
            state.warnings.append("料盘区框几乎覆盖整幅画面，疑似误检，建议检查标注")

    def _update_step(self, frame, state: RegionState) -> None:
        result = self.step_model.predict(source=frame, conf=self.step_conf, imgsz=self.imgsz,
                                         device=self.device, verbose=False)[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None or boxes.conf is None or len(boxes.conf) == 0:
            return
        best = int(boxes.conf.argmax())
        state.step_label = self.step_names.get(int(boxes.cls[best]), "?")
        state.step_conf = float(boxes.conf[best])

    def locate_hand(self, state: RegionState, hand) -> tuple[str, str]:
        """返回 (按手框中心判定, 按食指尖判定)。"""
        by_box = "outside"
        center = hand.center
        for region in state.regions:
            if region.contains(*center):
                by_box = region.name
                break
        by_tip = by_box
        tip = hand.tip(8) if hand.keypoints else None
        if tip is not None:
            by_tip = "outside"
            for region in state.regions:
                if region.contains(*tip):
                    by_tip = region.name
                    break
        return by_box, by_tip


def save_baseline(frame, state: RegionState, tag: str = "") -> Path:
    """保存一份区域基线 JSON：标注/训练前先用它确认几何兜底的位置对不对。"""
    OUTPUT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = OUTPUT / f"region_baseline_{stamp}{('_' + tag) if tag else ''}.json"
    payload = {"time": stamp, "frame_size": [frame.shape[1], frame.shape[0]],
               "state": state.to_dict()}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_baseline(path: str | Path) -> RegionState:
    """读回基线 JSON，作为固定区域使用（相机不动时最省算力的做法）。"""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    raw = payload.get("state", payload)
    state = RegionState(warnings=list(raw.get("warnings", [])))
    for item in raw.get("regions", []):
        state.regions.append(Region(item["name"], tuple(item["box"]),
                                    float(item.get("conf", 1.0)),
                                    item.get("source", "baseline")))
    return state
