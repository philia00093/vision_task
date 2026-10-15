"""视觉流水线：把区域检测 + 手部跟踪 + 绘制串成一条可复用的链。

调用方（run_pipeline.py / 以后的机器人主控）只需要：
    pipeline = VisionPipeline(...)
    frame, state, hands = pipeline.process(bgr_frame)
"""

from __future__ import annotations

import csv
import time
from datetime import datetime
from pathlib import Path

from ..common import OUTPUT, WERSION_TRAIN
from .draw import annotate
from .hand_tracker import HandTracker
from .regions import RegionDetector, RegionState, load_baseline, save_baseline


class VisionPipeline:
    def __init__(self, region_weights=None, hand_weights=None, step_weights=None,
                 tracker="wersion_train/configs/trackers/bytetrack.yaml", conf=0.35, hand_conf=0.35,
                 imgsz=640, device=None, split_ratio=0.5, baseline=None,
                 show_keypoints=True, use_green=True, green_config=None, debug_dir=None,
                 log_events=False, verbose=False):
        self.show_keypoints = show_keypoints
        self.baseline_state: RegionState | None = load_baseline(baseline) if baseline else None
        self.detector = RegionDetector(weights=region_weights, conf=conf, imgsz=imgsz,
                                       device=device, split_ratio=split_ratio,
                                       verbose=verbose, step_weights=step_weights,
                                       use_green=use_green, green_config=green_config,
                                       debug_dir=debug_dir)
        self.tracker = HandTracker(weights=hand_weights, tracker=tracker, conf=hand_conf,
                                   imgsz=imgsz, device=device, verbose=verbose)
        self.last_hands: list = []
        self.event_log = None
        self._writer = None
        self._last_zone: dict[int, str] = {}
        if log_events:
            self._open_event_log()

    # ---------------------------------------------------------------- 生命周期
    def _open_event_log(self) -> None:
        directory = WERSION_TRAIN / "events"
        directory.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.event_log = directory / f"events_{stamp}.csv"
        self._writer = self.event_log.open("w", newline="", encoding="utf-8-sig")
        self._csv = csv.writer(self._writer)
        self._csv.writerow(["time", "frame", "track_id", "from_zone", "to_zone",
                            "step_label", "step_conf", "hand_conf"])

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None

    # ------------------------------------------------------------------- 单帧
    def process(self, frame, frame_index: int = 0, fps: float = 0.0):
        """返回 (标注图, RegionState, hands)。"""
        started = time.perf_counter()

        if self.baseline_state is not None:
            # 相机位置固定时用基线区域，省掉一次检测推理
            import copy
            state = copy.deepcopy(self.baseline_state)
            if self.detector.step_model is not None:
                self.detector._update_step(frame, state)
        else:
            state = self.detector.update(frame)

        hands = self.tracker.update(frame)
        self.last_hands = hands

        zone_by_box, zone_by_tip = "none", "none"
        if hands:
            # 取置信度最高的那只手作为"当前操作手"
            main = max(hands, key=lambda h: h.conf)
            zone_by_box, zone_by_tip = self.detector.locate_hand(state, main)
            for hand in hands:
                box_zone, _ = self.detector.locate_hand(state, hand)
                self._record_transition(hand, box_zone, state, frame_index)
        state.hand_zone, state.hand_zone_tip = zone_by_box, zone_by_tip

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        canvas = annotate(frame, state, hands, fps=fps, show_keypoints=self.show_keypoints,
                          extra_lines=[f"推理 {elapsed_ms:.0f} ms"])
        return canvas, state, hands

    def _record_transition(self, hand, zone: str, state: RegionState, frame_index: int) -> None:
        """记录手从一个区到另一个区的跨越事件（以后可以直接作为机器人触发信号）。"""
        key = hand.track_id if hand.track_id >= 0 else -1
        previous = self._last_zone.get(key)
        if previous is None or previous == zone:
            self._last_zone[key] = zone
            return
        self._last_zone[key] = zone
        if self._writer is not None:
            self._csv.writerow([datetime.now().strftime("%H:%M:%S.%f")[:-3], frame_index, key,
                                previous, zone, state.step_label, f"{state.step_conf:.3f}",
                                f"{hand.conf:.3f}"])
            self._writer.flush()

    # ------------------------------------------------------------------ 基线
    def save_baseline(self, frame, state: RegionState, tag: str = "") -> Path:
        return save_baseline(frame, state, tag)
