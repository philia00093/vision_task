"""阶段识别（装配步骤）—— **这是你要动手写的文件**。

这个文件被 `test/run_test.py` 每帧调用一次（`--stage-mode parts` 或 `both` 时才调），
用"当前检出了哪些零件、它们落在哪个区域"推断"装配做到第几步"。
默认实现很粗（按零件组合硬映射），按你的规则替换 `update()` 即可。

另一条路是**模板匹配**（`edge/stage_match.py`，`--stage-mode template`，默认）：
它不看零件框，只把装配区那块裁剪图旋转一圈跟参考图比相似度。
两条路互相独立，可以同时开（`--stage-mode both`）对照看谁更稳。

============================ 数据契约（只依赖这两个参数） ============================

`zones`:  两个区域的列表（`edge/zone_tracker.py` 的 ZoneTracker.zones），每项是 dict：
    {
      "name":          "装配区" | "物料区",        # 左=装配区，右=物料区
      "box":           (x1, y1, x2, y2) | None,   # 原画面像素坐标；None=还没锁定
      "anchor":        (x1, y1, x2, y2) | None,   # 锁定时的框（不会变）
      "baseline":      float,                     # 该区域"完全露出"时的暗色像素量
      "visible_ratio": float,                     # 当前可见暗色量 / baseline（1.0=没被挡）
      "occluded":      bool,                      # 是否被手/零件挡住（取料中的判据）
      "cover":         float,                     # 框内"亮像素"占比
      "cover_delta":   float,                     # 亮占比相对最近静止水平的增量
    }

`detections`: 当前帧的零件列表（`edge/parts.py` 的 PartTracker.tick()），每项是 dict：
    {
      "box":       (x1, y1, x2, y2),   # 原画面像素坐标（卡尔曼平滑过）
      "name":      "base" | "support" | "compass" | "top",
      "conf":      float,
      "track_id":  int,                # 同一零件跨帧不变
      "trail":     deque[(cx, cy)],    # 轨迹点（最近 N 帧的中心）
      "predicted": bool,               # True = 这一帧模型没检出、位置是卡尔曼补的
    }

返回值：一个字符串（会显示在状态栏，变化时也会打印到控制台）。
    想返回更丰富的信息（比如进度、置信度），把返回值改成 dict，再同步改 run_test.py 里
    显示它的那两行即可。

=============================== 默认实现（可整体替换） ===============================

思路：只看"装配区"里已经出现了哪些类别，取满足条件的最高阶段。
    {}                                    → step_0（空盘）
    {base}                                → step_1
    {base, support}                       → step_2
    {base, support, compass}              → step_3
    {base, support, compass, top}          → complete

不够严谨的地方（你按实际工艺补）：
    · 没有顺序约束：即使 base 和 support 是同时拍到的，也会判成 step_2；
    · 没有时间约束：手抖一下多检出/漏检出就跳步；
    · 没有用到位置：零件是在装配区还是在物料区，默认实现只用"在装配区内"；
    · 没有置信度门槛：conf 0.35 的低分检出也算数。
"""

from __future__ import annotations

from typing import Iterable

# 装配顺序（从空盘到完整），按你的工艺改这个表即可
STAGES = (
    ("step_0", frozenset()),
    ("step_1", frozenset({"base"})),
    ("step_2", frozenset({"base", "support"})),
    ("step_3", frozenset({"base", "support", "compass"})),
    ("complete", frozenset({"base", "support", "compass", "top"})),
)


class StepRecognizer:
    """把"每帧的零件 + 区域"变成"当前阶段"。默认只看装配区里有哪些零件。"""

    def __init__(self, zone_name: str = "装配区", min_conf: float = 0.35,
                 use_predicted: bool = False):
        self.zone_name = zone_name
        self.min_conf = float(min_conf)
        self.use_predicted = bool(use_predicted)   # 卡尔曼补出来的框要不要算
        self.step = "step_0"
        self.history: list[str] = []               # 每次变化记一条，方便回看
        self._last_reported = None
        self._warned = False

    # ------------------------------------------------------------ 每帧调用
    def update(self, zones: Iterable[dict], detections: Iterable[dict]) -> str:
        """用本帧的（区域, 零件）推断阶段。返回阶段名。"""
        zone = self._find_zone(zones)
        if zone is None or zone.get("box") is None:
            return self._set(self.step if not self._warned else "等待区域锁定")

        inside = self._classes_inside(zone["box"], detections)
        self.step = self._match_stage(inside)
        if not self.history or self.history[-1] != self.step:
            self.history.append(self.step)
        return self._set(self.step)

    # ------------------------------------------------------------ 可以按需改这两步
    def _classes_inside(self, box, detections) -> set[str]:
        """装配区框里出现了哪些类别。"""
        x1, y1, x2, y2 = box
        found = set()
        for det in detections:
            if det.get("predicted") and not self.use_predicted:
                continue
            if float(det.get("conf", 0.0)) < self.min_conf:
                continue
            bx1, by1, bx2, by2 = det["box"]
            cx, cy = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0
            if x1 <= cx <= x2 and y1 <= cy <= y2:      # 中心点落在框内就算
                found.add(str(det.get("name", "")))
        return found

    def _match_stage(self, present: set[str]) -> str:
        """取"所需零件都被包含"的最高阶段。"""
        stage = "step_0"
        for name, needed in STAGES:
            if needed <= present:
                stage = name
        return stage

    # ------------------------------------------------------------ 内部小工具
    def _find_zone(self, zones):
        for zone in zones or []:
            if str(zone.get("name")) == self.zone_name:
                return zone
        return None

    def _set(self, text: str) -> str:
        if text != self._last_reported:
            self._last_reported = text
            print(f"[阶段] {text}")
        return text
