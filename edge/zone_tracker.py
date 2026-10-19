"""区域状态机：锁定两个料盘、判遮挡、产出"取料中/装配中"。

连续若干帧稳定认出两个料盘后**锁定并把框写进 zones.json**，之后框不再动，只量遮挡 —— 这是不抖的关键。"""

from __future__ import annotations

import numpy as np
import json
from collections import deque
from pathlib import Path

from edge.constants import DEFAULT_COVER_DELTA, DEFAULT_COVER_MARGIN, DEFAULT_COVER_WINDOW, DEFAULT_TRAY_MIN_SHARE, DEFAULT_ZONE_LOCK_FRAMES
from edge.zones import pick_tray_boxes, split_touching_trays, tray_pair_ok


def parse_zones(text) -> list:
    """解析 --zones "x1,y1,x2,y2;x1,y1,x2,y2" → [(x1,y1,x2,y2), (x1,y1,x2,y2)]（左、右）。"""
    boxes = []
    for chunk in str(text).split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        values = [int(float(v)) for v in chunk.replace("，", ",").split(",") if v.strip()]
        if len(values) != 4:
            raise SystemExit(f"--zones 每段要 4 个数字（x1,y1,x2,y2），收到 {chunk!r}")
        x1, y1, x2, y2 = values
        boxes.append((min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)))
    if boxes and len(boxes) != 2:
        raise SystemExit(f"--zones 需要正好两段（左=装配区，右=物料区），收到 {len(boxes)} 段")
    return boxes

def load_zones(path: Path) -> list:
    """读 zones.json；读不出或格式不对就返回空（退回自动识别）。"""
    try:
        import json
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return []
    boxes = []
    for item in payload.get("zones", []):
        box = item.get("box") if isinstance(item, dict) else item
        if isinstance(box, (list, tuple)) and len(box) == 4:
            boxes.append(tuple(int(v) for v in box))
    return boxes if len(boxes) == 2 else []

def save_zones(path: Path, boxes) -> bool:
    """把锁定的两个区域写到 zones.json，下次启动直接读。"""
    import json
    try:
        Path(path).write_text(json.dumps({
            "note": "左=装配区，右=物料区；改这里可以手工微调，或删掉让程序重新自动识别",
            "zones": [{"name": name, "box": [int(v) for v in box]}
                      for name, box in zip(ZoneTracker.NAMES, boxes)],
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        return True
    except OSError as error:
        print(f"[警告] 区域文件写不进去（{error}），本次仍可用")
        return False

class ZoneTracker:
    """跟踪固定两个区域：**左=装配区，右=物料区**（用户确认物料区在右侧）。

    料盘改成黑色后，原来"取最左两块暗块"不再可靠：两个料盘挨在一起时暗色掩码会把
    它们连成**一个**连通域，于是两个区会落到同一块（或被左边缘的干扰块抢走）。现在的流程：

        1. 暗色连通域 → 最大那块若明显偏宽，从"缝"处切成两块（split_touching_trays）
        2. 按料盘特征筛出左右两个区（pick_tray_boxes）
        3. **连续 lock_frames 帧都稳定认出两个区 → 锁定**，把框和锚点固定下来；
           锁定后只按锚点把每帧的碎块归到左/右，框不再乱跳
        4. 锁定的框写进 zones.json，下次启动直接读；也可以用 --zones 手工指定

    遮挡判断：以"锁定/首次检测到该区域时的掩码像素数"为基准，之后每帧比较当前可见
    像素数，低于 基准 × --occlude-ratio 判为被遮挡 —— 手伸进料盘会挡住料盘的暗色，
    可见量骤降，所以"取料中/装配中"就是这么判出来的。

    状态规则：只按**物料区**判断 —— 被遮挡显示"取料中"，否则"装配中"。
    """

    NAMES = ("装配区", "物料区")          # 左 → 右

    def __init__(self, occlude_ratio=0.75, zones=None, zones_file=None,
                 lock_frames=DEFAULT_ZONE_LOCK_FRAMES,
                 cover_margin=DEFAULT_COVER_MARGIN, cover_delta=DEFAULT_COVER_DELTA,
                 cover_window=DEFAULT_COVER_WINDOW):
        self.occlude_ratio = float(occlude_ratio)
        self.lock_frames = max(int(lock_frames), 1)
        self.cover_margin = float(cover_margin)
        self.cover_delta = float(cover_delta)
        self.cover_window = max(int(cover_window), 2)
        self.zones_file = Path(zones_file) if zones_file else None
        self.zones: list[dict] = [
            {"name": name, "box": None, "anchor": None, "baseline": 0.0,
             "occluded": False, "visible_ratio": 1.0,
             "cover": 0.0, "cover_delta": 0.0,
             "cover_hist": deque(maxlen=self.cover_window)}
            for name in self.NAMES
        ]
        self.frame = 0
        self.hits = 0
        self.locked = False
        if zones:
            self.lock(list(zones), quiet=True)

    # ------------------------------------------------------------ 锁定
    def lock(self, boxes, quiet=False) -> None:
        """把两个区域的框固定下来（不再自动重认）。"""
        for zone, box in zip(self.zones, boxes):
            box = tuple(int(v) for v in box[:4])
            zone["box"] = box
            zone["anchor"] = box
            # 基准必须跟"这个框"绑定：锁定前框每帧都在变，那时记的像素量作废，
            # 否则框一变小就会算成"可见量骤降"，一直误判被遮挡。
            zone["baseline"] = 0.0
            zone["visible_ratio"] = 1.0
            zone["occluded"] = False
            # 静止水平也要跟着框重来：锁定前量的亮占比是别的框，不能拿来当基准
            zone["cover_hist"].clear()
            zone["cover"] = 0.0
            zone["cover_delta"] = 0.0
        self.locked = True
        if not quiet:
            print(f"[区域] 已锁定 装配区{self.NAMES[0]}={self.zones[0]['box']} "
                  f"{self.NAMES[1]}={self.zones[1]['box']}")
            if self.zones_file and save_zones(self.zones_file, [z["box"] for z in self.zones]):
                print(f"       已写入 {self.zones_file}（下次启动直接读，不用再自动识别）")

    def _detect_trays(self, boxes, mask, frame_area) -> list:
        """一帧的自动识别：过滤 → 切缝 → 挑左右两块 → 过闸（像不像两个料盘）。"""
        floor = DEFAULT_TRAY_MIN_SHARE * frame_area
        real = [b for b in boxes if b[4] >= floor]
        if not real:
            return []
        merged = split_touching_trays(mask, real)
        picked = pick_tray_boxes(merged, frame_area)
        if len(picked) == 2 and tray_pair_ok(picked[0], picked[1], frame_area):
            return picked
        return []

    # ------------------------------------------------------------ 每帧
    @staticmethod
    def _to_full(box, scale):
        """识别尺度 → 原分辨率（--zone-width 把掩码缩小过，框要换算回去）。"""
        if not scale or scale == 1.0:
            return tuple(box)
        inv = 1.0 / scale
        return (box[0] * inv, box[1] * inv, box[2] * inv, box[3] * inv,
                box[4] * inv * inv)

    def _measure(self, zone, mask, v_channel=None, scale=1.0) -> None:
        """按当前框量遮挡：两路判据取"或"，任一路说有东西伸进去就算遮挡。

        路一 暗色可见量：手挡住黑料盘 → 暗色像素骤降（相对历史最大量）。
        路二 亮像素增量：手和木桌比料盘亮 → 框内亮占比抬头（相对最近静止水平）。
        两路一起用是因为单靠路一会漏判——手只盖住料盘一角时，暗色量可能只掉到
        0.8 左右，过不了阈值；而这时亮占比已经明显抬起来了（实测 0.16~0.61）。

        zone["box"] 存的是原分辨率坐标，mask/v_channel 可能是缩小过的（Pi 上省 CPU），
        所以这里按 scale 折算回识别尺度再取像素。
        """
        box = zone["box"]
        if box is None:
            return
        x1, y1, x2, y2 = (int(v * scale) for v in box)
        patch = mask[max(0, y1):y2, max(0, x1):x2]
        visible = float(np.count_nonzero(patch)) if patch.size else 0.0
        # 基准取**历史最大值** = "料盘完全露出来"时的暗色量。这样即使开机那一帧
        # 手正好在料盘里，之后手拿出来基准也会自己涨上去，不会出现"永远判不出遮挡"。
        if visible > zone["baseline"]:
            zone["baseline"] = visible
        # 上限钳到 1.0：框是碎块并集，可能比基准大，比例不该超过 1
        zone["visible_ratio"] = min(visible / max(zone["baseline"], 1.0), 1.0)
        dark_hit = zone["visible_ratio"] < self.occlude_ratio

        cover_hit = False
        if v_channel is not None:
            patch_v = v_channel[max(0, y1):y2, max(0, x1):x2]
            if patch_v.size:
                # 亮阈值跟着"料盘自己有多亮"走：现场整体偏暗时，固定 140 会把伸进来的手
                # 也判成暗的，两路判据会同时失效。夹在 70~220 之间避免极端光照下失控。
                tray_v = float(np.median(patch_v))
                bright_v = min(max(tray_v + self.cover_margin, 70.0), 220.0)
                cover = float(np.count_nonzero(patch_v > bright_v)) / patch_v.size
            else:
                cover = 0.0
            history = zone["cover_hist"]
            history.append(cover)
            # 静止水平 = 最近窗口里的最小值。用滑动窗口而不是历史最小值，
            # 是为了让"装配好一直放在料盘里的零件"过一会儿就不再算遮挡。
            resting = min(history)
            zone["cover"] = cover
            zone["cover_delta"] = cover - resting
            cover_hit = zone["cover_delta"] >= self.cover_delta
        zone["occluded"] = bool(dark_hit or cover_hit)

    def update(self, boxes, mask, frame_area, v_channel=None, scale=1.0) -> list[dict]:
        """boxes/mask/frame_area 都是"识别尺度"上的量（可能被 --zone-width 缩小过）。

        锁定之后**不再动框**：两个料盘挨在一起时掩码经常连成一块，若还按"最近锚点"
        归属，那个横跨两盘的巨块会把某个区的框撑大甚至让左右两个区互换。
        框固定下来只量遮挡，才是这个场景要的稳定行为。
        存进 zones 的框一律换算回原分辨率，方便绘制和写 zones.json。
        """
        self.frame += 1
        if self.locked:
            for zone in self.zones:
                self._measure(zone, mask, v_channel, scale)
            return self.zones

        found = self._detect_trays(boxes, mask, frame_area)
        if len(found) == 2:
            self.hits += 1
            if self.hits >= self.lock_frames:         # 连续多帧都稳定 → 锁定
                self.lock([self._to_full(b, scale) for b in found])
                for zone in self.zones:
                    self._measure(zone, mask, v_channel, scale)
                return self.zones
        else:
            self.hits = 0

        # 还没锁定：先用当前候选画出来，并按锚点归到左右（锚点为空则退化为左/右顺序）。
        # 这一阶段**不判遮挡**：框还在变，量出来的像素数没有可比性。
        found_full = [self._to_full(b, scale) for b in found]
        for zone, group in zip(self.zones, self._group_by_anchor(found_full)):
            if group:
                zone["box"] = (min(int(b[0]) for b in group),
                               min(int(b[1]) for b in group),
                               max(int(b[2]) for b in group),
                               max(int(b[3]) for b in group))
        return self.zones

    def _group_by_anchor(self, candidates) -> list[list]:
        """把候选块按"离哪个锚点近"归到左/右两个区，手切断产生的碎块也回到本区。

        用锚点（锁定/首次检测到的位置）而不是简单排序，是因为手遮挡会把料盘的暗区
        切成几块，纯按 x 排序会把切下来的碎块算到隔壁区域去。
        """
        anchors = [z["anchor"] for z in self.zones]
        if all(a is None for a in anchors):       # 还没有锚点：退化为左/右两块
            return [candidates[:1], candidates[1:2]]
        if any(a is None for a in anchors):       # 只认出一个：全给已锚定的那个
            known = anchors.index(next(a for a in anchors if a is not None))
            return [[c for c in candidates], []] if known == 0 else [[], list(candidates)]

        groups: list[list] = [[], []]
        for box in candidates:
            cx = (box[0] + box[2]) / 2.0
            distances = [abs(cx - (a[0] + a[2]) / 2.0) for a in anchors]
            groups[distances.index(min(distances))].append(box)
        return groups

    def drawable(self) -> list[dict]:
        """只返回已经有框的区域，供绘制使用。

        区域是在第一次检测到色块时创建的，那一帧 box 还是 None；
        直接解包 zone["box"] 会抛 TypeError: cannot unpack non-iterable NoneType。
        """
        return [z for z in self.zones if z["box"] is not None]

    def region(self, name: str = "装配区"):
        """某个区域的裁剪窗口 (x, y, w, h)；这个区域还没有框就返回 None。

        阶段识别只关心"装配区这一块图长什么样"，用它拿窗口，
        不用自己再遍历 zones 拼坐标。
        """
        for zone in self.zones:
            if zone["name"] == name and zone["box"] is not None:
                x1, y1, x2, y2 = zone["box"]
                return (int(x1), int(y1), int(x2 - x1), int(y2 - y1))
        return None

    def state_text(self, zone) -> str:
        """单个区域的遮挡状态（仅用于内部判断，不直接显示）。"""
        if not zone["occluded"]:
            return ""
        return "取料中" if zone["name"] == "物料区" else (
            "装配中" if zone["name"] == "装配区" else f"{zone['name']}被遮挡")

    def machine_state(self) -> str:
        """工位状态：**只按物料区判断**。

        逻辑（按你的说明）：物料区被手遮住 = 正在取料，可以据此确认装配是否正确
        → 显示"取料中"；物料区没有被遮 → 显示"装配中"。
        """
        material = next((z for z in self.zones if z["name"] == "物料区"), None)
        if material is None or material["box"] is None:
            return "等待区域锁定"
        return "取料中" if material["occluded"] else "装配中"
