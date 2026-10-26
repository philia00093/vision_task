#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""用 YOLO 的零件类别判阶段（分步规则版）。

为什么要有这一版
----------------
图像比对（近邻/SIFT）是在比"装配区长什么样"，受灯光、摆放角度影响，而且托盘
背景会抢戏。而工序本身是有**物理事实**的：哪几个零件已经被拿进装配区了。
现有 `best.pt` 已经输出 base / support / compass / top 四类，直接数类别就能定阶段。

12 张样本实测（装配区在 upload-crop 里是 (60,62,344,344)）：

    步骤     装配区内            物料区还剩下
    step0    base                compass + top (+support 偶尔)
    step1    base                compass + top
    step2    compass             top
    step3    compass             （空）

所以：

    compass 在装配区？ ── 是 ──▶ top 还在物料区？ ── 是 ──▶ step2
          │                          └── 否 ──▶ step3
          └── 否 ──▶ base 在装配区？ ── 是 ──▶ 0/1 分不出来，交给图像分支
                                        └── 否 ──▶ unknown（空盘/过渡）

**0 和 1 的区别是那根斜杆压在盘上**：模型在装配区里只看得到 `base`（杆子把盘面
盖住了），而 `support` 在物料区里又经常漏检（0.88 / 干脆没有），所以 0/1 这一步
只能靠盘面外观，图像近邻那边 6/6 全对，正好互补。

好处还有：2/3 步的时候直接出结果，**不用再跑 130ms 的 SIFT**。
"""

from __future__ import annotations

import time

UNKNOWN = "unknown"


def _center(box):
    return ((box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5)


def _inside(box, rect) -> bool:
    if rect is None:
        return False
    cx, cy = _center(box)
    x, y, w, h = rect
    return x <= cx <= x + w and y <= cy <= y + h


def classes_in(detections, names, rect) -> dict[str, float]:
    """框中心落在 rect 里的类别 → 该类最高置信度。"""
    found: dict[str, float] = {}
    if rect is None:
        return found
    for det in detections or []:
        # 现场传进来的是 PartTracker.tick() 的字典；离线测试用的是 6 元组
        if isinstance(det, dict):
            box, name, conf = det.get("box"), det.get("name"), float(det.get("conf") or 0.0)
            if box is None or name is None:
                continue
            name = str(name)
        else:
            if len(det) < 6:
                continue
            x1, y1, x2, y2, cls, conf = det[:6]
            box = (x1, y1, x2, y2)
            name = names.get(int(cls), str(cls)) if isinstance(names, dict) else str(cls)
            conf = float(conf)
        if not _inside(box, rect):
            continue
        if conf > found.get(name, 0.0):
            found[name] = conf
    return found


def stage_from_parts(detections, names, assembly_rect, material_rect,
                     min_conf: float = 0.30):
    """按零件分布判阶段。

    返回 (stage, reason, score)：
        stage  = "step2" / "step3" / None / "unknown"
                 None 表示"这一步该由图像分支判"（0/1 步）
        reason = 给人看的一句话，进日志方便排查
        score  = 定这个阶段的那个零件的最低置信度
    """
    inside = classes_in(detections, names, assembly_rect)
    outside = classes_in(detections, names, material_rect)
    inside = {k: v for k, v in inside.items() if v >= min_conf}
    outside = {k: v for k, v in outside.items() if v >= min_conf}

    has_wheel = "compass" in inside
    has_base = "base" in inside
    has_lever = "support" in inside
    spare_top = "top" in outside

    if has_wheel:
        if spare_top:
            return ("step2",
                    f"装配区有 compass，物料区还有 top（{outside['top']:.2f}）",
                    min(inside["compass"], outside["top"]))
        return "step3", "装配区有 compass，物料区已空（top 也装上了）", inside["compass"]
    if has_base and has_lever:
        # 盘上压了斜杆 = 第 1 步。现在模型还检不到装配区里的 support
        # （训练样本里它只躺在物料盘上），等补了标注这条分支就自动生效。
        return "step1", "装配区里 base + support", min(inside["base"], inside["support"])
    if has_base:
        # 盘在、杆/轮都不在：0 步（光板）还是 1 步（压了斜杆）—— 交给图像分支
        return None, "装配区只有 base，0/1 交给图像分支", inside["base"]
    if inside:
        return UNKNOWN, f"装配区只有 {sorted(inside)}，不是已知阶段", 0.0
    return UNKNOWN, "装配区是空的", 0.0


# ============================================================================
# 计数版：4 个零件，物料区里少了几个，就是走到了第几步
# ============================================================================
DEFAULT_PART_MEMORY = 4.0        # 一个零件多少秒没在物料区见到才算"被拿走了"


class PartCounter:
    """数零件定阶段。

    工序就是一条直线：零件从**物料区**被拿走、装到**装配区**。所以

        step = (4 个零件里已经离开物料区的个数) - 1

    装配之前：base 拿走 → 0 步；再拿走 support → 1 步；再 compass → 2 步；再 top → 3 步。

    为什么要"记忆"：单帧漏检很常见（杆子细、被托盘边缘挡住），如果一帧没检出就
    当成被拿走了，步数会乱跳。所以每个类别记一个"最后一次在物料区看见的时间"，
    超过 `memory` 秒没再看见，才算真的拿走了。
    """

    def __init__(self, expected, memory=DEFAULT_PART_MEMORY):
        self.expected = list(expected)
        self.memory = float(memory)
        now = time.time()
        # 开机先当作"都在托盘里"，否则第一帧还没检出就全算拿走 → 直接跳到第 3 步
        self._last_seen = {name: now for name in self.expected}

    def reset(self) -> None:
        for name in self._last_seen:
            self._last_seen[name] = 0.0

    def update(self, detections, names, assembly_rect, material_rect, now=None):
        """返回 (stage, reason, score)。stage 是 "step0".."stepN"，认不出就是 "unknown"。"""
        now = time.time() if now is None else now
        in_material = classes_in(detections, names, material_rect)
        in_assembly = classes_in(detections, names, assembly_rect)
        for name in in_material:
            self._last_seen[name] = now

        present = [name for name in self.expected
                   if self._last_seen[name] and (now - self._last_seen[name]) <= self.memory]
        taken = [name for name in self.expected if name not in present]

        # 装配区里已经出现的，也一定是拿走了（比"记忆"更可靠）
        for name in in_assembly:
            if name in self.expected and name not in taken:
                taken.append(name)
                if name in present:
                    present.remove(name)

        if not taken:
            return "unknown", "物料区 4 个零件都在，还没开始装", 0.0
        if "base" not in taken:
            # 盘还没拿，其它零件却不见了 —— 状态不成立
            return "unknown", f"盘还在物料区，但少了 {sorted(taken)}", 0.0

        step = min(len(taken) - 1, len(self.expected) - 1)
        confs = [in_material.get(n, in_assembly.get(n, 0.0)) for n in taken]
        score = min([c for c in confs if c > 0], default=0.0)
        return (f"step{step}",
                f"物料区少了 {sorted(taken)}（{len(present)}/{len(self.expected)} 还在）",
                score)
