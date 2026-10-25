#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""阶段样本对照窗口 —— 用来人眼指出 step2/step3 到底差在哪。

    python tools/inspect_steps.py

窗口里 4 列 = step0 / step1 / step2 / step3，每列显示顶部 3 张（可切换）。
每张图上叠了**坐标网格**（每 50 px 一条线，标的是 900x500 裁剪图里的原始坐标）。

操作：
    ← →      切换当前列里显示第几张样本
    1 2 3 4  直接把光标列切到对应阶段
    鼠标左键  点一下 = 打十字标记
    鼠标拖拽  按住拖动 = 画矩形框（松开即记录，坐标写进 output/roi_boxes.json）
    r        清掉所有标记
    s        把当前窗口存成 /tmp/steps_view.png
    q / ESC  退出
"""

from __future__ import annotations

import glob
import json
import os
import re
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STEP_DIR = os.path.join(ROOT, "vision_yolo_train", "images", "steps")

COL_W, COL_H = 440, 245          # 单张显示尺寸
HEADER = 34
GRID = 50                        # 网格间距（原始坐标）


def label_of(path: str) -> str:
    m = re.search(r"step[_-]?(\d+)", os.path.basename(path), re.IGNORECASE)
    return f"step{m.group(1)}" if m else os.path.basename(path)


def load_groups():
    groups: dict[str, list[str]] = {}
    for path in sorted(glob.glob(os.path.join(STEP_DIR, "*.jpg")) + glob.glob(os.path.join(STEP_DIR, "*.png"))):
        groups.setdefault(label_of(path), []).append(path)
    return dict(sorted(groups.items()))


def draw_grid(img, scale_x, scale_y):
    """在缩放后的图上画原始坐标网格。"""
    h, w = img.shape[:2]
    for x in range(0, 901, GRID):
        sx = int(x * scale_x)
        if sx >= w:
            break
        cv2.line(img, (sx, 0), (sx, h), (60, 60, 60), 1)
        if x % 100 == 0:
            cv2.putText(img, str(x), (sx + 2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 200, 255), 1)
    for y in range(0, 501, GRID):
        sy = int(y * scale_y)
        if sy >= h:
            break
        cv2.line(img, (0, sy), (w, sy), (60, 60, 60), 1)
        if y % 100 == 0:
            cv2.putText(img, str(y), (2, sy - 2 if sy > 12 else sy + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 200, 255), 1)


def main() -> int:
    groups = load_groups()
    if not groups:
        raise SystemExit(f"{STEP_DIR} 里没有图片")
    labels = list(groups.keys())
    print("阶段:", {k: len(v) for k, v in groups.items()})
    print("操作：← → 切换样本 | 1-4 选阶段 | 左键打点（打印坐标）| r 清除 | s 存图 | q 退出")

    idx = {label: 0 for label in labels}
    active = 0
    marks: list[tuple[int, int, int]] = []          # (列, x, y) 点标记
    boxes: list[tuple[int, int, int, int, int]] = []  # (列, x1, y1, x2, y2)
    drag = {"col": -1, "x1": 0, "y1": 0, "x2": 0, "y2": 0, "on": False}

    def to_source(x, y):
        """窗口坐标 → (列, 原始坐标 x, 原始坐标 y)。"""
        col = x // COL_W
        if col >= len(labels):
            return None
        sx = (x % COL_W) / (COL_W / 900.0)
        sy = (y - HEADER) / (COL_H / 500.0)
        if sy < 0:
            return None
        return col, int(max(0, min(sx, 900))), int(max(0, min(sy, 500)))

    def on_mouse(event, x, y, flags, _param):
        hit = to_source(x, y)
        if hit is None:
            return
        col, sx, sy = hit
        if event == cv2.EVENT_LBUTTONDOWN:
            drag.update(col=col, x1=sx, y1=sy, x2=sx, y2=sy, on=True)
        elif event == cv2.EVENT_MOUSEMOVE and drag["on"]:
            drag["x2"], drag["y2"] = sx, sy
        elif event == cv2.EVENT_LBUTTONUP and drag["on"]:
            drag["on"] = False
            x1, x2 = sorted((drag["x1"], sx))
            y1, y2 = sorted((drag["y1"], sy))
            if x2 - x1 < 4 or y2 - y1 < 4:          # 太小的当点击
                marks.append((col, sx, sy))
                print(f"[点击] {labels[col]}  x={sx}  y={sy}")
            else:
                boxes.append((col, x1, y1, x2, y2))
                print(f"[框] {labels[col]}  ({x1},{y1}) → ({x2},{y2})   宽 {x2-x1} 高 {y2-y1}")
                os.makedirs(os.path.join(ROOT, "output"), exist_ok=True)
                with open(os.path.join(ROOT, "output", "roi_boxes.json"), "w", encoding="utf-8") as handle:
                    json.dump([{"stage": labels[c], "x1": a, "y1": b, "x2": c2, "y2": d}
                               for (c, a, b, c2, d) in boxes], handle, ensure_ascii=False, indent=2)

    win = "step samples  (click to mark)"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(win, on_mouse)

    while True:
        canvas = np.full((HEADER + COL_H, COL_W * len(labels), 3), 30, np.uint8)
        for col, label in enumerate(labels):
            paths = groups[label]
            path = paths[idx[label] % len(paths)]
            img = cv2.imread(path)
            if img is None:
                continue
            img = cv2.resize(img, (COL_W, COL_H), interpolation=cv2.INTER_AREA)
            scale_x, scale_y = COL_W / 900.0, COL_H / 500.0
            draw_grid(img, scale_x, scale_y)
            for (bcol, bx1, by1, bx2, by2) in boxes:
                if bcol == col:
                    cv2.rectangle(img, (int(bx1*scale_x), int(by1*scale_y)),
                                  (int(bx2*scale_x), int(by2*scale_y)), (0, 255, 0), 2)
                    cv2.putText(img, f"({bx1},{by1})", (int(bx1*scale_x)+3, int(by1*scale_y)+14),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            if drag["on"] and drag["col"] == col:
                cv2.rectangle(img, (int(drag["x1"]*scale_x), int(drag["y1"]*scale_y)),
                              (int(drag["x2"]*scale_x), int(drag["y2"]*scale_y)), (0, 255, 255), 1)
            for (mcol, mx, my) in marks:
                if mcol == col:
                    px, py = int(mx * scale_x), int(my * scale_y)
                    cv2.drawMarker(img, (px, py), (0, 0, 255), cv2.MARKER_CROSS, 18, 2)
                    cv2.putText(img, f"{mx},{my}", (px + 6, py - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            canvas[HEADER:HEADER + COL_H, col * COL_W:(col + 1) * COL_W] = img
            title = f"{label}  {idx[label] + 1}/{len(paths)}   {os.path.basename(path)[-16:]}"
            color = (0, 255, 0) if col == active else (200, 200, 200)
            cv2.putText(canvas, title, (col * COL_W + 8, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            cv2.line(canvas, (col * COL_W, 0), (col * COL_W, HEADER + COL_H), (80, 80, 80), 1)

        cv2.imshow(win, canvas)
        key = cv2.waitKey(30) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("r"):
            marks.clear()
            boxes.clear()
        elif key == ord("s"):
            cv2.imwrite("/tmp/steps_view.png", canvas)
            print("已保存 /tmp/steps_view.png")
        elif key in (81, 83, 2, 3):         # 左右方向键
            step = -1 if key in (81, 2) else 1
            active = (active + step) % len(labels)
            idx[labels[active]] += 1
        elif ord("1") <= key <= ord("4"):
            active = min(key - ord("1"), len(labels) - 1)
            idx[labels[active]] += 1

    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
