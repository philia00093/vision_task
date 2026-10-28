#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""电脑上看罗盘正反 —— 用树莓派回传的画面（不占树莓派 CPU，也不用电脑摄像头）。

数据来源就是本机正在跑的识别服务：
    GET /frame.jpg   树莓派刚上传的原始帧
    GET /state       那一帧的检测框 + 电脑端算好的罗盘正反（compass 字段）

所以屏幕上看到的就是**树莓派相机真实视角**，判定也在电脑上完成。

按键：q/ESC 退出，s 存样本（原图 + 标注图）
用法：python tools/compass_view.py --url http://127.0.0.1:8080
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
NAMES = {0: "base", 1: "support", 2: "compass", 3: "top", 4: "hand"}
COLORS = {0: (200, 200, 0), 1: (0, 200, 200), 2: (0, 200, 0), 3: (200, 0, 200), 4: (120, 120, 120)}


def get(url: str, timeout: float = 2.0) -> bytes:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return resp.read()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--hz", type=float, default=6.0)
    ap.add_argument("--save-dir", default=str(ROOT / "output/snapshots/compass_live"))
    args = ap.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    n_save = len(list(save_dir.glob("*.jpg")))

    win = "compass view (Pi 画面)  [q]uit  [s]ave"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 900, 500)

    state: dict = {}
    raw = None
    last_print = 0.0
    pause = 1.0 / max(1e-3, args.hz)
    blank = np.zeros((500, 900, 3), np.uint8)

    while True:
        t0 = time.time()
        try:
            state = json.loads(get(args.url + "/state").decode("utf-8"))
            raw = get(args.url + "/frame.jpg")
        except Exception as error:
            print(f"[罗盘] 取数据失败：{error}", flush=True)
        frame = None
        if raw:
            frame = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            frame = blank.copy()
            cv2.putText(frame, "no frame from Pi", (20, 60), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.imshow(win, frame)
            if cv2.waitKey(200) & 0xFF in (ord("q"), 27):
                break
            continue

        for item in state.get("boxes") or []:
            x1, y1, x2, y2, cls, conf = item[:6]
            cls = int(cls)
            color = COLORS.get(cls, (255, 255, 255))
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(frame, f"{NAMES.get(cls, cls)} {float(conf):.2f}",
                        (int(x1), max(18, int(y1) - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

        compass = state.get("compass")
        if compass:
            x1, y1, x2, y2 = (int(v) for v in compass["box"])
            front = compass["label"] == "front"
            color = (0, 200, 0) if front else (0, 120, 255)
            text = "FRONT 正面" if front else "BACK 反面"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            cv2.putText(frame, f"{text}  {compass['ratio']:.3f}", (x1, min(frame.shape[0] - 10, y2 + 34)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 3, cv2.LINE_AA)
            now = time.time()
            if now - last_print > 0.4:
                last_print = now
                print(f"[罗盘] {compass['label']:5s} ratio={compass['ratio']:.3f}", flush=True)
        else:
            cv2.putText(frame, "no compass in view", (12, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.putText(frame, f"age {state.get('age')}s  count {state.get('count')}  ms {state.get('ms')}",
                    (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 2, cv2.LINE_AA)
        cv2.imshow(win, frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("s"):
            n_save += 1
            tag = compass["label"] if compass else "none"
            (save_dir / f"piframe_{n_save:03d}_{tag}.jpg").write_bytes(raw or b"")
            cv2.imwrite(str(save_dir / f"piframe_{n_save:03d}_{tag}_draw.jpg"), frame)
            print(f"已存第 {n_save} 张（{tag}）", flush=True)

        rest = pause - (time.time() - t0)
        if rest > 0:
            time.sleep(rest)

    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
