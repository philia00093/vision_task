"""相机部分：打开、协商格式/分辨率、实测帧率、模式探测。

针对 USB 2.0 双目模组的卡顿：先设分辨率再设 FOURCC=MJPG，避免驱动退回 YUY2（2560x960 只有 5 FPS）。"""

from __future__ import annotations

import cv2
import time

from edge.constants import BACKENDS, CANDIDATE_MODES


def fourcc_str(value):
    value = int(value)
    return "".join(chr((value >> (8 * i)) & 0xFF) for i in range(4))

def open_camera(index, backend, width, height, fps, fourcc="MJPG", buffersize=1):
    """协商相机格式；返回 (cap, 协商信息)。

    顺序至关重要（本机 HBV-1714-2 / USB2.0 / 2560x960 实测）：
        先设 FOURCC 再设分辨率 → 驱动丢弃 MJPG，回退 YUY2，只有 4.6 FPS；
        先设分辨率再设 FOURCC → 正确协商为 MJPG，实测 36.7 FPS。
    """
    cap = cv2.VideoCapture(index, BACKENDS[backend])
    if not cap.isOpened():
        raise RuntimeError(f"相机 {index} 无法打开（后端 {backend}）；请先关闭占用它的程序")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
    if fps:
        cap.set(cv2.CAP_PROP_FPS, fps)
    if buffersize:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, buffersize)      # 降低延迟（部分后端忽略）

    def negotiated():
        return {"width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "fps": cap.get(cv2.CAP_PROP_FPS),
                "fourcc": fourcc_str(cap.get(cv2.CAP_PROP_FOURCC))}

    info = negotiated()
    if info["fourcc"].upper() != fourcc.upper():           # 兜底：再设一次 FOURCC
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
        info = negotiated()
    return cap, info

def measure_fps(cap, warmup=10, frames=30):
    """读取若干帧并实测帧率，返回 (实测帧率, 单帧形状)。"""
    for _ in range(warmup):
        cap.read()
    shape, count, start = None, 0, time.time()
    for _ in range(frames):
        ok, frame = cap.read()
        if ok:
            count += 1
            shape = frame.shape
    elapsed = max(time.time() - start, 1e-6)
    return count / elapsed, shape

def probe_modes(index, backend, fourcc="MJPG"):
    print(f"相机 {index}（后端 {backend}，FOURCC {fourcc}）逐模式实测：")
    print(f"{'请求模式':>12} {'实际':>10} {'协商FOURCC':>10} {'报告FPS':>8} "
          f"{'实测FPS':>8}  评价")
    for width, height, fps in CANDIDATE_MODES:
        try:
            cap, info = open_camera(index, backend, width, height, fps, fourcc)
        except RuntimeError as error:
            print(f"{width}x{height:>6} 打开失败：{error}")
            continue
        measured, shape = measure_fps(cap)
        cap.release()
        actual = f"{shape[1]}x{shape[0]}" if shape else "无画面"
        verdict = ("可用" if measured >= 20 else
                   "偏慢" if measured >= 10 else "很卡")
        if info["fourcc"].upper() != fourcc.upper():
            verdict += f"（协商成 {info['fourcc']}，非 {fourcc}）"
        print(f"{width}x{height:>6} {actual:>10} {info['fourcc']:>10} "
              f"{info['fps']:>8.1f} {measured:>8.1f}  {verdict}")
