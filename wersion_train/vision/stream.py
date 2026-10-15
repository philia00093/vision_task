"""视频流读取：实时相机（复用 left_eye_view 的协商经验）或视频文件/图片。

设计要点：
    · 相机走 MJPEG + 先分辨率后 FOURCC 的顺序（本机 HBV-1714-2 实测：顺序反了会退回
      YUY2，2560x960 只有 4.6 FPS）；视频文件无此问题；
    · 双目拼接帧用 --layout 拆出左目，直接复用项目根的 stereo_fusion.split_stereo；
    · 支持 --loop 循环播视频文件，方便反复调参；
    · 图片输入只读一帧，用于截图离线验证。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np

from .common import ROOT

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # 让 import stereo_fusion 生效

try:
    from stereo_fusion import split_stereo
except Exception:  # 独立拷贝 src/ 出去用时不该直接崩
    split_stereo = None

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


class StreamReader:
    """统一封装相机 / 视频文件 / 单张图片的逐帧读取。"""

    @staticmethod
    def _resolve_source(source) -> str:
        """解析视频源路径。

        相机序号（纯数字）原样返回。文件路径先按当前工作目录找，
        找不到再按项目根（wersion_train/）找 —— 因为 run_pipeline.bat 会
        pushd 到 wersion_train 运行，用户给的相对路径可能是相对项目根的。
        """
        text = str(source)
        if text.isdigit():
            return text
        candidate = Path(text).expanduser()
        if candidate.is_file():
            return str(candidate)
        if not candidate.is_absolute():
            fallback = ROOT / candidate
            if fallback.is_file():
                return str(fallback)
        return str(candidate)

    def __init__(self, source="0", backend="dshow", width=0, height=0, fps=0,
                 fourcc="MJPG", buffersize=1, layout="auto", loop=False, warmup=0):
        self.source = self._resolve_source(source)
        self.loop = loop
        self.warmup = max(0, int(warmup))
        self.layout = layout
        self.is_image = Path(self.source).suffix.lower() in IMAGE_SUFFIXES
        self.is_file = not self.is_image and Path(self.source).is_file()
        self.fps_hint = 0.0
        self.frame_index = 0
        self._image = None
        self._cap = None

        if self.is_image:
            self._image = cv2.imdecode(
                np.fromfile(self.source, dtype="uint8"), cv2.IMREAD_COLOR)
            if self._image is None:
                raise RuntimeError(f"图片读取失败：{self.source}")
            return

        if self.is_file:
            self._cap = cv2.VideoCapture(self.source)
        else:
            backend_flag = {"auto": cv2.CAP_ANY, "dshow": cv2.CAP_DSHOW,
                            "msmf": getattr(cv2, "CAP_MSMF", cv2.CAP_ANY)}.get(backend, cv2.CAP_ANY)
            self._cap = cv2.VideoCapture(int(self.source), backend_flag)
            if not self._cap.isOpened():
                raise RuntimeError(f"相机 {self.source} 打不开（后端 {backend}），"
                                   "请确认没有其它程序占用")
            # 顺序关键：先分辨率后 FOURCC
            if width:
                self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            if height:
                self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
            if fps:
                self._cap.set(cv2.CAP_PROP_FPS, fps)
            if buffersize:
                self._cap.set(cv2.CAP_PROP_BUFFERSIZE, buffersize)

        if not self._cap.isOpened():
            raise RuntimeError(f"视频源打不开：{self.source}")

        self.fps_hint = float(self._cap.get(cv2.CAP_PROP_FPS) or 0.0)
        self.size = (int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                     int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self.fourcc = self._fourcc_str(self._cap.get(cv2.CAP_PROP_FOURCC))

    @staticmethod
    def _fourcc_str(value) -> str:
        value = int(value or 0)
        return "".join(chr((value >> (8 * i)) & 0xFF) for i in range(4)).strip("\x00")

    def describe(self) -> str:
        if self.is_image:
            h, w = self._image.shape[:2]
            return f"图片 {Path(self.source).name} {w}x{h}"
        kind = "视频文件" if self.is_file else "相机"
        return (f"{kind} {self.source} {self.size[0]}x{self.size[1]} "
                f"@{self.fps_hint:.1f} {self.fourcc}".strip())

    def left_view(self, frame):
        """按 layout 取左目；无 stereo_fusion 或 layout=none 时原样返回。"""
        if self.layout == "none" or split_stereo is None:
            return frame
        try:
            return split_stereo(frame, self.layout)[0]
        except ValueError:
            return frame

    def read(self):
        """返回 (ok, 左目视图)。图片源只成功一次。"""
        if self.is_image:
            if self.frame_index > 0:
                return False, None
            self.frame_index += 1
            return True, self._image

        ok, frame = self._cap.read()
        if not ok and self.is_file and self.loop:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self._cap.read()
        if not ok:
            return False, None
        self.frame_index += 1
        return True, self.left_view(frame)

    def drain_warmup(self):
        """丢弃启动预热帧，避免第一帧就是黑的或曝光未稳定。"""
        for _ in range(self.warmup):
            self._cap.read()

    def release(self):
        if self._cap is not None:
            self._cap.release()
            self._cap = None


class FpsMeter:
    """滑动窗口实测帧率，用于状态栏显示真实性能。"""

    def __init__(self, window=20):
        self.window = window
        self._times: list[float] = []

    def tick(self) -> float:
        now = time.time()
        self._times.append(now)
        if len(self._times) > self.window:
            self._times.pop(0)
        if len(self._times) < 2:
            return 0.0
        return (len(self._times) - 1) / max(self._times[-1] - self._times[0], 1e-6)
