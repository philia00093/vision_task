"""把一帧 BGR 图像直接写进 framebuffer（外接屏，无图形界面时用）。

树莓派 Lite 没有 X，`cv2.imshow` 不可用；但 `/dev/fb0` 一直存在。
这里做三件事：读屏幕几何（尺寸/色深/跨距）→ 等比缩放 → 转成 RGB565 或 XRGB8888 字节写进去。

用法（`edge/app.py` 里已经接好，命令行 `--fb /dev/fb0` 即可）：

    from edge.framebuffer import FbWriter
    screen = FbWriter("/dev/fb0", mode="fit")
    screen.show(bgr_frame)
    screen.close()

色深按 `/sys/class/graphics/fb*/bits_per_pixel` 自动判断（16=RGB565，32=XRGB8888），
不需要改代码就能适配不同的屏。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import cv2
import numpy as np


def _read_int(path: Path, default: int) -> int:
    try:
        return int(path.read_text().strip())
    except Exception:
        return default


def _read_pair(path: Path, default: tuple[int, int]) -> tuple[int, int]:
    try:
        width, height = path.read_text().strip().split(",")
        return int(width), int(height)
    except Exception:
        return default


class FbWriter:
    """一个 framebuffer 显示器。`show()` 每帧调用一次。"""

    def __init__(self, device: str = "/dev/fb0", mode: str = "fit", rotate: int = 0,
                 name: str | None = None):
        self.device = str(device)
        sysfs = Path("/sys/class/graphics") / (name or Path(self.device).name)
        self.width, self.height = _read_pair(sysfs / "virtual_size", (1024, 600))
        self.bpp = _read_int(sysfs / "bits_per_pixel", 16)
        stride = _read_int(sysfs / "stride", self.width * self.bpp // 8)
        self.stride = stride if stride > 0 else self.width * self.bpp // 8
        self.mode = mode if mode in ("fit", "crop", "stretch") else "fit"
        self.rotate = int(rotate) if rotate in (0, 90, 180, 270) else 0
        self.frames = 0
        try:
            self.fd = os.open(self.device, os.O_WRONLY)
        except PermissionError as error:
            raise SystemExit(f"没有权限写 {self.device}（把用户加进 video 组："
                             f"sudo usermod -aG video $USER 然后重新登录）") from error
        except OSError as error:
            raise SystemExit(f"打不开 {self.device}：{error}（确认这是 framebuffer 设备）") from error

    def describe(self) -> str:
        return (f"{self.width}x{self.height} {self.bpp}bpp stride={self.stride} "
                f"mode={self.mode}" + (f" rotate={self.rotate}" if self.rotate else ""))

    # ------------------------------------------------------------ 图像变换
    def _rotate(self, frame):
        if not self.rotate:
            return frame
        code = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180,
                270: cv2.ROTATE_90_COUNTERCLOCKWISE}[self.rotate]
        return cv2.rotate(frame, code)

    def _fit(self, frame):
        """等比缩放：fit 留黑边、crop 裁切铺满、stretch 拉伸。"""
        height, width = frame.shape[:2]
        if self.mode == "stretch":
            return cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        scale = (max(self.width / width, self.height / height) if self.mode == "crop"
                 else min(self.width / width, self.height / height))
        new_w = max(1, int(round(width * scale)))
        new_h = max(1, int(round(height * scale)))
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        if self.mode == "crop":
            x = max(0, (new_w - self.width) // 2)
            y = max(0, (new_h - self.height) // 2)
            return resized[y:y + self.height, x:x + self.width]
        canvas = np.zeros((self.height, self.width, 3), np.uint8)
        x = (self.width - new_w) // 2
        y = (self.height - new_h) // 2
        canvas[y:y + new_h, x:x + new_w] = resized
        return canvas

    def _encode(self, bgr) -> bytes:
        """BGR → framebuffer 原始字节。"""
        if self.bpp == 16:                        # RGB565 小端
            blue, green, red = cv2.split(bgr)
            value = ((red.astype(np.uint16) >> 3) << 11) | \
                    ((green.astype(np.uint16) >> 2) << 5) | \
                    (blue.astype(np.uint16) >> 3)
            data = value.astype("<u2").tobytes()
        elif self.bpp == 32:                      # XRGB8888，最高字节没用
            bgra = np.empty((bgr.shape[0], bgr.shape[1], 4), np.uint8)
            bgra[:, :, :3] = bgr
            bgra[:, :, 3] = 0
            data = bgra.tobytes()
        else:
            raise RuntimeError(f"不支持的色深 {self.bpp}bpp（只支持 16/32）")
        line = self.width * self.bpp // 8
        if line != self.stride:                   # 跨距比行宽大：每行右侧补零
            rows = np.frombuffer(data, np.uint8).reshape(self.height, line)
            pad = np.zeros((self.height, self.stride - line), np.uint8)
            data = np.hstack([rows, pad]).tobytes()
        return data

    # ------------------------------------------------------------ 对外
    def show(self, frame) -> None:
        """显示一帧（BGR，任意尺寸，内部缩放）。"""
        if frame is None or frame.size == 0:
            return
        screen = self._fit(self._rotate(frame))
        data = self._encode(screen)
        os.pwrite(self.fd, data, 0)
        self.frames += 1

    def close(self) -> None:
        try:
            os.close(self.fd)
        except Exception:
            pass


if __name__ == "__main__":
    # 自检：在屏幕上画彩条，按 Ctrl+C 退出
    writer = FbWriter(sys.argv[1] if len(sys.argv) > 1 else "/dev/fb0")
    print(writer.describe())
    bars = np.zeros((240, 640, 3), np.uint8)
    colors = [(255, 255, 255), (0, 255, 255), (255, 255, 0), (0, 255, 0),
              (255, 0, 255), (0, 0, 255), (255, 0, 0)]
    step = 640 // len(colors)
    for index, color in enumerate(colors):
        bars[:, index * step:(index + 1) * step] = color
    try:
        while True:
            writer.show(bars)
    except KeyboardInterrupt:
        print("退出")
    finally:
        writer.close()
