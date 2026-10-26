#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""读触摸屏（evdev 原始事件），把物理坐标换算成屏幕坐标。

为什么自己读 `/dev/input/event*` 而不装 evdev 库
----------------------------------------------
树莓派上多装一个依赖就多一份"换台机器要重装"的麻烦。evdev 的事件格式是固定的
24 字节 `struct input_event`（tv_sec, tv_usec, type, code, value），用 struct
解就够了。

坐标换算
--------
面板报的是**原始值**（实测这台是 0~4096，两个轴一样），而界面是按 1024×600 画的，
所以要按 `raw / raw_max * screen_size` 换。轴的范围在打开设备时用 EVIOCGABS
读一次，不用写死。

支持两种上报方式
----------------
· 单点：ABS_X / ABS_Y + BTN_TOUCH（按下/抬起）
· 多点 B 协议：ABS_MT_POSITION_X/Y + ABS_MT_TRACKING_ID（-1 表示抬起）

只关心"第一根手指按在哪里、什么时候按下/抬起"，所以收到 SYN_REPORT 时
把当前累积的坐标打包成一个事件交出去，上层做按钮命中判定。

权限：`/dev/input/event*` 属于 root:input，640。把运行用户加进 input 组即可
（`sudo usermod -aG input <user>`，重新登录生效）。
"""

from __future__ import annotations

import os
import select
import struct
import threading

# struct input_event 在 64 位 Linux 上是 24 字节（timeval 两个 long + type/code/value）
_EVENT_FMT = "llHHI"
_EVENT_SIZE = struct.calcsize(_EVENT_FMT)

EV_SYN, EV_KEY, EV_ABS = 0x00, 0x01, 0x03
SYN_REPORT = 0
BTN_TOUCH = 0x14A
ABS_X, ABS_Y = 0x00, 0x01
ABS_MT_POSITION_X, ABS_MT_POSITION_Y = 0x35, 0x36
ABS_MT_TRACKING_ID = 0x39

_EVIOCGABS = 0x80184540          # _IOR('E', 0x40 + abs, struct input_absinfo)


def _abs_range(fd, code, default):
    """读某个轴的 (min, max)；读不到就用默认值。"""
    try:
        import fcntl
        buf = bytearray(24)
        fcntl.ioctl(fd, _EVIOCGABS + code, buf)
        _value, vmin, vmax, _fuzz, _flat, _res = struct.unpack("iiiiii", bytes(buf))
        if vmax > vmin:
            return vmin, vmax
    except Exception:
        pass
    return 0, default


class TouchReader:
    """后台线程读触摸事件，`poll()` 取回 (x, y, pressed) 列表（屏幕坐标）。"""

    def __init__(self, device: str = "/dev/input/event0",
                 width: int = 1024, height: int = 600,
                 swap_xy: bool = False, invert_x: bool = False, invert_y: bool = False):
        self.device = device
        self.width = int(width)
        self.height = int(height)
        self.swap = bool(swap_xy)
        self.invert_x = bool(invert_x)
        self.invert_y = bool(invert_y)
        self.error = ""
        self.raw_x_range = (0, 4096)
        self.raw_y_range = (0, 4096)
        self._events: list[tuple[int, int, bool]] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self._fd = None

    # ------------------------------------------------------------ 生命周期
    def start(self) -> bool:
        try:
            self._fd = os.open(self.device, os.O_RDONLY | os.O_NONBLOCK)
        except Exception as error:
            self.error = f"打不开 {self.device}（{error}）"
            return False
        self.raw_x_range = _abs_range(self._fd, ABS_MT_POSITION_X, 4096)
        self.raw_y_range = _abs_range(self._fd, ABS_MT_POSITION_Y, 4096)
        if self.raw_x_range == (0, 4096):
            self.raw_x_range = _abs_range(self._fd, ABS_X, 4096)
            self.raw_y_range = _abs_range(self._fd, ABS_Y, 4096)
        self._thread = threading.Thread(target=self._run, name="touch", daemon=True)
        self._thread.start()
        return True

    def close(self) -> None:
        self._stop.set()
        if self._fd is not None:
            try:
                os.close(self._fd)
            except Exception:
                pass
            self._fd = None

    @property
    def ready(self) -> bool:
        return self._fd is not None and not self.error

    # ------------------------------------------------------------ 消费
    def poll(self) -> list[tuple[int, int, bool]]:
        with self._lock:
            events, self._events = self._events, []
        return events

    # ------------------------------------------------------------ 换算
    def _to_screen(self, raw_x: int, raw_y: int) -> tuple[int, int]:
        x0, x1 = self.raw_x_range
        y0, y1 = self.raw_y_range
        fx = (raw_x - x0) / max(x1 - x0, 1)
        fy = (raw_y - y0) / max(y1 - y0, 1)
        if self.invert_x:
            fx = 1.0 - fx
        if self.invert_y:
            fy = 1.0 - fy
        if self.swap:
            fx, fy = fy, fx
        return (int(round(fx * (self.width - 1))), int(round(fy * (self.height - 1))))

    # ------------------------------------------------------------ 线程
    def _run(self) -> None:
        raw_x = raw_y = None
        pressed = False
        touched = False
        while not self._stop.is_set() and self._fd is not None:
            try:
                ready, _, _ = select.select([self._fd], [], [], 0.2)
            except Exception:
                break
            if not ready:
                continue
            try:
                data = os.read(self._fd, _EVENT_SIZE * 64)
            except BlockingIOError:
                continue
            except Exception:
                break
            for offset in range(0, len(data) - _EVENT_SIZE + 1, _EVENT_SIZE):
                _s, _us, ev_type, code, value = struct.unpack(
                    _EVENT_FMT, data[offset:offset + _EVENT_SIZE])
                if ev_type == EV_ABS:
                    if code in (ABS_MT_POSITION_X, ABS_X):
                        raw_x = value
                    elif code in (ABS_MT_POSITION_Y, ABS_Y):
                        raw_y = value
                    elif code == ABS_MT_TRACKING_ID:
                        touched = value >= 0
                        if value < 0 and pressed:
                            pressed = False
                            if raw_x is not None and raw_y is not None:
                                x, y = self._to_screen(raw_x, raw_y)
                                self._push(x, y, False)
                elif ev_type == EV_KEY and code == BTN_TOUCH:
                    touched = value > 0
                    if value == 0 and pressed:
                        pressed = False
                        if raw_x is not None and raw_y is not None:
                            x, y = self._to_screen(raw_x, raw_y)
                            self._push(x, y, False)
                elif ev_type == EV_SYN and code == SYN_REPORT:
                    if touched and not pressed and raw_x is not None and raw_y is not None:
                        pressed = True
                        x, y = self._to_screen(raw_x, raw_y)
                        self._push(x, y, True)

    def _push(self, x: int, y: int, pressed: bool) -> None:
        with self._lock:
            if len(self._events) < 32:
                self._events.append((x, y, pressed))
