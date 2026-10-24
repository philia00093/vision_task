"""取帧线程：把耗时的 `cap.read()` 从主循环里挪出去。

为什么需要
----------
1280×960 MJPG 在树莓派 4B 上单帧解码约 20.8 ms——如果它躺在主循环里，
那么"解码 → 画框 → 写屏 → 发网络"只能串行，整条流水线被钉在 ~26 fps。

改成**生产者/消费者**之后：

    线程 A：cap.read() 循环 → 只保留最新一帧（覆盖旧的）
    线程 B：主循环 → 取最新帧 → 画框 → 写屏 → 发网络

两个线程跑在不同 CPU 核上，各干各的。主循环单帧成本从 ~38 ms 降到 ~12 ms，
整条流水线的吞吐由"解码线程"决定（约 45 fps），而不是两者之和。

设计取舍
--------
* **只保留最新一帧**：不排队。视觉应用要的是"现在"，不是"三帧以前"。
  队列会在网络/显示慢的时候越积越多，延迟越滚越大。
* `read()` 默认最多等 `timeout` 秒；超时返回上一帧（`fresh=False`），
  主循环据此判断相机是不是挂了，但不会卡死。
* 线程里任何异常都记录到 `error`，主循环读得到；不会因为相机掉线把整程序带崩。
"""

from __future__ import annotations

import threading
import time


class FrameGrabber:
    """后台线程持续 `cap.read()`，对外只暴露"最新一帧"。"""

    def __init__(self, cap, timeout=1.0):
        self.cap = cap
        self.timeout = float(timeout)
        self.error = ""
        self.frames = 0
        self.fps = 0.0
        self._frame = None
        self._stamp = 0.0
        self._serial = 0            # 每取到一帧 +1，用来判断"有没有新帧"
        self._cond = threading.Condition()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="frame-grabber", daemon=True)
        self._thread.start()

    # ------------------------------------------------------------ 线程
    def _loop(self) -> None:
        count, window_start = 0, time.time()
        while not self._stop.is_set():
            try:
                ok, frame = self.cap.read()
            except Exception as error:              # 相机拔掉/驱动崩了
                self.error = f"取帧异常：{error}"
                time.sleep(0.2)
                continue
            if not ok or frame is None:
                self.error = "读取画面失败"
                time.sleep(0.05)
                continue
            with self._cond:
                self._frame = frame
                self._stamp = time.time()
                self._serial += 1
                self._cond.notify_all()
            self.error = ""
            count += 1
            elapsed = time.time() - window_start
            if elapsed >= 2.0:
                self.fps = count / elapsed
                count, window_start = 0, time.time()

    # ------------------------------------------------------------ 对外
    def latest(self):
        """返回 (frame, stamp, serial, fresh)。没有新帧就返回上一帧。"""
        with self._cond:
            if self._frame is not None:
                return self._frame, self._stamp, self._serial, True
            self._cond.wait(self.timeout)
            if self._frame is not None:
                return self._frame, self._stamp, self._serial, True
            return None, 0.0, 0, False

    def read(self, last_serial: int, timeout: float | None = None):
        """等到比 last_serial 更新的一帧。

        返回 (frame, serial, stamp, fresh)：
            stamp  这一帧从相机读出来的时刻（用来算"画面滞后"）
            fresh  False 表示超时、拿到的是旧帧
        """
        deadline = time.time() + (self.timeout if timeout is None else timeout)
        with self._cond:
            # 注意：第一帧还没来时 self._frame 是 None，必须等；
            # 只比较 serial 会在启动瞬间直接返回空帧（last_serial=-1 vs serial=0）。
            while (self._frame is None or self._serial <= last_serial) \
                    and not self._stop.is_set():
                remaining = deadline - time.time()
                if remaining <= 0:
                    return self._frame, self._serial, self._stamp, False
                self._cond.wait(remaining)
            return self._frame, self._serial, self._stamp, self._serial > last_serial

    @property
    def alive(self) -> bool:
        return self._thread.is_alive()

    def close(self) -> None:
        self._stop.set()
        with self._cond:
            self._cond.notify_all()
        self._thread.join(timeout=1.0)
