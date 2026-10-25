#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""阶段识别放到后台线程跑，主循环只管「到点丢一帧过去」。

为什么需要这个
--------------
在 Pi 上实测一次级联匹配 135~160 ms（12 张参考图，SIFT 500 特征）。同步跑的话
主循环每 0.5 秒就被卡 135 ms，帧率从 27 掉到 24 —— 画面节奏立刻能看出来。

而 cv2 的 SIFT 是 C++ 实现，算的时候**会把 GIL 放开**，所以放到线程里是真并行：
主循环该采帧采帧、该画图画图，识别结果晚个几十毫秒回来完全无所谓。

队列只留 1 帧
-------------
识别比不过采集的时候，绝不排队（排了就会越算越旧）。新帧直接顶掉还没算的旧帧，
宁可少认一次，也不要拿 3 秒前的画面当"现在"。
"""

from __future__ import annotations

import queue
import threading
import time


class StageWorker:
    """给 StageMatcher 套一层后台线程：submit() 丢帧，drain() 收结果。"""

    def __init__(self, matcher):
        self.matcher = matcher
        self._todo: queue.Queue = queue.Queue(maxsize=1)
        self._done: queue.Queue = queue.Queue()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="stage-worker", daemon=True)
        self._thread.start()
        self.error: str | None = None

    # ---------------- 主循环这一侧 ----------------
    def submit(self, image, region, stamp: float | None = None) -> None:
        """把裁剪图交给后台。队列满就顶掉最旧的那帧（不排队、不滞后）。"""
        item = (image, region, time.time() if stamp is None else stamp)
        try:
            self._todo.put_nowait(item)
            return
        except queue.Full:
            pass
        try:
            self._todo.get_nowait()
        except queue.Empty:
            pass
        try:
            self._todo.put_nowait(item)
        except queue.Full:      # 极端竞争，这一帧算了
            pass

    def drain(self):
        """取回算完的结果：[(时间戳, 结果字典), ...]。"""
        out = []
        while True:
            try:
                out.append(self._done.get_nowait())
            except queue.Empty:
                return out

    def close(self) -> None:
        self._stop.set()

    @property
    def busy(self) -> bool:
        return self._thread.is_alive()

    # ---------------- 后台线程 ----------------
    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                image, region, stamp = self._todo.get(timeout=0.2)
            except queue.Empty:
                continue
            started = time.time()
            try:
                result = self.matcher.match(image, region=region)
            except Exception as exc:                      # 别让识别异常搞死线程
                self.error = f"{type(exc).__name__}: {exc}"
                result = {"label": "unknown", "score": 0.0, "note": self.error}
            cost = (time.time() - started) * 1000.0
            if result is None:
                continue
            result["cost_ms"] = cost
            self._done.put((stamp, result))
