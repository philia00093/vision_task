#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""服务端的推理流水线：有界 FIFO + 定数 worker + 丢掉过期的帧。

为什么需要它
------------
网络抖动时帧是**一阵一阵到的**：实测手机热点下 0.5 秒里有时到 17 帧、有时一帧不到，
帧龄能飙到 4 秒。原来的做法是每条 HTTP 连接一个线程直接调 infer()，
于是：

· 16 个线程同时抢 `MODEL_LOCK`，谁抢到谁先算 —— **顺序是乱的**
· 抢不到的线程就那么干等着，队列深度没有上限，等到的可能是 4 秒前的帧
· GPU 被反复唤醒，单帧耗时不稳

现在改成一条流水线：

    请求到达 ──▶ 有界队列（满了丢**最旧**的）──▶ worker 按 FIFO 取 ──▶ 推理 ──▶ 回包

好处：
· **顺序**：先到的先算，结果按到达顺序回，不再乱序
· **有界**：队列最多 N 帧，超了就丢最旧的 —— 宁可这一帧不回框，也不要拿 4 秒前的画面充数
· **可观测**：队列深度 / 排队毫秒 / 丢帧数都记下来，/state 里能看
· **限速**：`--pace-fps` 可以强制匀速消费（默认不限制，来多少算多少）

被丢掉的帧不会让请求线程干等：立刻回一个 `dropped: true` 的结果，
客户端照常 ack（窗口照常前进），只是这一帧没有框。
"""

from __future__ import annotations

import threading
import time
from collections import deque


class _Job:
    __slots__ = ("image", "kwargs", "boxes", "cost_ms", "wait_ms", "dropped",
                 "error", "stamp", "event")

    def __init__(self, image, kwargs):
        self.image = image
        self.kwargs = kwargs
        self.boxes = []
        self.cost_ms = 0.0
        self.wait_ms = 0.0
        self.dropped = False
        self.error = ""
        self.stamp = time.time()
        self.event = threading.Event()


class InferPipeline:
    """有界 FIFO + 定数 worker 的推理流水线。"""

    def __init__(self, infer, workers: int = 1, capacity: int = 8,
                 pace_fps: float = 0.0, wait_timeout: float = 2.0):
        self._infer = infer
        self.capacity = max(1, int(capacity))
        self.pace = (1.0 / pace_fps) if pace_fps and pace_fps > 0 else 0.0
        self.wait_timeout = float(wait_timeout)
        self._queue: deque[_Job] = deque()
        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)
        self._stop = threading.Event()
        self._workers = []
        # 统计
        self.accepted = 0
        self.dropped = 0
        self.done = 0
        self.depth = 0
        self.depth_max = 0
        self.wait_ms_sum = 0.0
        self._last_done_at = 0.0
        self._gaps = 0
        for index in range(max(1, int(workers))):
            thread = threading.Thread(target=self._worker, args=(index,),
                                      name=f"infer-{index}", daemon=True)
            thread.start()
            self._workers.append(thread)

    # ------------------------------------------------------------ 提交
    def run(self, image, **kwargs):
        """提交一帧并等结果，返回 job 对象（boxes / cost_ms / wait_ms / dropped / error）。"""
        job = _Job(image, kwargs)
        with self._lock:
            if len(self._queue) >= self.capacity:
                # 队列满了：丢最旧的（它已经等最久、最可能是过期画面）
                victim = self._queue.popleft()
                victim.dropped = True
                victim.event.set()
                self.dropped += 1
            self._queue.append(job)
            self.accepted += 1
            self.depth = len(self._queue)
            self.depth_max = max(self.depth_max, self.depth)
            self._not_empty.notify()
        job.event.wait(self.wait_timeout)
        return job

    # ------------------------------------------------------------ worker
    def _worker(self, index: int) -> None:
        while not self._stop.is_set():
            with self._lock:
                while not self._queue and not self._stop.is_set():
                    self._not_empty.wait(0.1)
                if self._stop.is_set():
                    return
                job = self._queue.popleft()
                self.depth = len(self._queue)
            started = time.time()
            job.wait_ms = (started - job.stamp) * 1000.0
            try:
                job.boxes = self._infer(job.image, **job.kwargs)
            except Exception as error:                   # 单帧失败不能弄死 worker
                job.error = f"{type(error).__name__}: {error}"
            job.cost_ms = (time.time() - started) * 1000.0
            with self._lock:
                self.done += 1
                self.wait_ms_sum += job.wait_ms
                now = time.time()
                if self._last_done_at and (now - self._last_done_at) > 0.35:
                    self._gaps += 1                       # 处理节奏出现空档
                self._last_done_at = now
            job.event.set()
            # 限速：匀速消费（默认关）
            if self.pace:
                left = self.pace - (time.time() - started)
                if left > 0:
                    time.sleep(left)

    # ------------------------------------------------------------ 观测
    def stats(self) -> dict:
        with self._lock:
            avg_wait = (self.wait_ms_sum / self.done) if self.done else 0.0
            return {"queue_depth": self.depth, "queue_max": self.depth_max,
                    "accepted": self.accepted, "dropped": self.dropped,
                    "done": self.done, "wait_ms": round(avg_wait, 1),
                    "gaps": self._gaps}

    def close(self) -> None:
        self._stop.set()
        with self._lock:
            self._not_empty.notify_all()
