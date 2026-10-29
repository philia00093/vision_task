#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把"上传识别"整个搬到独立进程（树莓派上的关键一步）。

为什么要分进程
--------------
树莓派只有一个 Python 解释器、一把 GIL。主循环要解码 1280x960、画 1024x600 的屏、
跑区域和卡尔曼；发送线程要 JPEG 编码、TCP 收发、解析 160 字节回包。
两拨人抢同一把 GIL：实测主循环 29 fps 时，端到端只能送出 13~14 帧/秒。

拆进程后发送侧有自己的 GIL 和 CPU 核：
    主进程：采集/绘制/跟踪 —— 只管把裁剪帧丢进共享内存（1.35 MB memcpy，约 0.5 ms）
    子进程：JPEG 编码 + HTTP 收发 + 解析结果 —— 跑满自己的核，回包按 ref_seq 送回

怎么传帧
--------
用共享内存环形槽（默认 6 块，够 window 条连接吃）：
    主进程 put()：拿一个空槽 → 把裁剪图拷进去 → 把 (seq, slot, w, h) 扔队列
    子进程 worker：取任务 → 编码 → POST → 解析 → 结果回队列 → 把槽还回去
槽满了就丢这一帧（和原来"窗口满就丢"语义一致），不阻塞主循环。

子进程回什么
------------
(ref_seq, boxes, hand_box, hand, cost_ms)：
  boxes     那一帧的检测框（模型坐标，主进程再换算回原画面）
  hand_box  尾部 6 字节压缩的手框
  hand      阶段模型结果（cls/conf/box），主进程直接当"电脑端说了算"的字段用
"""

from __future__ import annotations

import multiprocessing as mp
import queue
import time
from typing import Optional

import cv2
import numpy as np

try:
    from multiprocessing import shared_memory
except ImportError:                                        # py3.7 及更早
    shared_memory = None

MAX_W, MAX_H = 1280, 960          # 单帧最大尺寸（上传裁剪不可能比这大）
MAX_SLOTS = 24


def _child_main(url, token, quality, window, wire_format, timeout, shm_name, shape,
                jobs_q, free_q, result_q, stop_evt, root):
    """子进程：拉任务 → 编码 → POST → 回结果。"""
    import http.client
    import socket
    import struct
    import sys
    import threading
    import urllib.parse

    if root not in sys.path:
        sys.path.insert(0, root)
    try:
        import protocol as wire
    except ImportError:                                    # 从别处起时补路径
        sys.path.insert(0, root)
        import protocol as wire

    shm = shared_memory.SharedMemory(name=shm_name, create=False)
    slots, height, width, channels = shape
    view = np.ndarray((slots, height, width, channels), dtype=np.uint8, buffer=shm.buf)

    parsed = urllib.parse.urlsplit(url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 80

    def connect():
        connection = http.client.HTTPConnection(host, port, timeout=timeout)
        connection.connect()
        try:
            connection.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except Exception:
            pass
        return connection

    def worker():
        connection = None
        while not stop_evt.is_set():
            try:
                job = jobs_q.get(timeout=0.2)
            except queue.Empty:
                continue
            if job is None:
                break
            seq, slot, frame_h, frame_w = job
            try:
                if connection is None:
                    connection = connect()
                frame = view[slot, :frame_h, :frame_w]
                ok, buffer = cv2.imencode(".jpg", frame,
                                          [int(cv2.IMWRITE_JPEG_QUALITY), quality])
                if not ok:
                    continue
                jpeg = buffer.tobytes()
                started = time.time()
                body = wire.encode_image_frame(jpeg, frame_w, frame_h, seq=seq,
                                               quality=quality)
                headers = {"Content-Type": "application/octet-stream"}
                if token:
                    headers["X-Token"] = token
                connection.request("POST", "/detect", body=body, headers=headers)
                response = connection.getresponse()
                raw = response.read()
                elapsed = (time.time() - started) * 1000.0
                if response.status != 200:
                    raise RuntimeError(f"HTTP {response.status}")
                if not wire.looks_like_frame(raw):
                    raise RuntimeError("回包不是结果帧")
                result = wire.parse_result_frame(raw)
                boxes = [(float(d["xyxy"][0]), float(d["xyxy"][1]),
                          float(d["xyxy"][2]), float(d["xyxy"][3]),
                          int(d["cls"]), float(d["conf"]))
                         for d in result["detections"]]
                result_q.put((int(result["ref_seq"]), boxes, result.get("hand_box"),
                              result.get("hand"), float(result.get("cost_ms", 0.0)),
                              elapsed))
            except Exception as error:
                try:
                    if connection is not None:
                        connection.close()
                except Exception:
                    pass
                connection = None
                result_q.put((-1, repr(error), None, None, 0.0, 0.0))
            finally:
                try:
                    free_q.put_nowait(slot)
                except Exception:
                    pass
        try:
            if connection is not None:
                connection.close()
        except Exception:
            pass

    workers = [threading.Thread(target=worker, daemon=True) for _ in range(max(1, window))]
    for thread in workers:
        thread.start()
    for thread in workers:
        thread.join()
    try:
        shm.close()
    except Exception:
        pass


class ProcessSender:
    """主进程这边看到的发送接口：put 一帧、drain 结果。"""

    def __init__(self, url: str, token: str = "", quality: int = 60, window: int = 8,
                 wire_format: str = "binary", timeout: float = 1.0, slots: int = 0):
        assert shared_memory is not None, "需要 Python 3.8+ 的 shared_memory"
        self.url = url
        self.token = token
        self.quality = int(quality)
        self.window = max(1, int(window))
        self.wire_format = wire_format
        self.timeout = float(timeout)
        # 槽数要 >= 线程数，否则 worker 抢不到帧；留 4 个余量做流水线
        self.slots = max(1, min(int(slots) or (self.window + 6), MAX_SLOTS))
        self.shape = None
        self.shm = None
        self.buffer = None
        self.jobs = None
        self.free = None
        self.results = None
        self.stop = None
        self.proc = None
        self.seq = 0
        self.sent = 0
        self.acked = 0
        self.dropped = 0
        self.errors = 0
        self.closed = False
        self.last_error = ""

    # ------------------------------------------------------------ 主进程接口
    def _start(self, height: int, width: int) -> None:
        """按真实帧尺寸建共享内存 + 起子进程（第一帧到了才知道裁剪多大）。"""
        self.shape = (self.slots, int(height), int(width), 3)
        self.shm = shared_memory.SharedMemory(create=True,
                                              size=int(self.slots * height * width * 3))
        self.buffer = np.ndarray(self.shape, dtype=np.uint8, buffer=self.shm.buf)
        self.jobs = mp.Queue(maxsize=self.slots)
        self.free = mp.Queue()
        self.results = mp.Queue()
        for slot in range(self.slots):
            self.free.put_nowait(slot)
        self.stop = mp.Event()
        self.proc = mp.Process(
            target=_child_main,
            args=(self.url, self.token, self.quality, self.window, self.wire_format,
                  self.timeout, self.shm.name, self.shape, self.jobs, self.free,
                  self.results, self.stop, str(__file__).rsplit("/edge/", 1)[0]),
            daemon=True, name="sender-proc")
        self.proc.start()

    def put(self, frame: np.ndarray) -> int:
        """拷进共享内存并排队；槽满就丢这一帧，返回 0 表示没送出去。"""
        if self.closed or frame is None or frame.size == 0:
            return 0
        h, w = frame.shape[:2]
        if h > MAX_H or w > MAX_W:
            return 0
        if self.proc is None:
            try:
                self._start(h, w)
            except Exception as error:
                print(f"[发送] 子进程启动失败：{error}", flush=True)
                self.closed = True
                return 0
        try:
            slot = self.free.get_nowait()
        except queue.Empty:
            self.dropped += 1
            return 0
        self.buffer[slot, :h, :w] = frame            # 1.35 MB memcpy，约 0.5 ms
        self.seq = (self.seq + 1) & 0xFFFFFFFF
        try:
            self.jobs.put_nowait((self.seq, slot, h, w))
        except queue.Full:
            self.free.put_nowait(slot)
            self.dropped += 1
            return 0
        self.sent += 1
        return self.seq

    def drain(self):
        """把子进程回来的结果全取走：[(ref_seq, boxes, hand_box, hand, cost_ms, rtt)]。"""
        out = []
        if self.results is None:
            return out
        while True:
            try:
                item = self.results.get_nowait()
            except queue.Empty:
                break
            if item[0] == -1:                        # 子进程报错
                self.errors += 1
                self.last_error = item[1]
                continue
            self.acked += 1
            out.append(item)
        return out

    @property
    def ready(self) -> bool:
        if self.closed:
            return False
        return self.proc is None or self.proc.is_alive()

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        if self.proc is None:
            return
        self.stop.set()
        for _ in range(self.window):
            try:
                self.jobs.put_nowait(None)
            except Exception:
                pass
        try:
            self.proc.join(timeout=2.0)
        except Exception:
            pass
        if self.proc.is_alive():
            try:
                self.proc.terminate()
            except Exception:
                pass
        for handle in (self.shm,):
            try:
                handle.close()
                handle.unlink()
            except Exception:
                pass
