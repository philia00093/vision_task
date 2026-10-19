#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""伪装成「树莓派」：把一帧图按二进制帧发给电脑端服务，打印回来的结果。

用途
----
不用树莓派、不用摄像头，就能验证电脑端服务有没有把协议实现对：

    # 电脑上先起服务（真服务或 sim/mock_server.py 都行）
    python3 server/run_server.py --host 0.0.0.0 --port 8080

    # 另一个终端：伪造树莓派发包
    python3 sim/mock_edge.py --server-url http://127.0.0.1:8080 --frames 10 --fps 2

    # 故意发坏包，验证服务端会按 CRC 拒收
    python3 sim/mock_edge.py --corrupt-every 3

    # 走老的 JSON 路（裸 JPEG 上行）
    python3 sim/mock_edge.py --json

图像从哪来
----------
* `--image 某张图.png`：直接用这张图的字节（PNG/JPEG 都行，不用装 opencv）
* 不给图片：自己合成一张 640x480 的 BMP（彩条 + 一个移动的白块），**纯标准库**，
  所以笔记本上没装 opencv 也能跑

输出示例
--------
    seq=1  160B  det=3  base@0.86 support@0.71 compass@0.63  step=2 flags=1,1,0  srv 12.3ms  往返 4.1ms
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import protocol as wire  # noqa: E402


# ---------------------------------------------------------------- 合成图像
class BmpSource:
    """自己画一张 24 位 BMP：彩条背景 + 一个来回移动的白块，纯标准库。"""

    def __init__(self, width=640, height=480):
        self.width, self.height = int(width), int(height)
        self.row_bytes = ((self.width * 3 + 3) // 4) * 4
        self.base = bytearray(54 + self.row_bytes * self.height)
        self._header()
        self._paint_base()
        self.t0 = time.time()

    def _header(self) -> None:
        buf = self.base
        buf[0:2] = b"BM"
        struct.pack_into("<I", buf, 2, len(buf))
        struct.pack_into("<I", buf, 10, 54)
        struct.pack_into("<I", buf, 14, 40)
        struct.pack_into("<i", buf, 18, self.width)
        struct.pack_into("<i", buf, 22, self.height)
        struct.pack_into("<H", buf, 26, 1)
        struct.pack_into("<H", buf, 28, 24)
        struct.pack_into("<I", buf, 34, self.row_bytes * self.height)
        struct.pack_into("<i", buf, 38, 2835)
        struct.pack_into("<i", buf, 42, 2835)

    def _paint_base(self) -> None:
        """竖向彩条，从上到下渐暗（BMP 是自下而上存的，这里照样铺满即可）。"""
        colors = [(0, 0, 255), (0, 255, 255), (0, 255, 0), (255, 255, 0),
                  (255, 0, 0), (255, 0, 255)]
        bar = max(1, self.width // len(colors))
        for y in range(self.height):
            row = 54 + y * self.row_bytes
            fade = 1.0 - 0.55 * (y / max(self.height - 1, 1))
            for x in range(self.width):
                b, g, r = colors[min(x // bar, len(colors) - 1)]
                offset = row + x * 3
                self.base[offset] = int(b * fade)
                self.base[offset + 1] = int(g * fade)
                self.base[offset + 2] = int(r * fade)

    def frame(self) -> bytes:
        """返回当前时刻的一帧 BMP 字节。"""
        buf = bytearray(self.base)
        t = time.time() - self.t0
        size = max(24, self.width // 8)
        x0 = int((0.5 + 0.4 * __import__("math").sin(t * 2)) * (self.width - size))
        y0 = int((0.5 + 0.4 * __import__("math").cos(t * 1.3)) * (self.height - size))
        white = b"\xff\xff\xff" * size
        for y in range(y0, min(y0 + size, self.height)):
            row = 54 + y * self.row_bytes
            buf[row + x0 * 3: row + (x0 + size) * 3] = white
        return bytes(buf)


# ---------------------------------------------------------------- 发送
def guess_format(data: bytes) -> int:
    if data[:2] == b"\xff\xd8":
        return wire.IMG_JPEG
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return wire.IMG_PNG
    if data[:2] == b"BM":
        return wire.IMG_PNG          # 服务端只用 cv2.imdecode，BMP 也行，格式位仅作标识
    return wire.IMG_JPEG


def one_round(url: str, image: bytes, width: int, height: int, seq: int,
               quality: int, wire_format: str, corrupt: bool, timeout: float,
               verbose: bool) -> dict:
    """发一帧，返回 {'ok','ms','det','step','flags','cost_ms','error'}。"""
    if wire_format == "json":
        body = image
        content_type = "image/jpeg"
        request = urllib.request.Request(url.rstrip("/") + "/detect", data=body,
                                         method="POST", headers={"Content-Type": content_type})
        start = time.time()
        try:
            payload = json.loads(urllib.request.urlopen(request, timeout=timeout).read())
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", "replace")[:160]
            return {"ok": False, "error": f"HTTP {error.code} {detail}"}
        except Exception as error:
            return {"ok": False, "error": str(error)}
        round_trip = (time.time() - start) * 1000.0
        boxes = payload.get("boxes", [])
        if verbose:
            pretty = " ".join(f"{wire.CLS_NAMES[b[4]]}@{b[5]:.2f}" for b in boxes
                              if 0 <= b[4] < len(wire.CLS_NAMES))
            print(f"seq={seq}  det={len(boxes)}  {pretty or '无目标'}  "
                  f"srv {payload.get('ms', 0):.1f}ms  往返 {round_trip:.1f}ms")
        return {"ok": bool(payload.get("ok")), "ms": round_trip, "det": len(boxes),
                "step": 0, "flags": (0, 0, 0), "cost_ms": payload.get("ms", 0.0)}

    frame = wire.encode_image_frame(image, width, height, seq=seq,
                                    fmt=guess_format(image), quality=quality,
                                    stamp_ms=int(time.time() * 1000) & 0xFFFFFFFF)
    if corrupt:                                   # 故意把 CRC 弄坏，验证服务端会拒收
        frame = frame[:-1] + bytes([frame[-1] ^ 0xFF])
    request = urllib.request.Request(url.rstrip("/") + "/detect", data=frame, method="POST",
                                     headers={"Content-Type": "application/octet-stream"})
    start = time.time()
    try:
        raw = urllib.request.urlopen(request, timeout=timeout).read()
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", "replace")[:160]
        return {"ok": False, "error": f"HTTP {error.code} {detail}",
                "corrupt_expected": corrupt}
    except Exception as error:
        return {"ok": False, "error": str(error)}
    round_trip = (time.time() - start) * 1000.0
    if not wire.looks_like_frame(raw):
        return {"ok": False, "error": f"回包不是二进制帧（{len(raw)}B）"}
    try:
        result = wire.parse_result_frame(raw)
    except wire.FrameError as error:
        return {"ok": False, "error": f"回包解析失败：{error}"}
    if verbose:
        pretty = " ".join(f"{d['name']}@{d['conf']:.2f}" for d in result["detections"]) or "无目标"
        hand = f" 手@{result['hand']['conf']:.2f}" if result["hand"] else ""
        print(f"seq={seq}  {len(raw)}B  det={result['det_count']}  {pretty}{hand}  "
              f"step={result['step']} flags={','.join(str(v) for v in result['flags'])}  "
              f"srv {result['cost_ms']:.1f}ms  往返 {round_trip:.1f}ms")
    return {"ok": True, "ms": round_trip, "det": result["det_count"],
            "step": result["step"], "flags": result["flags"],
            "cost_ms": result["cost_ms"], "size": len(raw)}


def main() -> int:
    parser = argparse.ArgumentParser(description="伪装树莓派向识别服务发包")
    parser.add_argument("--server-url", default="http://127.0.0.1:8080")
    parser.add_argument("--frames", type=int, default=5, help="发多少帧（0=一直发）")
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--image", default=None, help="用这张图当帧（PNG/JPEG，纯读字节）")
    parser.add_argument("--width", type=int, default=640, help="合成图宽度（没有 --image 时）")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--quality", type=int, default=80)
    parser.add_argument("--wire", choices=("binary", "json"), default="binary")
    parser.add_argument("--json", action="store_true", help="等价于 --wire json")
    parser.add_argument("--corrupt-every", type=int, default=0,
                        help="每 N 帧故意弄坏一次 CRC（验证服务端拒收）")
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--seq-start", type=int, default=1)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    wire_format = "json" if args.json else args.wire
    if args.image:
        data = Path(args.image).expanduser().read_bytes()
        width, height = 1280, 960
        print(f"[sim] 用图片 {args.image}（{len(data)} 字节）当每一帧")
        source = lambda: data                                    # noqa: E731
    else:
        bmp = BmpSource(args.width, args.height)
        width, height = bmp.width, bmp.height
        print(f"[sim] 合成 {width}x{height} BMP（彩条 + 移动白块）")
        source = bmp.frame

    total = args.frames if args.frames > 0 else None
    sent = ok = 0
    corrupt_ok = 0
    seq = args.seq_start
    interval = 1.0 / max(args.fps, 0.1)
    print(f"[sim] 目标 {args.server_url}/detect  帧格式 {wire_format}"
          f"  {'一直发' if total is None else f'{total} 帧'}  {args.fps} fps")
    started = time.time()
    try:
        while total is None or sent < total:
            corrupt = bool(args.corrupt_every) and (sent + 1) % args.corrupt_every == 0
            result = one_round(args.server_url, source(), width, height, seq,
                               args.quality, wire_format, corrupt, args.timeout,
                               not args.quiet)
            sent += 1
            if result.get("ok"):
                ok += 1
            elif corrupt:
                corrupt_ok += 1
                if not args.quiet:
                    print(f"seq={seq}  坏包被拒 ✓  {result.get('error', '')}")
            else:
                print(f"seq={seq}  失败：{result.get('error')}")
            seq += 1
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\n[sim] 中断")
    elapsed = max(time.time() - started, 1e-6)
    print(f"\n[sim] 共发 {sent} 帧，成功 {ok}，坏包被拒 {corrupt_ok}，"
          f"用时 {elapsed:.1f}s（{sent / elapsed:.1f} fps）")
    return 0 if ok or corrupt_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
