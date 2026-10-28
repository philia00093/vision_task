"""帧格式（树莓派 ⇄ 电脑）—— 二进制，定长头 + 定长字段 + CRC32。

**两边共用这一份**：末端 `from protocol import ...`，服务器端也 `from protocol import ...`
（两个入口都把自己的上一层放进了 sys.path，所以放在项目根就行）。

============================ 为什么这么定 ============================

1. **定长头 16 字节**：magic + 版本 + 类型 + 标志 + 保留 + 序号 + 载荷长度。
   `payload_len` 让接收方知道"还差多少字节"，TCP 粘包/半包都不怕。
2. **CRC32 覆盖 magic 之后到载荷结尾**（不含 CRC 自身），比 CRC16 强，
   而 Python 的 `zlib.crc32` 是 C 实现，树莓派上算 100KB 只要几十微秒。
3. **下行帧定长 160 字节**：4 个检测槽 + 1 个手部槽 + 步数 + 3 个布尔 + 尾部保留。
   定长意味着**没有 TLV 解析**，`struct.unpack` 一次拆完，Pi 上不费算力。
4. **每个检测槽 20 字节、4 组点，而不是两个点**：
   现在模型给的是 `[x1,y1,x2,y2]`（轴对齐），落成 4 个点时按顺时针写
   `左上→右上→右下→左下`，等于 `(x1,y1) (x2,y1) (x2,y2) (x1,y2)`。
   以后换旋转框/分割轮廓（4 个角点）时**帧格式一个字都不用改**。
5. **预留区留着，不动版本**：手部槽先占好 20 字节（`valid=0` 表示没检测到手），
   尾部再留 16 字节。要加字段先吃预留区；确实不够了才升 `VERSION`，
   老解析器会明确拒绝而不是读出乱码。

============================ 通用帧头（16 字节，小端）============================

    偏移 长度 字段          说明
    0    2    magic         b"VP"（0x56 0x50），用于流里重新同步
    2    1    version       当前 1；版本不同直接丢弃
    3    1    type          1=图像上行  2=结果下行  3=心跳/应答  0x7F=错误
    4    1    flags         现在恒为 0（按位预留，见下）
    5    1    rsv8          0
    6    2    rsv16         0
    8    4    seq           u32 序号：发送方自增（Pi 每张图 +1）
    12   4    payload_len   u32 载荷字节数（<= 4MB）

    载荷 payload_len 字节
    CRC32         4        覆盖 0..15+payload_len（不含自己）

帧总长 = 16 + payload_len + 4。上行图像帧整帧约 16 + 12 + JPEG + 4。

============================ 上行：图像帧（type=1）============================

载荷 = 12 字节元信息 + 图像字节：

    偏移 长度 字段       说明
    0    2    width      u16 像素宽（如 1280）
    2    2    height     u16 像素高（如 960）
    4    1    format     1=JPEG  2=PNG
    5    1    quality    JPEG 质量 1~100；PNG 写 0
    6    2    rsv        0
    8    4    stamp_ms   拍摄时刻（u32，相对启动的毫秒，够 49 天）

图像字节就是 `cv2.imencode(".jpg", frame)[1].tobytes()` 的原样输出，
接收方直接 `cv2.imdecode`，不用再包一层。

============================ 下行：结果帧（type=2）============================

载荷恰好 140 字节，整帧 160 字节：

    偏移 长度 字段
    0    4    ref_seq       u32 这张结果对应哪一帧图像（= 上行帧头的 seq）
    4    1    det_count     u8  有效检测数 0~4
    5    3    rsv
    8    4    cost_us       u32 服务端推理耗时（微秒，0=没报）
    12   80   det[4]        4 个检测槽，每槽 20 字节
    92   20   hand          手部槽（**预留**，每槽 20 字节，格式同检测槽）
    112  4    step          u8 装配步数 0=未知，1~4=做到第几步；+ 3 字节 rsv
    116  8    flags         b0,b1,b2 各 u8（0/1，**3 组布尔**）+ 5 字节 rsv
    124  4    ack_seq       u32 **确认帧**：服务端已经处理完的这一帧（= 该图像的上行 seq）
    128  4    rcv_nxt       u32 累积确认：服务端收到的连续最高帧号
    132  2    win           u16 服务端还能接几帧（窗口，0=没上报）
    134  6    rsv           保留（全 0）

    ↑ 这 16 字节原来叫 tail（尾部预留），现在被 **ACK 确认** 占用了前 10 字节，
      依旧在 CRC 之前，整帧还是 160 字节，不用改版本。

**检测槽 / 手部槽（20 字节，小端）**：

    偏移 长度 字段     说明
    0    1    cls      u8 类别 0~3（base/support/compass/top，按模型 names 顺序）
    1    1    valid    u8 1=本槽有效，0=空槽（解码成 None）
    2    2    conf     u16 置信度千分比（0~1000，即 0.873 → 873）
    4    16   points   4 个点 × (x u16, y u16)，顺序 左上→右上→右下→左下

`hand` 槽的 `cls` 恒为 0，`valid=1` 表示检测到手，坐标同上（手部框的 4 个角点）。

为什么是"4 组 × 4 个点"：一组 = 一个检测目标。`det_count` 说明真正用了几组，
固定 4 组是为了**定长**——解析不用循环找边界。目标多于 4 个时，要么只发最可信的 4 个，
要么把 `tail` 扩展到 8 组并升版本（留了 16 字节，正好再放 0 组……所以真要 8 组请升 VERSION，
别偷偷改长度）。

============================ ACK 帧（type=3，32 字节）============================

握手/独立确认用。载荷 12 字节：

    偏移 长度 字段
    0    4    ack_seq      u32 已处理完的帧号
    4    4    rcv_nxt      u32 连续收到的最高帧号（累积确认）
    8    2    win          u16 还能接几帧
    10   2    rsv

它和结果帧里的 ack 字段冗余：走 HTTP 时结果帧自带 ack，够用；
将来换成裸 TCP 流、服务端要主动告诉客户端"我收到了但这一帧我不打算回结果"时，
就发一个独立的 ACK 帧（省掉一次推理和 160 字节回包）。

============================ 枚举与约定 ============================

* 类别：`CLS_NAMES = ("base", "support", "compass", "top")`，和 `edge/constants.py`
  的 `DEFAULT_PART_NAMES` 一致。
* 步数：`step` 0=未知，1~4=实际装配步（对应 `vision_yolo_train/images/steps` 的四张参考图）。
* 3 组布尔：语义由业务定，帧里只保证是 0/1。建议先用
  `(物料区被遮挡, 装配区被遮挡, 阶段已确认)`，等你定了再改注释。
* `ref_seq` 的意义：Pi 侧并发/乱序回来时，`ref_seq` 比上次小的直接丢
  （旧结果不能覆盖新结果）。

============================ 怎么接 ============================

**A. 走 HTTP（和现在一样，只把 body 换成二进制）**

    Pi  :  jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()
           body = encode_image_frame(jpeg, 1280, 960, seq=seq)
           # POST /detect，Content-Type: application/octet-stream，body 就是这帧
           结果 = parse_result_frame(响应体)

    PC  :  图像 = parse_image_frame(请求体)        # {'jpeg':..., 'width':..., 'seq':...}
           boxes = infer(cv2.imdecode(图像['jpeg'], cv2.IMREAD_COLOR), ...)
           回包 = encode_result_frame(图像['seq'], boxes, step=3, flags=(1,0,1))

**B. 走裸 TCP（两件事：`split_frames` 拆包，帧头里的 seq 做应答）**

    收：buf += sock.recv(65536)
        frames, buf = split_frames(buf)          # 半包/粘包/脏数据都在里面处理
        对每帧 parse_xxx_frame(frame) → 处理 → encode_xxx_frame(...) → sendall

**C. 走 UDP**：不建议（100KB 的图像要自己分片重传）。真要试，先在 `flags` 里
   约定分片位，再用 `tail` 存 `total_frags / frag_index`——这是另一套东西，别混进来。

============================ 自检 ============================

    python protocol.py        # 往返编解码 + 各字段边界 + 坏 CRC 拒收，全绿就说明两侧可用
"""

from __future__ import annotations

import struct
import sys
import zlib

# ---------------------------------------------------------------- 常量

MAGIC = b"VP"                       # 帧头标记（流里重新同步用）
VERSION = 1                         # 不兼容改动才升版本

TYPE_IMAGE = 1                      # 上行：一帧图像
TYPE_RESULT = 2                     # 下行：识别结果
TYPE_ACK = 3                        # 心跳/应答（载荷 0 字节，seq 回显对方序号）
TYPE_ERROR = 0x7F                   # 错误（载荷是一段 UTF-8 说明）

IMG_JPEG = 1
IMG_PNG = 2

MAX_DET = 4                         # 下行固定 4 个检测槽
MAX_PAYLOAD = 4 * 1024 * 1024       # 4MB：1280x960 JPEG 才 100KB 左右，这个上限只是防脏数据
MAX_IMAGE_SIDE = 65535              # 坐标/尺寸都是 u16

CLS_NAMES = ("base", "support", "compass", "top")

STEP_UNKNOWN = 0
STEP_RANGE = (1, 4)                 # 进行到第 1~4 步

# ---------------------------------------------------------------- 结构体

# 帧头 16 字节：magic, version, type, flags, rsv8, rsv16, seq, payload_len
HEADER = struct.Struct("<2sBBBBHII")
HEADER_SIZE = HEADER.size                    # 16
CRC = struct.Struct("<I")
CRC_SIZE = CRC.size                          # 4

# 图像帧元信息 12 字节
IMAGE_META = struct.Struct("<HHBBHI")        # width, height, format, quality, rsv, stamp_ms
IMAGE_META_SIZE = IMAGE_META.size            # 12

# 一个检测槽 20 字节：cls, valid, conf(千分比), 4 个点 × (x, y)
SLOT = struct.Struct("<BBH8H")
SLOT_SIZE = SLOT.size                        # 20

# 结果帧头 12 字节：ref_seq, det_count, rsv×3, cost_us
RESULT_HEAD = struct.Struct("<IBBBBI")
STEP_BLOCK = struct.Struct("<4B")            # step + 3 字节保留
FLAG_BLOCK = struct.Struct("<8B")            # 3 组布尔 + 5 字节保留
TAIL_SIZE = 16                               # 尾部 16 字节（前 10 字节现在是 ACK 字段）
# 尾部 16 字节：ack_seq(4) + rcv_nxt(4) + win(2) + 压缩手框(4×u8, 单位 4px) + 2 保留
TAIL_BLOCK = struct.Struct("<IIH4B2x")
ACK_PAYLOAD = struct.Struct("<IIH2x")        # ACK 帧载荷（12 字节）
assert TAIL_BLOCK.size == TAIL_SIZE and ACK_PAYLOAD.size == 12

RESULT_PAYLOAD_SIZE = (RESULT_HEAD.size + SLOT_SIZE * MAX_DET + SLOT_SIZE
                       + STEP_BLOCK.size + FLAG_BLOCK.size + TAIL_SIZE)
RESULT_FRAME_SIZE = HEADER_SIZE + RESULT_PAYLOAD_SIZE + CRC_SIZE
assert RESULT_PAYLOAD_SIZE == 140 and RESULT_FRAME_SIZE == 160, "帧长算错了"


class FrameError(ValueError):
    """帧不合法（长度/magic/版本/CRC/字段越界）。收到它就当这一帧没到过。"""


# ---------------------------------------------------------------- 小工具

def crc32(data: bytes) -> int:
    """标准 CRC32（多项式 0xEDB88320，同 zlib/以太网）。"""
    return zlib.crc32(data) & 0xFFFFFFFF


def quad_from_xyxy(x1, y1, x2, y2):
    """轴对齐框 → 4 个角点：左上 → 右上 → 右下 → 左下（顺时针）。"""
    return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]


def xyxy_from_quad(quad):
    """4 个角点 → 轴对齐外接框（画框、IoU 都用得上）。"""
    xs = [int(p[0]) for p in quad]
    ys = [int(p[1]) for p in quad]
    return (min(xs), min(ys), max(xs), max(ys))


def _u16(value, name: str) -> int:
    """坐标/尺寸统一收进 u16，越界就报错而不是悄悄截断。"""
    number = int(round(float(value)))
    if not 0 <= number <= MAX_IMAGE_SIDE:
        raise FrameError(f"{name}={value} 超出 0~{MAX_IMAGE_SIDE}")
    return number


def _conf_u16(conf) -> int:
    """置信度 → 千分比（0.873 → 873）。"""
    number = int(round(float(conf) * 1000))
    return max(0, min(number, 1000))


# ---------------------------------------------------------------- 帧封装

def build_frame(type_: int, seq: int, payload: bytes = b"", flags: int = 0) -> bytes:
    """套上 16 字节头 + 4 字节 CRC，得到可以直接发出去的整帧。"""
    if len(payload) > MAX_PAYLOAD:
        raise FrameError(f"载荷 {len(payload)} 字节超过上限 {MAX_PAYLOAD}")
    head = HEADER.pack(MAGIC, VERSION, int(type_) & 0xFF, int(flags) & 0xFF, 0, 0,
                       int(seq) & 0xFFFFFFFF, len(payload))
    body = head + payload
    return body + CRC.pack(crc32(body))


def looks_like_frame(body: bytes) -> bool:
    """一段请求体是不是本协议的帧。

    服务端用它区分"二进制帧"和"老的裸 JPEG"：JPEG 以 FF D8 开头，绝不会是 "VP"，
    所以两种客户端可以同时在线，不用改端口、不用改路由。
    """
    return len(body) >= HEADER_SIZE + CRC_SIZE and body[:2] == MAGIC


def parse_frame(frame: bytes) -> dict:
    """拆一整帧（必须正好是一帧）→ {'type','flags','seq','payload'}。"""
    if len(frame) < HEADER_SIZE + CRC_SIZE:
        raise FrameError(f"帧太短：{len(frame)} 字节")
    magic, version, type_, flags, _r8, _r16, seq, length = HEADER.unpack_from(frame, 0)
    if magic != MAGIC:
        raise FrameError(f"magic 不对：{magic!r}")
    if version != VERSION:
        raise FrameError(f"版本 {version} 不认识（本端只认 {VERSION}）")
    total = HEADER_SIZE + length + CRC_SIZE
    if len(frame) != total:
        raise FrameError(f"长度不符：头里说 {total} 字节，实际 {len(frame)} 字节")
    if length > MAX_PAYLOAD:
        raise FrameError(f"载荷 {length} 字节超过上限 {MAX_PAYLOAD}")
    (given,) = CRC.unpack_from(frame, total - CRC_SIZE)
    calculated = crc32(frame[:total - CRC_SIZE])
    if given != calculated:
        raise FrameError(f"CRC 校验失败：收到 {given:08X}，算出 {calculated:08X}")
    return {"type": type_, "flags": flags, "seq": seq,
            "payload": frame[HEADER_SIZE:HEADER_SIZE + length]}


def split_frames(buffer: bytes) -> tuple[list[bytes], bytes]:
    """从 TCP 字节流里拆出所有完整帧，返回 (完整帧列表, 剩下的半帧)。

    粘包/半包/中间混进垃圾都能应付：magic 或 CRC 不对就往后挪一个字节重新找。
    """
    frames: list[bytes] = []
    data = buffer
    while True:
        start = data.find(MAGIC)
        if start < 0:
            return frames, b""                  # 连 magic 都没有，全丢
        if start:
            data = data[start:]
        if len(data) < HEADER_SIZE:
            return frames, data
        _magic, version, _type, _flags, _r8, _r16, _seq, length = HEADER.unpack_from(data, 0)
        if version != VERSION or length > MAX_PAYLOAD:
            data = data[1:]                     # magic 撞车了，往后挪一个字节再找
            continue
        total = HEADER_SIZE + length + CRC_SIZE
        if len(data) < total:
            return frames, data                 # 半帧，等下次 recv
        (given,) = CRC.unpack_from(data, total - CRC_SIZE)
        if given != crc32(data[:total - CRC_SIZE]):
            data = data[1:]                     # CRC 不对，重新同步
            continue
        frames.append(data[:total])
        data = data[total:]


# ---------------------------------------------------------------- 上行：图像

def encode_image_frame(jpeg: bytes, width: int, height: int, seq: int = 0,
                       fmt: int = IMG_JPEG, quality: int = 80,
                       stamp_ms: int = 0) -> bytes:
    """一张图 → 一整帧（Pi 侧调用）。"""
    if not jpeg:
        raise FrameError("图像字节是空的")
    meta = IMAGE_META.pack(_u16(width, "width"), _u16(height, "height"),
                           int(fmt) & 0xFF, max(0, min(int(quality), 100)) & 0xFF,
                           0, int(stamp_ms) & 0xFFFFFFFF)
    return build_frame(TYPE_IMAGE, seq, meta + jpeg)


def parse_image_frame(frame: bytes) -> dict:
    """整帧 → {'seq','width','height','format','quality','stamp_ms','jpeg'}。"""
    info = parse_frame(frame)
    if info["type"] != TYPE_IMAGE:
        raise FrameError(f"这不是图像帧（type={info['type']}）")
    payload = info["payload"]
    if len(payload) < IMAGE_META_SIZE:
        raise FrameError("图像帧载荷不够放元信息")
    width, height, fmt, quality, _rsv, stamp = IMAGE_META.unpack_from(payload, 0)
    jpeg = payload[IMAGE_META_SIZE:]
    if not jpeg:
        raise FrameError("图像帧里没有图像字节")
    return {"seq": info["seq"], "width": width, "height": height, "format": fmt,
            "quality": quality, "stamp_ms": stamp, "jpeg": jpeg}


# ---------------------------------------------------------------- 下行：结果

def _pack_slot(cls: int, valid: bool, conf, quad) -> bytes:
    if valid:
        if len(quad) != 4:
            raise FrameError(f"一个检测槽要 4 个点，收到 {len(quad)}")
        coords = []
        for point in quad:
            coords.append(_u16(point[0], "x"))
            coords.append(_u16(point[1], "y"))
        return SLOT.pack(int(cls) & 0xFF, 1, _conf_u16(conf), *coords)
    return SLOT.pack(0, 0, 0, *([0] * 8))


def encode_result_frame(ref_seq: int, detections=(), step: int = STEP_UNKNOWN,
                        flags=(0, 0, 0), hand=None, hand_box=None, seq: int = 0,
                        cost_us: int = 0, ack_seq: int = 0, rcv_nxt: int = 0,
                        win: int = 0) -> bytes:
    """识别结果 → 一整帧（PC 侧调用），恒为 160 字节。

    detections：可迭代，每项 `{"cls":0~3, "conf":0.873, "quad":[(x,y)×4]}`
                或 `{"cls":0~3, "conf":0.873, "box":(x1,y1,x2,y2)}`（自动变 4 个点）
                也接受现成的 `[x1,y1,x2,y2,cls,conf]` 列表（就是 server/model.py 的输出）。
    hand      ：`None` 或 `{"conf":0.8, "quad":[...]}` / `{"conf":0.8, "box":(...)}`。
    step      ：0=未知，1~4。
    flags     ：3 个 0/1，语义由业务定（见文件头）。
    cost_us   ：服务端推理耗时（微秒），树莓派状态栏直接显示这个数。
    """
    step = int(step)
    if step != STEP_UNKNOWN and not STEP_RANGE[0] <= step <= STEP_RANGE[1]:
        raise FrameError(f"step={step} 超出 0（未知）或 1~{STEP_RANGE[1]}")
    if len(flags) != 3:
        raise FrameError(f"flags 要正好 3 个，收到 {len(flags)}")

    items = list(detections)
    if len(items) > MAX_DET:                 # 多的截掉（调用方应先按置信度排序）
        items = items[:MAX_DET]

    payload = bytearray(RESULT_HEAD.pack(int(ref_seq) & 0xFFFFFFFF, len(items), 0, 0, 0,
                                         int(cost_us) & 0xFFFFFFFF))
    for index in range(MAX_DET):
        if index < len(items):
            item = items[index]
            if isinstance(item, (list, tuple)):          # [x1,y1,x2,y2,cls,conf]
                x1, y1, x2, y2, cls, conf = item
                payload += _pack_slot(int(cls), True, conf, quad_from_xyxy(x1, y1, x2, y2))
            else:
                quad = item.get("quad") or quad_from_xyxy(*item["box"])
                payload += _pack_slot(int(item.get("cls", 0)), True,
                                      item.get("conf", 0.0), quad)
        else:
            payload += _pack_slot(0, False, 0.0, None)

    if hand:                                             # 预留槽：现在装「阶段模型」的框
        quad = hand.get("quad") or quad_from_xyxy(*hand["box"])
        # cls 以前写死 0；现在允许填，用来传阶段模型的类别号（0=step_0 … 4=complete）
        payload += _pack_slot(int(hand.get("cls", 0)) & 0xFF, True,
                              hand.get("conf", 0.0), quad)
    else:
        payload += _pack_slot(0, False, 0.0, None)

    payload += STEP_BLOCK.pack(step & 0xFF, 0, 0, 0)
    payload += FLAG_BLOCK.pack(*[1 if flag else 0 for flag in flags], 0, 0, 0, 0, 0)
    # 尾部 6 个保留字节里塞一个压缩手框（4 个 u8，单位 4px）：检测槽被零件占满时，
    # 手框还能回给树莓派。全 0 表示"这一帧没有手"。
    packed = [0, 0, 0, 0]
    if hand_box is not None:
        packed = [max(0, min(int(round(float(v) / 4.0)), 255)) for v in hand_box[:4]]
    payload += TAIL_BLOCK.pack(ack_seq & 0xFFFFFFFF, rcv_nxt & 0xFFFFFFFF,
                               max(0, min(int(win), 0xFFFF)), *packed)

    if len(payload) != RESULT_PAYLOAD_SIZE:              # 结构体改了长度就会在这里炸
        raise FrameError(f"结果载荷 {len(payload)} 字节，应为 {RESULT_PAYLOAD_SIZE}")
    return build_frame(TYPE_RESULT, seq, bytes(payload))


def parse_result_frame(frame: bytes) -> dict:
    """整帧 → {'ref_seq','det_count','step','flags','detections','hand'}。

    detections 里每项：{'cls','name','conf','quad','xyxy'}；hand 为 None 或同结构。
    """
    info = parse_frame(frame)
    if info["type"] != TYPE_RESULT:
        raise FrameError(f"这不是结果帧（type={info['type']}）")
    payload = info["payload"]
    if len(payload) != RESULT_PAYLOAD_SIZE:
        raise FrameError(f"结果载荷 {len(payload)} 字节，应为 {RESULT_PAYLOAD_SIZE}")

    ref_seq, det_count, _r1, _r2, _r3, cost_us = RESULT_HEAD.unpack_from(payload, 0)
    if det_count > MAX_DET:
        raise FrameError(f"det_count={det_count} 超过 {MAX_DET}")

    offset = RESULT_HEAD.size
    slots = []
    for _ in range(MAX_DET):                             # 检测槽
        slots.append(SLOT.unpack_from(payload, offset))
        offset += SLOT_SIZE
    hand_raw = SLOT.unpack_from(payload, offset)         # 手部槽
    offset += SLOT_SIZE
    step, _s1, _s2, _s3 = STEP_BLOCK.unpack_from(payload, offset)
    offset += STEP_BLOCK.size
    b0, b1, b2, _f1, _f2, _f3, _f4, _f5 = FLAG_BLOCK.unpack_from(payload, offset)
    offset += FLAG_BLOCK.size
    ack_seq, rcv_nxt, win, hx1, hy1, hx2, hy2 = TAIL_BLOCK.unpack_from(payload, offset)
    offset += TAIL_SIZE
    hand_box = (None if not any((hx1, hy1, hx2, hy2))
                else (hx1 * 4, hy1 * 4, hx2 * 4, hy2 * 4))

    if step != STEP_UNKNOWN and not STEP_RANGE[0] <= step <= STEP_RANGE[1]:
        raise FrameError(f"step={step} 不在 0 或 1~{STEP_RANGE[1]}")
    for flag in (b0, b1, b2):
        if flag not in (0, 1):
            raise FrameError(f"布尔位只能是 0/1，收到 {flag}")

    return {"seq": info["seq"], "ref_seq": ref_seq, "det_count": det_count,
            "step": step, "flags": (b0, b1, b2),
            "ack_seq": int(ack_seq), "rcv_nxt": int(rcv_nxt), "win": int(win),
            "cost_us": cost_us, "cost_ms": cost_us / 1000.0,
            "detections": [_unpack_slot(slot) for slot in slots[:det_count]],
            "hand": _unpack_slot(hand_raw) if hand_raw[1] else None,
            "hand_box": hand_box}


def _unpack_slot(slot) -> dict:
    """一个 20 字节槽 → dict。slot = (cls, valid, conf, x0,y0,...,x3,y3)。"""
    cls, valid, conf = slot[0], slot[1], slot[2]
    quad = [(slot[3 + i * 2], slot[4 + i * 2]) for i in range(4)]
    return {"cls": int(cls),
            "name": CLS_NAMES[cls] if 0 <= cls < len(CLS_NAMES) else f"cls{cls}",
            "valid": bool(valid), "conf": conf / 1000.0,
            "quad": quad, "xyxy": xyxy_from_quad(quad)}


# ---------------------------------------------------------------- ACK 帧（type=3）
def encode_ack_frame(ack_seq: int, rcv_nxt: int = 0, win: int = 0, seq: int = 0) -> bytes:
    """组一个独立的确认帧（32 字节）。

    走 HTTP 时一般用不到（结果帧自带 ack）；换成裸 TCP 流、或者服务端想
    "只确认不推理"时用它，省一次推理 + 一个 160 字节结果帧。
    """
    payload = ACK_PAYLOAD.pack(int(ack_seq) & 0xFFFFFFFF, int(rcv_nxt) & 0xFFFFFFFF,
                               max(0, min(int(win), 0xFFFF)))
    return build_frame(TYPE_ACK, seq, payload)


def parse_ack_frame(frame: bytes) -> dict:
    """解析确认帧 → {'seq','ack_seq','rcv_nxt','win'}。"""
    info = parse_frame(frame)
    if info["type"] != TYPE_ACK:
        raise FrameError(f"这不是 ACK 帧（type={info['type']}）")
    payload = info["payload"]
    if len(payload) != ACK_PAYLOAD.size:
        raise FrameError(f"ACK 载荷 {len(payload)} 字节，应为 {ACK_PAYLOAD.size}")
    ack_seq, rcv_nxt, win = ACK_PAYLOAD.unpack_from(payload, 0)
    return {"seq": info["seq"], "ack_seq": int(ack_seq),
            "rcv_nxt": int(rcv_nxt), "win": int(win)}


# ---------------------------------------------------------------- 调试

def describe(frame: bytes) -> str:
    """一行摘要，抓包/打日志时看。"""
    try:
        info = parse_frame(frame)
    except FrameError as error:
        return f"坏帧（{len(frame)} 字节）：{error}"
    kind = {TYPE_IMAGE: "图像", TYPE_RESULT: "结果",
            TYPE_ACK: "应答", TYPE_ERROR: "错误"}.get(info["type"], f"type{info['type']}")
    if info["type"] == TYPE_IMAGE:
        body = parse_image_frame(frame)
        return (f"{kind} seq={body['seq']} {body['width']}x{body['height']}"
                f" fmt={body['format']} q={body['quality']}"
                f" jpeg={len(body['jpeg'])}B 共 {len(frame)}B")
    if info["type"] == TYPE_ACK:
        body = parse_ack_frame(frame)
        return (f"{kind} seq={info['seq']} ack={body['ack_seq']} "
                f"rcv_nxt={body['rcv_nxt']} win={body['win']} 共 {len(frame)}B")
    if info["type"] == TYPE_RESULT:
        body = parse_result_frame(frame)
        found = ", ".join(f"{d['name']}@{d['conf']:.2f}" for d in body["detections"]) or "无"
        hand = f"手@{body['hand']['conf']:.2f}" if body["hand"] else "无手"
        return (f"{kind} seq={info['seq']} ref={body['ref_seq']} 检出[{found}] {hand}"
                f" step={body['step']} 布尔={body['flags']} {body['cost_ms']:.1f}ms"
                f" 共 {len(frame)}B")
    return f"{kind} seq={info['seq']} 载荷 {len(info['payload'])}B"


# ---------------------------------------------------------------- 自检

def _self_test() -> int:
    """往返编解码 + 边界 + 坏帧拒收。全绿返回 0。"""
    failures = []

    def check(name, condition):
        print(f"  {'ok  ' if condition else 'FAIL'} {name}")
        if not condition:
            failures.append(name)

    print("上行图像帧：")
    jpeg = bytes(range(256)) * 40                    # 假装 10KB 的 JPEG
    frame = encode_image_frame(jpeg, 1280, 960, seq=7, quality=80, stamp_ms=12345)
    body = parse_image_frame(frame)
    check("整帧长度 = 16 + 12 + 图像 + 4", len(frame) == 16 + 12 + len(jpeg) + 4)
    check("字段原样回来", (body["width"], body["height"], body["quality"],
                          body["stamp_ms"], body["seq"], body["jpeg"])
          == (1280, 960, 80, 12345, 7, jpeg))

    print("下行结果帧：")
    boxes = [[100.6, 200.2, 300.9, 400.1, 2, 0.873],      # server/model.py 的原样输出
             {"cls": 0, "conf": 0.91, "box": (10, 20, 30, 40)}]
    result = encode_result_frame(7, boxes, step=3, flags=(1, 0, 1),
                                 hand={"conf": 0.66, "box": (5, 6, 7, 8)}, cost_us=12345)
    check("下行帧恒为 160 字节", len(result) == 160)
    back = parse_result_frame(result)
    check("ref_seq / det_count / step / flags",
          (back["ref_seq"], back["det_count"], back["step"], back["flags"])
          == (7, 2, 3, (1, 0, 1)))
    check("服务端耗时透传", abs(back["cost_ms"] - 12.345) < 1e-6)
    first = back["detections"][0]
    check("检测 0 类别与置信度", (first["name"], first["conf"]) == ("compass", 0.873))
    check("检测 0 四个角点",
          first["quad"] == [(101, 200), (301, 200), (301, 400), (101, 400)])
    check("检测 1 轴对齐框", back["detections"][1]["xyxy"] == (10, 20, 30, 40))
    check("手部槽", back["hand"]["xyxy"] == (5, 6, 7, 8)
          and abs(back["hand"]["conf"] - 0.66) < 1e-6)

    print("空帧 / 边界：")
    empty = parse_result_frame(encode_result_frame(9))
    check("没有目标时 det_count=0", empty["det_count"] == 0 and empty["detections"] == [])
    check("没手时 hand=None", empty["hand"] is None)
    try:
        encode_result_frame(1, [], step=5)
        check("step=5 应报错", False)
    except FrameError:
        check("step=5 报错", True)
    try:
        encode_image_frame(jpeg, 70000, 960, seq=1)
        check("宽度超 u16 应报错", False)
    except FrameError:
        check("宽度超 u16 报错", True)
    four = encode_result_frame(1, [{"cls": i % 4, "conf": 0.5, "box": (i, i, i + 9, i + 9)}
                                   for i in range(9)])
    check("多于 4 个只保留 4 个", parse_result_frame(four)["det_count"] == 4)

    print("坏帧：")
    broken = bytearray(result)
    broken[-1] ^= 0xFF
    try:
        parse_result_frame(bytes(broken))
        check("CRC 坏了应拒收", False)
    except FrameError as error:
        check(f"CRC 坏了拒收（{error}）", "CRC" in str(error))
    try:
        parse_result_frame(encode_image_frame(jpeg, 640, 480))
        check("图像帧当结果帧解析应报错", False)
    except FrameError:
        check("类型不符报错", True)

    print("ACK 字段 / ACK 帧：")
    acked = parse_result_frame(encode_result_frame(5, boxes, step=2, ack_seq=5,
                                                   rcv_nxt=7, win=4))
    check("结果帧回带 ack_seq/rcv_nxt/win",
          (acked["ack_seq"], acked["rcv_nxt"], acked["win"]) == (5, 7, 4))
    check("加了 ack 仍是 160 字节", len(encode_result_frame(5, boxes, ack_seq=5)) == 160)
    ack = parse_ack_frame(encode_ack_frame(9, rcv_nxt=11, win=6))
    check("ACK 帧往返", (ack["ack_seq"], ack["rcv_nxt"], ack["win"]) == (9, 11, 6))
    check("ACK 帧恒 32 字节", len(encode_ack_frame(9, 11, 6)) == 32)

    print("TCP 拆包：")
    stream = encode_image_frame(jpeg, 1280, 960, seq=1) + b"junk" + result + result[:50]
    frames, rest = split_frames(stream)
    check("两整帧 + 半帧", len(frames) == 2 and rest == result[:50])
    frames, rest = split_frames(rest + result[50:])
    check("补齐后半帧能拼回", len(frames) == 1 and rest == b"")

    print("\n" + ("全部通过" if not failures else f"失败 {len(failures)} 项：{failures}"))
    return 0 if not failures else 1


def _print_format() -> int:
    """打印当前帧格式（文件头的字节表），方便对照抓包。"""
    doc = __doc__ or ""
    start = doc.find("============================ 通用帧头")
    print(doc[start:] if start >= 0 else doc)
    return 0


if __name__ == "__main__":
    if "--format" in sys.argv:
        sys.exit(_print_format())
    sys.exit(_self_test())
