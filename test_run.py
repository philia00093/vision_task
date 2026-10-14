"""右目视图 + 绿色区域识别（单目，无拼接、无测距）。

相机部分针对 USB 2.0 双目模组（如 HBV-1714-2）的卡顿问题：
    · 先设分辨率再设 FOURCC=MJPG，避免驱动回退到 YUY2（2560x960 时仅 5 FPS）；
    · 可用 --fps / --fourcc / --buffersize 调整，启动即打印实际协商到的格式与帧率；
    · 状态栏实时显示实测帧率，便于判断是否落入低帧率模式；
    · --layout none 用于"整帧就是单目画面"的模式（如 1280x960 单目输出），不做对半切；
    · 取的是**右目**画面（双目拼接帧的后半幅）。

识别部分（纯 OpenCV，不训练、不加载任何模型）：
    · 绿色区域：OpenCV 在 HSV 空间直接分割绿色料盘，不需要训练、不需要模型。
      画面里**固定只有两个框**：右侧绿色料盘 = 红框「物料区」，左侧 = 蓝框「装配区」。
      桌面上的小绿色杂物会被面积门槛滤掉，不会多出第三个框。
      中文标签用 PIL + 系统字体绘制（cv2.putText 画不出中文，会变成问号）。
    · 工位状态：只按物料区判断 —— 物料区被手遮住显示「取料中」，否则显示「装配中」，
      固定显示在整个屏幕的右上角。

用法：
    python test_run.py                                  # 右目实时预览 + 识别
    python test_run.py --no-detect                      # 只看画面，不做识别
    python test_run.py --mode 2560x720 --fps 60         # 换更省带宽的模式
    python test_run.py --layout none                    # 整帧即单目，不对半切
    python test_run.py --snapshot                       # 抓一帧保存后退出（无 GUI 可用）
    python test_run.py --probe                          # 只测各模式的实测帧率，不显示
    python test_run.py --detect-width 960               # 识别用 960 宽，提高帧率
    python test_run.py --green-lo 30 60 80 --min-area 0.01   # 光照不同时调绿色阈值
热键：S 保存当前帧（含识别结果），Q / ESC 退出。

路径规则：
    · 项目根目录（权重 / configs / output 的基准）按以下顺序确定：
      环境变量 VESION_TASK_ROOT → 脚本所在目录 → 向上找到带 weights/ 或 configs/ 的目录
      → 当前工作目录。启动会打印实际用的根目录与输出目录。
    · --source 给的相对路径先按当前工作目录找，找不到再按项目根目录找。
    · --save-dir 的相对路径以项目根目录为基准，不存在会自动创建。
"""

import argparse
import os
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from stereo_fusion import save_image, split_stereo

def find_root(start: Path | None = None) -> Path:
    """确定项目根目录：权重、configs、output 都以它为基准。

    依次尝试：
        1. 环境变量 VESION_TASK_ROOT —— 显式指定，只要目录存在就以它为准，优先级最高
        2. 脚本所在目录（若其下有 weights/ 或 configs/）
        3. 从脚本目录逐级向上找带 weights/ 或 configs/ 的目录
        4. 当前工作目录
        5. 兜底：脚本所在目录
    """
    # 1) 显式指定优先：存在即采纳，避免被脚本目录"抢答"
    env = os.environ.get("VESION_TASK_ROOT")
    if env:
        candidate = Path(env).expanduser()
        if candidate.is_dir():
            return candidate.resolve()

    here = (start or Path(__file__)).resolve().parent
    # 2) + 3) 脚本目录及其各级父目录
    for candidate in (here, *here.parents):
        try:
            if (candidate / "weights").is_dir() or (candidate / "configs").is_dir():
                return candidate
        except OSError:
            continue
    # 4) 当前工作目录
    try:
        cwd = Path.cwd()
        if (cwd / "weights").is_dir() or (cwd / "configs").is_dir():
            return cwd
    except OSError:
        pass
    # 5) 兜底
    return here


def ensure_dir(path: Path, what: str = "目录") -> Path:
    """确保目录存在；建不出来时给出可操作的报错，而不是让后面写到一半才失败。"""
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise SystemExit(f"无法创建{what}：{path}\n  原因：{error}\n"
                         f"  可用 --save-dir 指定一个可写目录，或设置环境变量 VESION_TASK_ROOT")
    return path


ROOT = find_root()
WINDOW = "Right Eye View"
BACKENDS = {"auto": cv2.CAP_ANY, "dshow": cv2.CAP_DSHOW,
            "msmf": getattr(cv2, "CAP_MSMF", cv2.CAP_ANY)}
# 规格书列出的模式（宽, 高）＋推荐的 MJPEG 帧率
CANDIDATE_MODES = [(2560, 960, 30), (2560, 720, 60), (1280, 960, 60), (1280, 720, 60),
                   (960, 960, 60), (1280, 480, 60), (640, 480, 60)]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# ---------------------------------------------------------------- 识别相关常量
# 绿色 HSV 范围：实测绿板 H≈75 S≈120 V≈224-255。
# S/V 下限抬高是为了不把深色地板、深蓝布料误判成绿色。
DEFAULT_GREEN_LO = (35, 80, 100)
DEFAULT_GREEN_HI = (95, 255, 255)
GREEN_MIN_AREA_RATIO = 0.004        # 连通域最小面积（占整幅画面的比例）
GREEN_ZONE_MIN_AREA_RATIO = 0.012   # 被当作"区域"的最小面积；比这小的绿色算杂物，
                                    # 否则桌面上的小绿物件会抢走物料区/装配区的名字
GREEN_MIN_FILL_RATIO = 0.55         # 外接矩形填充率下限，滤掉细长杂散色块

# 区域配色（BGR）：画面里只有两个区域 —— 物料区（红框，在画面右侧）、
# 装配区（蓝框，在画面左侧）。索引顺序 = 从左到右的排列顺序。
ZONE_COLORS = [(255, 0, 0),      # 左侧：装配区（蓝）
               (0, 0, 255)]      # 右侧：物料区（红）
STATE_COLOR = (0, 255, 255)      # 状态文字（取料中/装配中）：黄底黑字
# 中文标签字体（cv2.putText 不支持中文，用 PIL 画）；按顺序找第一个存在的
CJK_FONTS = ("C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf",
             "C:/Windows/Fonts/simsun.ttc", "C:/Windows/Fonts/Deng.ttf")
LABEL_FONT_SIZE = 22

# ------------------------------------------------------------------ 相机部分
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


def take_right(frame, layout):
    """取出右目画面：layout=none 表示整帧即单目，不做拆分。

    双目拼接帧里右目是后半幅，所以取 split_stereo 的索引 1（左目是索引 0）。
    """
    if layout == "none":
        return frame
    return split_stereo(frame, layout)[1]


def draw_overlay(image, text, height=30):
    """在 image 上就地画顶部状态栏。

    注意：为了不与标签定位混淆，本函数返回**状态栏高度**（int），不返回图像。
    需要图像的地方直接用传入的 image（它是就地修改的）。
    """
    cv2.rectangle(image, (0, 0), (image.shape[1], height), (0, 0, 0), -1)
    cv2.putText(image, text, (8, int(height * 0.72)), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 255, 255), 1, cv2.LINE_AA)
    return height


# ------------------------------------------------------------ 绿色区域识别
def detect_green_regions(frame, lo, hi, min_area_ratio=GREEN_MIN_AREA_RATIO,
                         min_fill_ratio=GREEN_MIN_FILL_RATIO, morph=7, merge_gap=0,
                         mask=None):
    """HSV 分割绿色料盘，按 x 排序返回 [(x1, y1, x2, y2, area), ...]。

    纯 OpenCV，不训练、不依赖模型；左边第一块=物料区，第二块=装配区。
    mask 参数可以复用外部已经算好的绿色掩码，避免重复做 HSV 转换。
    """
    if mask is None:
        mask = green_mask(frame, lo, hi, morph)
    return contours_to_boxes(mask, frame.shape[1], frame.shape[0], min_area_ratio,
                             min_fill_ratio, merge_gap)


def green_mask(frame, lo, hi, morph=7):
    """绿色区域的二值掩码（0/255），供区域识别与遮挡判断共用。"""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
    if morph > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph, morph))
        # 先开（去噪点）后闭（补反光造成的空洞）
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    return mask


def contours_to_boxes(mask, width, height, min_area_ratio=GREEN_MIN_AREA_RATIO,
                      min_fill_ratio=GREEN_MIN_FILL_RATIO, merge_gap=0):
    """掩码 → 按 x 排序的框列表 [(x1, y1, x2, y2, area), ...]。"""
    min_area = min_area_ratio * width * height
    boxes = []
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if w <= 0 or h <= 0 or area / float(w * h) < min_fill_ratio:
            continue
        boxes.append([x, y, x + w, y + h, area])

    if merge_gap > 0 and len(boxes) > 1:            # 料盘被反光/手切成两块时合并
        boxes.sort(key=lambda b: b[0])
        merged = [boxes[0]]
        for box in boxes[1:]:
            last = merged[-1]
            if box[0] - last[2] <= merge_gap:
                last[2], last[3] = max(last[2], box[2]), max(last[3], box[3])
                last[1], last[4] = min(last[1], box[1]), last[4] + box[4]
            else:
                merged.append(box)
        boxes = merged

    boxes.sort(key=lambda b: b[0])                  # 左 → 右
    return [tuple(b) for b in boxes]


class ZoneTracker:
    """固定只跟踪两个区域：物料区（左）、装配区（右）。

    框的数量：**永远只有两个**。每帧把检测到的绿色块按 x 排序，最左的并入物料区、
    次左的并入装配区；同一区域匹配到多个碎块时取并集。多出来的绿色块直接忽略，
    所以画面里再出现别的小绿块也不会多画框、不会留下"框残留"。

    框的尺寸：每帧都按当前实际检测到的外接矩形画，跟着实际轮廓走，不锁定。

    遮挡判断：以"第一次检测到该区域时的绿色像素数"作为基准面积，之后每帧比较当前
    可见绿色像素数，低于 基准 × --occlude-ratio 判为被遮挡。基准只记录一次、不再更新，
    所以手伸进去导致的面积骤降能被检出。

    状态规则：只按**物料区**判断 —— 被遮挡显示"取料中"，否则"装配中"。
    """

    # 画面里从左到右依次是：装配区、物料区（用户确认物料区在右）
    # 只有这两个固定区域，多出来的绿色块一律忽略
    NAMES = ("装配区", "物料区")

    def __init__(self, occlude_ratio=0.75):
        self.occlude_ratio = float(occlude_ratio)
        self.zones: list[dict] = []      # 最多两个 {name, box, anchor, baseline, ...}
        self.frame = 0

    def update(self, boxes, mask) -> list[dict]:
        """boxes: 当前帧的绿色块（已按 x 排序并过滤小杂物）；mask: 同一帧的绿色掩码。"""
        self.frame += 1
        candidates = sorted(boxes, key=lambda b: b[0])

        while len(self.zones) < 2:                # 固定两个区域：物料区、装配区
            self.zones.append({"name": self.NAMES[len(self.zones)], "box": None,
                               "anchor": None, "baseline": 0.0, "occluded": False,
                               "visible_ratio": 1.0})

        # 第一次检测到两块时，最左=装配区、次左=物料区，并记住各自锚点
        if candidates and any(z["anchor"] is None for z in self.zones):
            for zone, box in zip(self.zones, candidates[:2]):
                if zone["anchor"] is None:
                    zone["anchor"] = tuple(int(v) for v in box[:4])

        groups = self._group_by_anchor(candidates)
        for zone, group in zip(self.zones, groups):
            if not group:
                continue                          # 该侧这一帧没检测到：保留上次结果
            x1 = min(int(b[0]) for b in group)
            y1 = min(int(b[1]) for b in group)
            x2 = max(int(b[2]) for b in group)
            y2 = max(int(b[3]) for b in group)
            zone["box"] = (x1, y1, x2, y2)

            patch = mask[max(0, y1):y2, max(0, x1):x2]
            visible = float(np.count_nonzero(patch)) if patch.size else 0.0
            if not zone["baseline"]:              # 首次检测：记绿色基准量
                zone["baseline"] = visible
            # 上限钳到 1.0：框是碎块并集，可能比基准大，比例不该超过 1
            zone["visible_ratio"] = min(visible / max(zone["baseline"], 1.0), 1.0)
            zone["occluded"] = zone["visible_ratio"] < self.occlude_ratio
        return self.zones

    def _group_by_anchor(self, candidates) -> list[list]:
        """把候选绿块按"离哪个锚点近"归到物料区/装配区，手切断产生的碎块也回到本区。

        用锚点（首次检测到的位置）而不是简单排序，是因为手遮挡会把料盘的绿色
        切成几块，纯按 x 排序会把切下来的碎块算到隔壁区域去。
        """
        anchors = [z["anchor"] for z in self.zones]
        if all(a is None for a in anchors):       # 还没有锚点：退化为左/右两块
            return [candidates[:1], candidates[1:2]]
        if any(a is None for a in anchors):       # 只认出一个：全给已锚定的那个
            known = anchors.index(next(a for a in anchors if a is not None))
            return [[c for c in candidates], []] if known == 0 else [[], list(candidates)]

        groups: list[list] = [[], []]
        for box in candidates:
            cx = (box[0] + box[2]) / 2.0
            distances = [abs(cx - (a[0] + a[2]) / 2.0) for a in anchors]
            groups[distances.index(min(distances))].append(box)
        return groups

    def drawable(self) -> list[dict]:
        """只返回已经有框的区域，供绘制使用。

        区域是在第一次检测到绿色块时创建的，那一帧 box 还是 None；
        直接解包 zone["box"] 会抛 TypeError: cannot unpack non-iterable NoneType。
        """
        return [z for z in self.zones if z["box"] is not None]

    def state_text(self, zone) -> str:
        """单个区域的遮挡状态（仅用于内部判断，不直接显示）。"""
        if not zone["occluded"]:
            return ""
        return "取料中" if zone["name"] == "物料区" else (
            "装配中" if zone["name"] == "装配区" else f"{zone['name']}被遮挡")

    def machine_state(self) -> str:
        """工位状态：**只按物料区判断**。

        逻辑（按你的说明）：物料区被手遮住 = 正在取料，可以据此确认装配是否正确
        → 显示"取料中"；物料区没有被遮 → 显示"装配中"。
        """
        material = next((z for z in self.zones if z["name"] == "物料区"), None)
        if material is None:
            return "等待物料区"
        return "取料中" if material["occluded"] else "装配中"


def load_cjk_font(size=LABEL_FONT_SIZE):
    """载入系统中文字体；不可用时返回 None（调用方退回 cv2 英文标签）。"""
    try:
        from PIL import ImageFont
    except ImportError:
        return None
    for path in CJK_FONTS:
        if Path(path).is_file():
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return None


_CJK_FONT_CACHE = {}
_LABEL_CACHE = {}


def draw_text_box(image, text, x, y, color, font=None, pad=4, right=False):
    """在 (x, y) 画一个底色实心、文字为黑的中文标签块。

    cv2.putText 不能画中文（会渲染成 ???），所以用 PIL 渲染后贴回 BGR 图。
    渲染结果按 (文本, 颜色, 字号) 缓存，实时循环里几乎不产生额外开销。
    right=True 时 x 视为贴片右边缘（用于贴画面右边的提示）。
    """
    font = font or _CJK_FONT_CACHE.get("font")
    if font is None and "font" not in _CJK_FONT_CACHE:
        font = load_cjk_font()
        _CJK_FONT_CACHE["font"] = font
    if font is None or not text:
        return image

    key = (text, tuple(int(c) for c in color), font.size)
    cached = _LABEL_CACHE.get(key)
    if cached is None:
        try:
            from PIL import Image, ImageDraw
        except ImportError:
            return image
        # 先量文字尺寸
        probe = Image.new("RGB", (8, 8))
        left, top, right, bottom = ImageDraw.Draw(probe).textbbox((0, 0), text, font=font)
        width, height = max(1, right - left), max(1, bottom - top)
        patch = Image.new("RGB", (width + pad * 2, height + pad * 2),
                          (int(color[2]), int(color[1]), int(color[0])))   # BGR → RGB
        ImageDraw.Draw(patch).text((pad - left, pad - top), text, font=font, fill=(0, 0, 0))
        cached = cv2.cvtColor(np.array(patch), cv2.COLOR_RGB2BGR)
        _LABEL_CACHE[key] = cached

    height, width = image.shape[:2]
    patch_h, patch_w = cached.shape[:2]
    if right:
        x = int(x) - patch_w                     # 右对齐：x 是贴片右边缘
    else:
        # 左对齐：框太靠画面左边缘时标签会溢出，往右挪到刚好放得下
        x = max(0, min(int(x), max(0, width - patch_w)))
    x = max(0, min(x, max(0, width - patch_w)))
    y = max(0, min(int(y), max(0, height - patch_h)))
    image[y:y + patch_h, x:x + patch_w] = cached
    return image


def _measure_label(text, font):
    """量出标签贴片尺寸（与 draw_text_box 的贴片一致）。"""
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return 0, 0
    probe = Image.new("RGB", (8, 8))
    left, top, right, bottom = ImageDraw.Draw(probe).textbbox((0, 0), text, font=font)
    return max(1, right - left) + 8, max(1, bottom - top) + 8


def draw_zone(image, x1, y1, x2, y2, name, color, font=None, top_margin=0):
    """画一个区域框 + 左上角标签。每个区域只画一层框，不额外加装饰框。"""
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

    label = f"{name} {x2 - x1}x{y2 - y1}"
    safe_top = max(int(top_margin), 2)          # 状态栏是最后画的，标签必须避开它
    label_h = _measure_label(label, font)[1] if font is not None else 30
    # 标签放框的左上角：优先框外上方，空间不够就贴框内左上角
    text_y = y1 - label_h if y1 - label_h >= safe_top else max(y1 + 2, safe_top + 2)
    draw_text_box(image, label, x1 + 2, text_y, color, font=font)
    return image


def draw_machine_state(image, text, font=None, color=STATE_COLOR, margin=10, top=0):
    """把工位状态提示画在**整个屏幕的右上角**（不是某个框上面）。

    位置：状态栏正下方再让开一行，避免和贴着顶部的「装配区」标签挤在一起。
    """
    if not text:
        return image
    patch_h = _measure_label(text, font)[1] if font is not None else 30
    y = max(int(top), 0) + patch_h + 10
    draw_text_box(image, text, image.shape[1] - margin, y, color, font=font, right=True)
    return image


# --------------------------------------------------------------------- 主流程
# 绿色区域识别相关的命令行开关，统一在 build_parser 里注册
DETECT_ARGUMENTS = (
    (("--detect-width",), dict(type=int, default=960,
                               help="识别前把画面缩到该宽度（0=不缩放，用原分辨率，较慢）")),
    (("--detect-interval",), dict(type=int, default=1,
                                  help="每 N 帧做一次识别（>1 可提帧率，中间帧沿用上次结果）")),
    (("--green-lo",), dict(type=int, nargs=3, metavar=("H", "S", "V"),
                           default=list(DEFAULT_GREEN_LO), help="绿色 HSV 下限")),
    (("--green-hi",), dict(type=int, nargs=3, metavar=("H", "S", "V"),
                           default=list(DEFAULT_GREEN_HI), help="绿色 HSV 上限")),
    (("--min-area",), dict(type=float, default=GREEN_MIN_AREA_RATIO,
                           help="绿色块最小面积占整幅画面的比例（连通域过滤）")),
    (("--zone-min-area",), dict(type=float, default=GREEN_ZONE_MIN_AREA_RATIO,
                                help="被当作「区域」的最小面积占整幅画面的比例；小于它的绿色"
                                     "杂物会被忽略，避免它抢走物料区/装配区的框")),
    (("--merge-gap",), dict(type=int, default=0,
                            help="水平间距小于该值的绿色块合并（料盘被反光切开时用）")),
    (("--occlude-ratio",), dict(type=float, default=0.75,
                                help="当前绿色可见量低于「首次检测量×该比例」判为被遮挡"
                                     "（显示取料中/装配中）；实测手遮住料盘一部分时约 0.66")),
)


def build_parser():
    parser = argparse.ArgumentParser(
        description="右目视图 + 绿色区域识别",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--camera", type=int, default=0, help="相机序号（见 list_cameras.py）")
    parser.add_argument("--source", default=None,
                        help="直接指定视频文件或图片路径；给了就不用相机（离线验证用）")
    parser.add_argument("--backend", choices=tuple(BACKENDS), default="dshow")
    parser.add_argument("--mode", default="2560x960",
                        help="请求分辨率，形如 2560x960（默认；若卡顿可试 2560x720 或 1280x960）")
    parser.add_argument("--width", type=int, help="直接指定帧宽（优先于 --mode）")
    parser.add_argument("--height", type=int, help="直接指定帧高")
    parser.add_argument("--fps", type=int, default=30, help="请求帧率（默认 30）")
    parser.add_argument("--fourcc", default="MJPG", help="像素格式，默认 MJPG（YUY2 会极卡）")
    parser.add_argument("--buffersize", type=int, default=1, help="驱动缓冲帧数，1 最省延迟")
    parser.add_argument("--layout", choices=("auto", "side_by_side", "top_bottom", "none"),
                        default="auto", help="auto 对半切；none 表示整帧即单目（右目）")
    parser.add_argument("--warmup", type=int, default=30, help="启动丢弃帧数")
    parser.add_argument("--resize", type=int, default=0, help="缩放到该宽度（0 不缩放）")
    parser.add_argument("--roi-px", type=int, nargs=4, metavar=("X", "Y", "W", "H"),
                        help="标出识别区域（像素，缩放前坐标）")
    parser.add_argument("--snapshot", action="store_true", help="抓一帧保存后退出")
    parser.add_argument("--save", action="store_true",
                        help="把带识别结果的画面存到 --save-dir（图片源存单张结果图）")
    parser.add_argument("--probe", action="store_true", help="逐模式实测帧率后退出")
    parser.add_argument("--fps-test", type=int, default=0, metavar="N",
                        help="读 N 帧并报告端到端实测帧率后退出")
    parser.add_argument("--save-dir", default="output")

    group = parser.add_argument_group("绿色区域识别")
    group.add_argument("--no-detect", action="store_true",
                       help="关掉识别，只做右目预览")
    for flags, options in DETECT_ARGUMENTS:
        group.add_argument(*flags, **options)
    return parser


def main():
    args = build_parser().parse_args()
    detect = not args.no_detect

    if args.probe:
        probe_modes(args.camera, args.backend, args.fourcc)
        return

    if args.width and args.height:
        width, height = args.width, args.height
    else:
        try:
            width, height = (int(v) for v in args.mode.lower().split("x"))
        except ValueError:
            raise SystemExit(f"--mode 格式应为 宽x高，例如 2560x960，收到 {args.mode!r}")

    # 视频源：--source 给的是视频/图片文件时走离线路径，否则开相机
    source_path = Path(args.source).expanduser() if args.source else None
    if source_path is not None:
        # 相对路径按"当前工作目录"解析；找不到再试项目根目录，并打印解析结果
        if not source_path.is_absolute() and not source_path.is_file():
            source_path = ROOT / source_path
        source_path = source_path.resolve()
    is_image = bool(source_path and source_path.suffix.lower() in IMAGE_SUFFIXES)
    is_video = bool(source_path and not is_image)
    if source_path and not source_path.is_file():
        raise SystemExit(f"找不到文件：{source_path}\n"
                         f"  已按「当前目录」和「项目根 {ROOT}」查找过；"
                         f"请检查路径或改用绝对路径")

    if is_image:
        frame0 = cv2.imdecode(np.fromfile(str(source_path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame0 is None:
            raise SystemExit(f"图片读取失败：{source_path}")
        info = {"width": frame0.shape[1], "height": frame0.shape[0], "fps": 0.0,
                "fourcc": "IMG"}
        cap = None
        print(f"图片源 {source_path.name} {info['width']}x{info['height']}")
    elif is_video:
        cap = cv2.VideoCapture(str(source_path))
        if not cap.isOpened():
            raise SystemExit(f"视频打不开：{source_path}")
        info = {"width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "fps": cap.get(cv2.CAP_PROP_FPS) or 0.0,
                "fourcc": fourcc_str(cap.get(cv2.CAP_PROP_FOURCC))}
        print(f"视频源 {source_path.name} {info['width']}x{info['height']} @{info['fps']:.1f}")
    else:
        cap, info = open_camera(args.camera, args.backend, width, height, args.fps,
                                args.fourcc, args.buffersize)
        print(f"请求 {width}x{height}@{args.fps} {args.fourcc} → "
              f"实际 {info['width']}x{info['height']} "
              f"@{info['fps']:.1f} {info['fourcc']}")
        if info["fourcc"].upper() != args.fourcc.upper():
            print(f"警告：相机未按 {args.fourcc} 协商（当前 {info['fourcc']}）；"
                  "YUY2 在 2560x960 下仅 5 FPS，会非常卡。可试 --backend msmf 或降分辨率。")
        if (info["width"], info["height"]) != (width, height):
            print(f"警告：实际分辨率与请求不一致，驱动可能回退到了默认模式")

    if args.fps_test and cap is not None:
        for _ in range(args.warmup):                      # 预热
            cap.read()
        count, start, shape = 0, time.time(), None
        for _ in range(args.fps_test):
            ok, frame = cap.read()
            if not ok:
                break
            view = take_right(frame, args.layout)          # 与实时流程相同的处理量
            if args.resize and args.resize != view.shape[1]:
                scale = args.resize / float(view.shape[1])
                view = cv2.resize(view, (args.resize, int(round(view.shape[0] * scale))),
                                  interpolation=cv2.INTER_AREA)
            _ = draw_overlay(view.copy(), "fps-test")     # 计入与实时流程相同的绘制开销
            count += 1
            shape = frame.shape
        elapsed = max(time.time() - start, 1e-6)
        print(f"端到端实测：{count} 帧 / {elapsed:.2f} s = {count / elapsed:.1f} FPS"
              f"（{info['width']}x{info['height']} {info['fourcc']}）")
        cap.release()
        return

    # 识别准备：绿色阈值全部来自命令行参数
    green_lo = tuple(args.green_lo)
    green_hi = tuple(args.green_hi)
    label_font = load_cjk_font() if detect else None
    if detect:
        print(f"绿色识别：HSV {green_lo}~{green_hi} | 最小面积 {args.min_area:.1%}"
              f" | 识别宽度 {args.detect_width or '原分辨率'}"
              f" | 间隔 {args.detect_interval} 帧")
        print(f"区域标签：左→红框「物料区」，右→蓝框「装配区」"
              f"，中文字体 {'已载入' if label_font else '未找到（标签用拼音代替）'}")

    save_dir = Path(args.save_dir).expanduser()
    if not save_dir.is_absolute():
        save_dir = ROOT / save_dir
    save_dir = ensure_dir(save_dir.resolve(), "输出目录")
    print(f"项目根目录：{ROOT}")
    print(f"输出目录  ：{save_dir}")
    show_window = (not args.snapshot) and not is_image and hasattr(cv2, "imshow")
    if show_window:
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    else:
        print("无 GUI 或使用 --snapshot/图片源：只保存图像，不显示窗口")

    frame_count, last_time, last_count, fps_now = 0, time.time(), 0, 0.0
    sharpness, brightness = None, 0.0
    tracker_zones = ZoneTracker(occlude_ratio=args.occlude_ratio)
    try:
        while True:
            if is_image:                                  # 图片源只处理一帧
                if frame_count:
                    break
                frame, frame_count = frame0, frame_count + 1
            else:
                ok, frame = cap.read()
                if not ok:
                    # 视频文件读完是正常结束；相机读不到才是真故障
                    print(f"视频结束，共处理 {frame_count} 帧" if is_video else "读取画面失败")
                    break
                frame_count += 1
                if frame_count <= args.warmup or float(frame[::8, ::8].mean()) < 12:
                    continue                              # 抽样亮度：比整帧 mean 快一个量级

            view = take_right(frame, args.layout)
            if args.resize and args.resize != view.shape[1]:
                scale = args.resize / float(view.shape[1])
                view = cv2.resize(view, (args.resize, int(round(view.shape[0] * scale))),
                                  interpolation=cv2.INTER_AREA)
            if args.roi_px:
                x, y, w, h = args.roi_px
                cv2.rectangle(view, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # ---------------- 识别：绿色区域 ----------------
            if detect and frame_count % max(args.detect_interval, 1) == 0:
                # 绿色掩码算一次，区域识别和遮挡判断共用
                mask = green_mask(view, green_lo, green_hi)
                boxes = contours_to_boxes(mask, view.shape[1], view.shape[0],
                                          min_area_ratio=args.min_area,
                                          merge_gap=args.merge_gap)
                # 桌面上的小绿杂物按面积滤掉，否则它会抢走"物料区"的框
                zone_floor = args.zone_min_area * view.shape[0] * view.shape[1]
                boxes = [b for b in boxes if b[4] >= zone_floor]
                zones = tracker_zones.update(boxes, mask)
            if detect:
                # 先占住状态栏区域拿到高度，区域标签与右上角提示据此避开顶部
                status_height = draw_overlay(view, "")
                for index, zone in enumerate(tracker_zones.drawable()):
                    x1, y1, x2, y2 = zone["box"]
                    color = ZONE_COLORS[index % len(ZONE_COLORS)]
                    draw_zone(view, x1, y1, x2, y2, zone["name"], color, font=label_font,
                              top_margin=status_height)
            now = time.time()
            if now - last_time >= 0.5:                      # 滑动窗口瞬时帧率（不含启动预热）
                fps_now = (frame_count - last_count) / max(now - last_time, 1e-6)
                last_time, last_count = now, frame_count
            gray = None
            if frame_count % 10 == 0 or sharpness is None:   # 清晰度每 10 帧算一次
                gray = cv2.cvtColor(view, cv2.COLOR_BGR2GRAY)
                sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
                brightness = float(gray.mean())
            status = (f"RIGHT EYE {view.shape[1]}x{view.shape[0]} {info['fourcc']} | "
                      f"fps {fps_now:.1f} | mean {brightness:.0f} | sharp {sharpness:.1f}")
            if detect:
                status += f" | 绿区 {len(tracker_zones.drawable())}"
            # 状态栏就地画在 view 上并盖掉刚才的占位黑条；
            # draw_overlay 返回的是高度（int），图像仍是 view 本身
            draw_overlay(view, status)
            annotate = view

            if detect:
                # 工位状态提示固定在整个屏幕的右上角（状态栏正下方）
                draw_machine_state(view, tracker_zones.machine_state(), font=label_font,
                                   top=status_height)

            if args.save:                                     # 存带识别结果的这一帧
                if is_image:
                    out = save_dir / f"识别结果_{source_path.stem}.png"
                else:
                    out = save_dir / f"结果_{datetime.now():%Y%m%d_%H%M%S}.jpg"
                save_image(out, annotate)
                if is_image or frame_count % 100 == 0:         # 视频源每 100 帧存一张
                    print(f"已保存：{out}")

            if args.snapshot:
                gray = cv2.cvtColor(view, cv2.COLOR_BGR2GRAY)
                out = save_dir / "右目画面.jpg"
                save_image(out, annotate)
                print(f"已保存：{out}")
                print(f"右目画面：{view.shape[1]} × {view.shape[0]}，格式 {info['fourcc']}，"
                      f"亮度均值 {float(gray.mean()):.1f}，"
                      f"清晰度 {float(cv2.Laplacian(gray, cv2.CV_64F).var()):.1f}")
                if detect:
                    print(f"识别结果：绿色区域 {len(tracker_zones.zones)} 个")
                break

            if show_window:
                cv2.imshow(WINDOW, annotate)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("s"):
                    out = save_dir / f"右目_{datetime.now():%Y%m%d_%H%M%S}.jpg"
                    save_image(out, annotate)              # 存带识别结果的那一帧
                    print(f"已保存：{out}")
    finally:
        if cap is not None:                               # 图片源没有 VideoCapture
            cap.release()
        if show_window:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
