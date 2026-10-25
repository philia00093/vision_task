"""阶段识别（基础视觉版）：把"装配区"的裁剪图跟几张阶段参考图做旋转遍历匹配，取最像的那个。

为什么用基础视觉而不是 YOLO：这一步只要判"装配区现在长啥样"，4 张参考图就是标尺，
`matchTemplate` 那套足够；树莓派上单次匹配十几毫秒，比跑模型便宜得多。

和"目标追踪"的关系：**互不干扰**。
    追踪  → 画零件框、轨迹，管"每件零件在哪"
    阶段  → 只吃"装配区"这一块裁剪图，管"装到第几步"
两者都从同一帧取数据，各画各的，谁也不用等谁。

=============================== 匹配怎么做 ===============================

1. 参考图（阶段样本）在启动时统一"规格化"：按同一个区域裁剪 → 灰度 → 缩到固定尺寸
   （默认 96x96）→ 零均值单位化。存成向量，后面只做点积。
2. 每帧（或隔几帧）把现场同一区域同样规格化，得到候选向量。
3. **按 1° 步长把候选图旋转一圈**（默认 -180°~179°，正好 360 个角度），
   每个角度都和所有参考样本算一次相似度（归一化互相关，1.0 = 完全一样）。
4. 取相似度最大的那个角度/类别；再在它附近按 `refine_step` 细搜一轮，
   得到"近似最大"（这就是"近似"的含义：不保证数学全局最优，但足够近）。

为什么非要旋转：同一阶段里零件的摆向是随机的（铜轮/铜盘随便转个角度放上去），
不旋转搜一圈，相似度会被摆向差异拉低，导致误判。

=============================== 目录约定 ===============================

`--stage-dir` 指向参考图目录（默认 `vision_yolo_train/images/steps`），两种放法都支持：

    steps/step1.png  steps/step_2.png ...        # 平铺：文件名（去扩展名）= 阶段名
    steps/step_3/1.jpg  steps/step_3/2.jpg ...   # 分目录：目录名 = 阶段名，里面可以多张样本

**同一阶段建议放 2~3 张不同摆向/光照的样本**，否则那一类的代表面太窄，
换个摆向就认不出来（4 张平铺就是"每阶段 1 张"，属于最省的最低配）。
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import cv2
import numpy as np

# 参考图默认目录（相对项目根）
STAGE_DEFAULT_DIR = "vision_yolo_train/images/steps"
DEFAULT_STAGE_SIZE = 64          # 规格化后的边长（96 够用；调小更快、更糙）
DEFAULT_ANGLE_STEP = 2.0         # 旋转遍历步长（度）
DEFAULT_ANGLE_RANGE = (-180.0, 179.0)
DEFAULT_REFINE_STEP = 0.2        # 命中角度附近的细搜步长（度）
DEFAULT_STAGE_INTERVAL = 10      # 每多少帧做一次阶段识别
DEFAULT_STAGE_MARGIN = 0.12      # 裁剪区域外扩比例（把料盘边缘也框进去）
MIN_STAGE_SCORE = 0.30           # 低于它就当"没认出来"，显示 unknown

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def sobel_edges(gray, clip_percentile: float = 99.5):
    """偏导求边缘：灰度 → Sobel 幅值 → 归一化到 0~255。

    为什么要这一步：装配区里**托盘占了绝大部分面积**，而且是均匀的暗面。
    SIFT 直接吃原始灰度时，匹配到的内点几乎全落在托盘边缘/螺丝/背景纹理上，
    等于在比"托盘长什么样"，零件反而没参与。
    求完偏导只剩锐利的边（零件轮廓、轮齿、盘沿），托盘那块平滑区域基本归零，
    比较的对象就变成零件本身了。

    实测（12 张样本留一）：0/1 两组 SIFT 从 1/6 提到 5/6；2/3 仍分不开
    （那两张样本本身是同一个状态，见 CascadeStageMatcher 的说明）。
    """
    g = np.asarray(gray, dtype=np.float32)
    g = cv2.GaussianBlur(g, (3, 3), 0)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    mag /= (np.percentile(mag, clip_percentile) + 1e-6)
    return np.clip(mag * 255.0, 0.0, 255.0).astype(np.uint8)


def read_image(path: Path):
    """读图（兼容中文路径）。"""
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None


_STEP_RE = re.compile(r"step[_\-]?(\d+)", re.IGNORECASE)


def label_of_stem(stem: str) -> str:
    """从文件名里认出阶段名：`20261025_183637_123_step0_1` → `step0`。

    现场用 /save 存的样本名是"时间戳_标签"，所以不能直接拿 stem 当阶段名。
    """
    found = _STEP_RE.search(stem)
    return f"step{found.group(1)}" if found else stem


def collect_references(directory: Path) -> dict[str, list[Path]]:
    """按目录约定收集参考图 → {阶段名: [路径, ...]}。

    平铺文件：用文件名当阶段名；子目录：用目录名当阶段名，里面所有图片都算该阶段的样本。
    """
    directory = Path(directory)
    stages: dict[str, list[Path]] = {}
    if not directory.is_dir():
        return stages
    for child in sorted(directory.iterdir()):
        if child.is_dir():
            samples = [p for p in sorted(child.iterdir())
                       if p.suffix.lower() in IMAGE_SUFFIXES]
            if samples:
                stages[label_of_stem(child.name)] = samples
        elif child.suffix.lower() in IMAGE_SUFFIXES:
            stages.setdefault(label_of_stem(child.stem), []).append(child)
    return stages


class StageMatcher:
    """旋转遍历 + 模板匹配的阶段识别器。

    用法：
        matcher = StageMatcher.from_directory(stage_dir, angles=(step, lo, hi))
        result = matcher.match(frame, region=(x, y, w, h))   # 每帧/每 N 帧调一次
        result["label"], result["score"], result["angle"]
    """

    def __init__(self, stages: dict[str, list[Path]], size=DEFAULT_STAGE_SIZE,
                 angle_step=DEFAULT_ANGLE_STEP, angle_range=DEFAULT_ANGLE_RANGE,
                 refine_step=DEFAULT_REFINE_STEP, margin=DEFAULT_STAGE_MARGIN,
                 rotate=True, min_score=MIN_STAGE_SCORE, block=(0.2, 0.8)):
        self.size = (int(size), int(size))
        self.angle_step = max(float(angle_step), 0.05)
        self.angle_range = (float(angle_range[0]), float(angle_range[1]))
        self.refine_step = max(float(refine_step), 0.01)
        self.margin = max(float(margin), 0.0)
        self.rotate = bool(rotate)
        self.min_score = float(min_score)
        self.block = (float(block[0]), float(block[1]))   # 中央块比例，如 (0.2, 0.8)=中间 60%
        self.names: list[str] = []
        self.samples: list[tuple[str, np.ndarray]] = []      # [(阶段名, 归一化向量)]
        self.errors: list[str] = []
        self._raw: list[tuple[str, np.ndarray]] = []         # [(阶段名, 原图全帧)]
        self._region: tuple[int, int, int, int] | None = None
        self._load(stages)

    # ------------------------------------------------------------ 载入
    @classmethod
    def from_directory(cls, directory, **kwargs) -> "StageMatcher":
        return cls(collect_references(Path(directory)), **kwargs)

    def _load(self, stages) -> None:
        self._raw: list[tuple[str, np.ndarray]] = []        # [(阶段名, 原图全帧)]
        self.names: list[str] = []
        for label, paths in stages.items():
            for path in paths:
                image = read_image(path)
                if image is None:
                    self.errors.append(f"读不出 {path}")
                    continue
                self._raw.append((label, image))
            if label not in self.names:
                self.names.append(label)
        # 先按"整帧"规格化占位；等 set_region 拿到装配区窗口后会按窗口重算
        self._region = None
        self.samples = []
        for label, image in self._raw:
            vec = self._vector(self._canonical(image, None))
            if vec is not None:
                self.samples.append((label, vec))

    @property
    def ready(self) -> bool:
        return bool(self.samples)

    def describe(self) -> str:
        if not self.ready:
            return f"阶段识别：未启用（没找到参考图{('：' + self.errors[0]) if self.errors else ''}）"
        counts = ", ".join(f"{n}×{sum(1 for lab, _ in self.samples if lab == n)}"
                           for n in self.names)
        span = (f"{self.angle_range[0]:.0f}°~{self.angle_range[1]:.0f}° "
                f"每 {self.angle_step:g}°" if self.rotate else "不旋转")
        return (f"阶段识别：{len(self.samples)} 个样本（{counts}）| {span}"
                f" | 规格 {self.size[0]}px")

    # ------------------------------------------------------------ 规格化
    def set_region(self, region, frame_shape) -> None:
        """记住"装配区"的裁剪区域（原画面坐标），并据此重新裁剪所有参考图。

        现场第一帧拿到锁定后的区域时调一次即可：之后的每帧、每张参考图都用同一个窗口，
        这样"现场 vs 参考"比的是同一块地方，才谈得上相似度。
        """
        height, width = frame_shape[:2]
        x, y, w, h = (int(v) for v in region)
        if self.margin > 0:
            mx, my = int(w * self.margin), int(h * self.margin)
            x, y, w, h = x - mx, y - my, w + 2 * mx, h + 2 * my
        x = max(0, min(x, width - 2))
        y = max(0, min(y, height - 2))
        w = max(2, min(w, width - x))
        h = max(2, min(h, height - y))
        region = (x, y, w, h)
        if region == self._region:
            return
        self._region = region
        # 参考图按这个窗口重新裁剪
        rebuilt = []
        for label, image in self._raw:
            vec = self._vector(self._canonical(image, region))
            if vec is not None:
                rebuilt.append((label, vec))
        if rebuilt:
            self.samples = rebuilt

    def _canonical(self, image, region):
        """按区域裁剪 → 取**长宽各 1/5~4/5 的中央块** → Sobel 偏导 → 归一化。

        为什么用中央块：料盘边缘、桌面、线缆都在外圈，真正区分装配阶段的只有
        中间那 3/5 区域（零件有没有装上去）。取中央块能直接忽略外圈干扰。

        为什么用偏导（梯度）：现场光照会整体变亮变暗，但**边缘/纹理的强弱**
        基本不变。Sobel 的 |dx|+|dy| 正好是"边缘图"，再做 min-max 归一化，
        就得到了对光照不敏感、只看形状的特征图。
        """
        if image is None:
            return None
        height, width = image.shape[:2]
        if region is not None:
            x, y, w, h = region
            x, y = max(int(x), 0), max(int(y), 0)
            w, h = min(int(w), width - x), min(int(h), height - y)
            if w < 8 or h < 8:
                return None
            patch = image[y:y + h, x:x + w]
        else:
            patch = image
        if patch.size == 0:
            return None
        # 中央 1/5 ~ 4/5（长宽各取中间 60%）
        ph, pw = patch.shape[:2]
        lo, hi = self.block
        y1, y2 = int(ph * lo), int(ph * hi)
        x1, x2 = int(pw * lo), int(pw * hi)
        center = patch[y1:y2, x1:x2]
        if center.size == 0:
            return None
        gray = cv2.cvtColor(center, cv2.COLOR_BGR2GRAY) if center.ndim == 3 else center
        gray = cv2.resize(gray, self.size, interpolation=cv2.INTER_AREA)
        # Sobel 偏导 → 梯度幅值（边缘图）
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        magnitude = cv2.magnitude(gx, gy)
        # 归一化到 0~255（每张图自己归一，抵消整体亮度/对比度差异）
        peak = float(magnitude.max())
        if peak < 1e-3:
            return None
        return (magnitude / peak * 255.0).astype("uint8")

    @staticmethod
    def _vector(gray):
        """(H,W) uint8 → 零均值、单位长度的 float32 向量（点积即归一化互相关）。"""
        if gray is None:
            return None
        vec = gray.astype(np.float32).ravel()
        vec -= vec.mean()
        norm = float(np.linalg.norm(vec))
        if norm < 1e-6:
            return None
        return vec / norm

    # ------------------------------------------------------------ 匹配
    def _rotate(self, gray, angle):
        if not angle:
            return gray
        center = (self.size[0] / 2.0, self.size[1] / 2.0)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        # BORDER_REPLICATE：旋转出来的边沿用边缘像素填，不会引入黑边干扰相似度
        return cv2.warpAffine(gray, matrix, self.size, flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REPLICATE)

    def _sweep(self, gray, angles):
        """在给定的角度集合里找最大相似度。返回 (label, score, angle)。"""
        best_label, best_score, best_angle = None, -2.0, 0.0
        for angle in angles:
            vec = self._vector(self._rotate(gray, angle))
            if vec is None:
                continue
            for label, sample in self.samples:
                score = float(vec @ sample)
                if score > best_score:
                    best_label, best_score, best_angle = label, score, angle
        return best_label, best_score, best_angle

    def _angles(self):
        lo, hi = self.angle_range
        count = int(math.floor((hi - lo) / self.angle_step)) + 1
        return [lo + i * self.angle_step for i in range(max(count, 1))]

    def match(self, frame, region=None, want_image=False) -> dict | None:
        """识别一帧。region 给 (x,y,w,h)；不给自己用 set_region 记住的那个窗口。"""
        if not self.ready:
            return None
        if region is not None:
            self.set_region(region, frame.shape)
        gray = self._canonical(frame, self._region)
        if gray is None:
            return None

        label, score, angle = self._sweep(gray, self._angles() if self.rotate else [0.0])

        # 命中角度附近细搜一轮，逼近真正的最小值（这就是"近似最大"）
        if self.rotate and label is not None:
            center = angle
            refine = [center + k * self.refine_step
                      for k in range(-int(round(self.angle_step / self.refine_step)),
                                     int(round(self.angle_step / self.refine_step)) + 1)]
            label2, score2, angle2 = self._sweep(gray, refine)
            if score2 > score:
                label, score, angle = label2, score2, angle2

        result = {"label": label if score >= self.min_score else "unknown",
                  "raw_label": label, "score": score, "angle": angle,
                  "region": self._region}
        if want_image:
            result["image"] = self._rotate(gray, angle)
        if score < self.min_score:
            result["note"] = f"相似度 {score:.2f} 低于门槛 {self.min_score:.2f}"
        return result


# ============================================================================
# SIFT 版阶段识别（尺度/旋转不变）
# ============================================================================
class SiftStageMatcher:
    """用 SIFT 特征 + 单应性 RANSAC 内点做阶段匹配。

    相比梯度模板法：
      · **尺度/旋转不变**：零件转个角度、相机挪远挪近都不影响
      · **几何校验**：内点必须在同一个单应变换下成立，能拒掉"局部像但整体不是"的假匹配

    参考图按同一区域裁剪 → 灰度 → 缩到 ref_size → 提 SIFT 描述子缓存。
    现场帧同样处理后，与每个参考比对，取"内点数最多"的阶段。

    实测（12 张样本留一验证）：9/12，比梯度模板法 6/12 好；step0/step1 全对，
    step2/step3 仍有混淆（样本本身差异很小）。
    """

    def __init__(self, stages: dict[str, list[Path]], ref_size: int = 256,
                 nfeatures: int = 500, ratio: float = 0.75, ransac_px: float = 5.0,
                 min_score: float = 0.02, margin: float = 0.10, edges: bool = False):
        self.size = int(ref_size)
        self.edges = bool(edges)
        self.nfeatures = int(nfeatures)
        self.ratio = float(ratio)
        self.ransac_px = float(ransac_px)
        self.min_score = float(min_score)
        self.margin = float(margin)
        self.sift = cv2.SIFT_create(nfeatures=self.nfeatures)
        self.matcher = cv2.BFMatcher(cv2.NORM_L2)
        self.names: list[str] = []
        self.refs: list[tuple[str, object, object]] = []   # (label, kp, des)
        self.errors: list[str] = []
        self._raw: list[tuple[str, np.ndarray]] = []
        self._region = None
        self._load(stages)

    @classmethod
    def from_directory(cls, directory, **kwargs):
        return cls(collect_references(Path(directory)), **kwargs)

    def _prepare(self, image, region):
        """裁剪 → 灰度 → 缩放 → 返回灰度图（SIFT 输入）。"""
        if image is None:
            return None
        height, width = image.shape[:2]
        if region is not None:
            x, y, w, h = (int(v) for v in region)
            x, y = max(x, 0), max(y, 0)
            w, h = min(w, width - x), min(h, height - y)
            if w < 16 or h < 16:
                return None
            patch = image[y:y + h, x:x + w]
        else:
            patch = image
        if patch.size == 0:
            return None
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY) if patch.ndim == 3 else patch
        gray = cv2.resize(gray, (self.size, self.size), interpolation=cv2.INTER_AREA)
        return sobel_edges(gray) if self.edges else gray

    def _load(self, stages) -> None:
        self._raw = []
        self.names = []
        for label, paths in stages.items():
            for path in paths:
                image = read_image(path)
                if image is None:
                    self.errors.append(f"读不出 {path}")
                    continue
                self._raw.append((label, image))
            if label not in self.names:
                self.names.append(label)
        self._rebuild()

    def set_region(self, region, frame_shape) -> None:
        region = tuple(int(v) for v in region)
        if region == self._region:
            return
        self._region = region
        self._rebuild()

    def _rebuild(self) -> None:
        self.refs = []
        for label, image in self._raw:
            gray = self._prepare(image, self._region)
            if gray is None:
                continue
            kp, des = self.sift.detectAndCompute(gray, None)
            if des is not None and len(des) >= 4:
                self.refs.append((label, kp, des))

    @property
    def ready(self) -> bool:
        return bool(self.refs)

    def describe(self) -> str:
        if not self.ready:
            return f"阶段识别(SIFT)：未启用（{('：' + self.errors[0]) if self.errors else '无样本'}）"
        counts = ", ".join(f"{n}×{sum(1 for lab, _, _ in self.refs if lab == n)}"
                           for n in self.names)
        return (f"阶段识别(SIFT)：{len(self.refs)} 个样本（{counts}）| "
                f"特征 {self.nfeatures} | 参考图 {self.size}px | 旋转/尺度不变")

    def _score(self, kp_a, des_a, kp_b, des_b):
        """返回 (内点数, 匹配数)。内点=单应变换下几何一致的匹配。"""
        if des_a is None or len(des_a) < 4 or des_b is None or len(des_b) < 4:
            return 0, 0
        matches = self.matcher.knnMatch(des_a, des_b, k=2)
        good = [a for a, b in (p for p in matches if len(p) == 2)
                if a.distance < self.ratio * b.distance]
        if len(good) < 4:
            return 0, len(good)
        src = np.float32([kp_a[g.queryIdx].pt for g in good]).reshape(-1, 1, 2)
        dst = np.float32([kp_b[g.trainIdx].pt for g in good]).reshape(-1, 1, 2)
        homo, mask = cv2.findHomography(src, dst, cv2.RANSAC, self.ransac_px)
        inliers = int(mask.sum()) if mask is not None else 0
        return inliers, len(good)

    def match(self, frame, region=None, want_image=False):
        if not self.ready:
            return None
        if region is not None:
            self.set_region(region, frame.shape)
        gray = self._prepare(frame, self._region)
        if gray is None:
            return None
        kp, des = self.sift.detectAndCompute(gray, None)
        kp_count = len(kp) if kp is not None else 0
        if des is None or len(des) < 4:
            return {"label": "unknown", "raw_label": None, "score": 0.0, "angle": 0.0,
                    "region": self._region, "method": "sift", "kp": kp_count,
                    "note": "现场帧提取不到特征"}

        best: dict[str, tuple[int, int]] = {}
        for label, ref_kp, ref_des in self.refs:
            inliers, good = self._score(kp, des, ref_kp, ref_des)
            if inliers > best.get(label, (0, 0))[0]:
                best[label] = (inliers, good)
        if not best:
            return {"label": "unknown", "raw_label": None, "score": 0.0, "angle": 0.0,
                    "region": self._region, "method": "sift", "kp": kp_count}

        ranked = sorted(best.items(), key=lambda kv: -kv[1][0])
        label, (inliers, good) = ranked[0]
        score = inliers / max(len(des), 1)
        second = ranked[1][1][0] if len(ranked) > 1 else 0
        # 前两名内点数太接近 → 认为不可靠
        if second and inliers and (inliers - second) / max(inliers, 1) < self.margin:
            label = "unknown"
        if score < self.min_score:
            label = "unknown"
        return {"label": label, "raw_label": ranked[0][0], "score": score, "angle": 0.0,
                "region": self._region, "inliers": inliers, "good": good,
                "second": second, "method": "sift", "kp": kp_count}


# ============================================================================
# 稠密块近邻（patch nearest-neighbour）——给"没纹理"的阶段用
# ============================================================================
DEFAULT_NN_SIZE = 128            # 规格化边长
DEFAULT_NN_PATCH = 8             # 小块边长
DEFAULT_NN_STRIDE = 4            # 小块步长
DEFAULT_NN_SEARCH = 1            # 邻域搜索半径（格），容忍几个像素的位移
DEFAULT_NN_MAX_DIST = 0.30       # 距离超过它就算"没见过这个阶段"
DEFAULT_KP_GATE = 140            # 关键点数低于它 → 走近邻；高于它 → 走 SIFT
DEFAULT_MIN_INLIERS = 12         # SIFT 分支：内点数低于它就当 unknown
DEFAULT_NN_LABELS = ("step0", "step1")
DEFAULT_NN_WEIGHT_MAG = 1.0      # 梯度块在距离里的权重（1.0 = 灰度块与梯度块同权）


class PatchNnStageMatcher:
    """把裁剪图切成小块，**每块在参考图里找最近的邻居**，距离小 = 像。

    为什么需要它：step0/step1 的装配区里只有一块光溜溜的铜盘，SIFT 只能提
    55~71 个点（还都在盘边缘和背景），两两比对只有 4 个内点，分不出来。
    但"铜盘上有没有压那根斜杆"这件事，在**每一块像素**上都写着 —— 切块后
    逐块找最近邻，杆子那块地方怎么都解释不掉，距离自然拉开。

    实验（12 张样本）：0/1 两组，本方法 6/6 全对；距离间隔也够
    （step0 探针：step0 0.148~0.230 / step1 0.190~0.243）。
    对 2/3 反而乱（铜轮一挪位就整体错位），所以它只负责 0/1 —— 见 CascadeStageMatcher。
    """

    def __init__(self, stages: dict[str, list[Path]], size: int = DEFAULT_NN_SIZE,
                 patch: int = DEFAULT_NN_PATCH, stride: int = DEFAULT_NN_STRIDE,
                 search: int = DEFAULT_NN_SEARCH, max_dist: float = DEFAULT_NN_MAX_DIST,
                 labels=DEFAULT_NN_LABELS, weight_mag: float = DEFAULT_NN_WEIGHT_MAG,
                 edges: bool = False):
        self.size = int(size)
        self.edges = bool(edges)
        self.patch = int(patch)
        self.stride = int(stride)
        self.search = int(search)
        self.max_dist = float(max_dist)
        self.labels = tuple(labels) if labels else None
        self.weight_mag = float(weight_mag)
        self.names: list[str] = []
        self.refs: list[tuple[str, np.ndarray, np.ndarray]] = []
        self.errors: list[str] = []
        self._raw: list[tuple[str, np.ndarray]] = []
        self._region = None
        self._load(stages)

    @classmethod
    def from_directory(cls, directory, **kwargs):
        return cls(collect_references(Path(directory)), **kwargs)

    # ---------------- 预处理 ----------------
    def _prepare(self, image, region):
        if image is None:
            return None
        height, width = image.shape[:2]
        if region is not None:
            x, y, w, h = (int(v) for v in region)
            x, y = max(x, 0), max(y, 0)
            w, h = min(w, width - x), min(h, height - y)
            if w < 16 or h < 16:
                return None
            patch = image[y:y + h, x:x + w]
        else:
            patch = image
        if patch.size == 0:
            return None
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY) if patch.ndim == 3 else patch
        gray = cv2.resize(gray, (self.size, self.size), interpolation=cv2.INTER_AREA)
        return sobel_edges(gray) if self.edges else gray

    def _grid(self, image):
        """切成小块，每块去均值 + 单位化（这样余弦距离=1-相关系数）。"""
        p, s = self.patch, self.stride
        h, w = image.shape
        gh = (h - p) // s + 1
        gw = (w - p) // s + 1
        rows = [image[y:y + p, x:x + p].ravel()
                for y in range(0, gh * s, s) for x in range(0, gw * s, s)]
        out = np.stack(rows).astype(np.float32)
        out -= out.mean(axis=1, keepdims=True)
        out /= (np.linalg.norm(out, axis=1, keepdims=True) + 1e-6)
        return out.reshape(gh, gw, -1), gh, gw

    def _patches(self, gray):
        """返回 (灰度块, 梯度块)，都按网格排好。"""
        f = cv2.GaussianBlur(gray.astype(np.float32), (3, 3), 0)
        gx = cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy)
        mag = np.clip(mag / (np.percentile(mag, 99) + 1e-6), 0.0, 1.0)
        f = (f - f.mean()) / (f.std() + 1e-6)
        gimg, gh, gw = self._grid(f)
        mimg, _, _ = self._grid(mag)
        return (gimg, mimg), gh, gw

    # ---------------- 参考集 ----------------
    def _load(self, stages) -> None:
        self._raw = []
        self.names = []
        for label, paths in stages.items():
            if self.labels and label not in self.labels:
                continue
            for path in paths:
                image = read_image(path)
                if image is None:
                    self.errors.append(f"读不出 {path}")
                    continue
                self._raw.append((label, image))
            if label not in self.names:
                self.names.append(label)
        self._rebuild()

    def set_region(self, region, frame_shape=None) -> None:
        region = tuple(int(v) for v in region)
        if region == self._region:
            return
        self._region = region
        self._rebuild()

    def _rebuild(self) -> None:
        self.refs = []
        for label, image in self._raw:
            gray = self._prepare(image, self._region)
            if gray is None:
                continue
            (gp, mp), gh, gw = self._patches(gray)
            self.refs.append((label, gp, mp))

    @property
    def ready(self) -> bool:
        return bool(self.refs)

    def describe(self) -> str:
        if not self.ready:
            return f"阶段识别(近邻)：未启用（{('：' + self.errors[0]) if self.errors else '无样本'}）"
        counts = ", ".join(f"{n}×{sum(1 for lab, _, _ in self.refs if lab == n)}"
                           for n in self.names)
        return (f"阶段识别(近邻)：{len(self.refs)} 个样本（{counts}）| "
                f"参考图 {self.size}px | 块 {self.patch}/{self.stride} | "
                f"搜索 ±{self.search} 格 | 距离门槛 {self.max_dist:.2f}")

    # ---------------- 距离 ----------------
    def _dist(self, a, b):
        """a/b: (gh, gw, D) 单位向量；返回每块在 ±search 格内的最近余弦距离。"""
        r = self.search
        gh, gw = a.shape[:2]
        b = np.pad(b, ((r, r), (r, r), (0, 0)), mode="edge")
        best = np.full((gh, gw), np.inf, np.float32)
        for dy in range(2 * r + 1):
            for dx in range(2 * r + 1):
                shift = b[dy:dy + gh, dx:dx + gw]
                np.minimum(best, 1.0 - (a * shift).sum(axis=2), out=best)
        return np.maximum(best, 0.0)

    def score_label(self, patches, gp, mp) -> dict[str, float]:
        """每个阶段的分数 = 该阶段所有参考图里**最小的**距离（越大越像 → 转成相似度）。"""
        best: dict[str, float] = {}
        for label, rgp, rmp in self.refs:
            d = self._dist(gp, rgp) + self.weight_mag * self._dist(mp, rmp)
            dist = float(d.mean())
            if label not in best or dist < best[label]:
                best[label] = dist
        return {label: 1.0 - dist for label, dist in best.items()}

    def match(self, frame, region=None, want_image=False):
        if not self.ready:
            return None
        if region is not None:
            self.set_region(region, frame.shape)
        gray = self._prepare(frame, self._region)
        if gray is None:
            return None
        (gp, mp), _gh, _gw = self._patches(gray)
        sims = self.score_label(None, gp, mp)
        if not sims:
            return {"label": "unknown", "raw_label": None, "score": 0.0, "angle": 0.0,
                    "region": self._region, "method": "nn"}
        ranked = sorted(sims.items(), key=lambda kv: -kv[1])
        label, score = ranked[0]
        second = ranked[1][1] if len(ranked) > 1 else 0.0
        best_dist = 1.0 - score
        if best_dist > self.max_dist:
            label = "unknown"
        return {"label": label, "raw_label": ranked[0][0], "score": score, "angle": 0.0,
                "region": self._region, "method": "nn", "dist": best_dist,
                "second": second}


class CascadeStageMatcher:
    """先数关键点，再决定用哪把尺子：

        kp <  kp_gate  → 近邻（光面阶段，靠"每块像素找邻居"）
        kp >= kp_gate  → SIFT（有齿/有孔的阶段，靠特征点 + 单应性内点）

    为什么这么分：12 张样本里 step0/1 的关键点 55~71，step2/3 是 193~248，
    中间空得很干净，不会骑墙。两条腿各自补对方的短板，互不干扰。
    """

    def __init__(self, stages: dict[str, list[Path]], kp_gate: int = DEFAULT_KP_GATE,
                 min_inliers: int = DEFAULT_MIN_INLIERS, nn_labels=DEFAULT_NN_LABELS,
                 ref_size: int = 256, nfeatures: int = 500, ratio: float = 0.75,
                 ransac_px: float = 5.0, min_score: float = 0.02,
                 margin: float = 0.10, nn_size: int = DEFAULT_NN_SIZE,
                 nn_max_dist: float = DEFAULT_NN_MAX_DIST, nn_kwargs=None,
                 edges: bool = False):
        self.kp_gate = int(kp_gate)
        self.min_inliers = int(min_inliers)
        self.edges = bool(edges)
        self.sift = SiftStageMatcher(stages, ref_size=ref_size, nfeatures=nfeatures,
                                     ratio=ratio, ransac_px=ransac_px,
                                     min_score=min_score, margin=margin,
                                     edges=edges)
        self.nn_kwargs = dict(nn_kwargs or {})
        self.nn_kwargs.setdefault("size", nn_size)
        self.nn_kwargs.setdefault("max_dist", nn_max_dist)
        self.nn_kwargs.setdefault("edges", edges)
        self.nn = PatchNnStageMatcher(stages, labels=nn_labels, **self.nn_kwargs)

    @classmethod
    def from_directory(cls, directory, **kwargs):
        return cls(collect_references(Path(directory)), **kwargs)

    @property
    def ready(self) -> bool:
        return self.sift.ready or self.nn.ready

    @property
    def names(self):
        return self.sift.names

    def describe(self) -> str:
        return (f"{self.sift.describe()} | 关键点门槛 {self.kp_gate}："
                f"低于它走近邻，高于它走 SIFT（内点≥{self.min_inliers} 才算数）\n"
                f"         {self.nn.describe()}")

    def set_region(self, region, frame_shape=None):
        self.sift.set_region(region, frame_shape)
        self.nn.set_region(region, frame_shape)

    def match(self, frame, region=None, want_image=False):
        if region is not None:
            self.set_region(region, frame.shape)
        gray = self.sift._prepare(frame, self.sift._region)
        if gray is None:
            return None
        kp, _des = self.sift.sift.detectAndCompute(gray, None)
        count = len(kp) if kp is not None else 0

        if count < self.kp_gate and self.nn.ready:
            result = self.nn.match(frame, None)
            if result is not None:
                result["kp"] = count
                result["method"] = "nn"
                return result

        result = self.sift.match(frame, None)
        if result is None:
            return None
        result["kp"] = count
        result["method"] = "sift"
        if result.get("label") != "unknown" and result.get("inliers", 0) < self.min_inliers:
            result["label"] = "unknown"
            result["note"] = f"内点 {result.get('inliers', 0)} < {self.min_inliers}"
        return result
