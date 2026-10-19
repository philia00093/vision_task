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
from pathlib import Path

import cv2
import numpy as np

# 参考图默认目录（相对项目根）
STAGE_DEFAULT_DIR = "vision_yolo_train/images/steps"
DEFAULT_STAGE_SIZE = 96          # 规格化后的边长（96 够用；调小更快、更糙）
DEFAULT_ANGLE_STEP = 1.0         # 旋转遍历步长（度）
DEFAULT_ANGLE_RANGE = (-180.0, 179.0)
DEFAULT_REFINE_STEP = 0.2        # 命中角度附近的细搜步长（度）
DEFAULT_STAGE_INTERVAL = 10      # 每多少帧做一次阶段识别
DEFAULT_STAGE_MARGIN = 0.12      # 裁剪区域外扩比例（把料盘边缘也框进去）
MIN_STAGE_SCORE = 0.30           # 低于它就当"没认出来"，显示 unknown

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def read_image(path: Path):
    """读图（兼容中文路径）。"""
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None


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
                stages[child.name] = samples
        elif child.suffix.lower() in IMAGE_SUFFIXES:
            stages.setdefault(child.stem, []).append(child)
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
                 rotate=True, min_score=MIN_STAGE_SCORE):
        self.size = (int(size), int(size))
        self.angle_step = max(float(angle_step), 0.05)
        self.angle_range = (float(angle_range[0]), float(angle_range[1]))
        self.refine_step = max(float(refine_step), 0.01)
        self.margin = max(float(margin), 0.0)
        self.rotate = bool(rotate)
        self.min_score = float(min_score)
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
        """按区域裁剪 → 灰度 → 统一尺寸。返回 (size, size) uint8；非法区域返回 None。"""
        if image is None:
            return None
        height, width = image.shape[:2]
        if region is not None:
            x, y, w, h = region
            x, y = max(0, x), max(0, y)
            w, h = min(w, width - x), min(h, height - y)
            if w < 4 or h < 4:
                return None
            patch = image[y:y + h, x:x + w]
        else:
            patch = image
        if patch.size == 0:
            return None
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY) if patch.ndim == 3 else patch
        gray = cv2.resize(gray, self.size, interpolation=cv2.INTER_AREA)
        # 直方图均衡一点，抵消现场光照差异（这一步对 4 张参考图也是同样处理）
        return cv2.equalizeHist(gray)

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
