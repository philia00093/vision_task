"""卡尔曼滤波跟踪（只跟踪中心点）。

踩过的坑：把宽高也放进状态会让框每帧线性长大/缩小，表现就是"预测全部跑飞"；所以尺寸不参与滤波，只做轻量平滑。"""

from __future__ import annotations

import numpy as np
import cv2

from edge.constants import DEFAULT_KF_ACC, DEFAULT_KF_DT_MAX, DEFAULT_KF_LOST_DAMP, DEFAULT_KF_MEAS, DEFAULT_KF_SIZE_ALPHA, DEFAULT_KF_VMAX


class KalmanTrack:
    """匀速卡尔曼滤波跟踪**中心点**：状态 [x, y, vx, vy]，尺寸不参与滤波。

    这里照参考实现的做法，并且是针对"预测跑飞"踩过的坑定的：

    1) **宽高不进状态**。之前把 [cx,cy,w,h,vx,vy,vw,vh] 全塞进卡尔曼，宽高带上速度后
       会每帧线性长大/缩小 —— 表现就是"预测全部跑飞"。零件框尺寸抖动很小，
       直接用最近检出的宽高做一次轻量平滑就够（size_alpha），滤波器只专注"往哪走"。
    2) **不用 cv2.KalmanFilter**。它的 errorCovPre 默认全 0，predict 之后协方差只剩
       processNoiseCov（很小）→ 增益趋近 0 → 滤波器几乎不信检测、只信自己的外推，
       速度稍微估歪就飞出去。这里手写标准方程，初始 P 给到位，增益正常。
    3) 三道防飞保险（丢检时尤其重要）：
       · dt 上限：卡顿一下不会按大 dt 瞬移；
       · 速度上限 + 中心点夹在画面内；
       · 丢检后速度按比例衰减 → 慢慢滑停，而不是沿直线飞出画面。
    """

    def __init__(self, track_id, box, cls_id, name, conf, age_max, bounds,
                 std_acc=DEFAULT_KF_ACC, meas_std=DEFAULT_KF_MEAS,
                 v_max=DEFAULT_KF_VMAX, dt_max=DEFAULT_KF_DT_MAX,
                 lost_damp=DEFAULT_KF_LOST_DAMP, size_alpha=DEFAULT_KF_SIZE_ALPHA):
        self.track_id = int(track_id)
        self.cls_id = int(cls_id)
        self.name = name
        self.conf = float(conf)
        self.miss = 0
        self.age_max = int(age_max)
        self.std_acc = float(std_acc)
        self.meas_std = float(meas_std)
        self.v_max = float(v_max)
        self.dt_max = float(dt_max)
        self.lost_damp = float(lost_damp)
        self.size_alpha = float(size_alpha)
        self.width, self.height = bounds
        x1, y1, x2, y2 = box
        # 状态 [x, y, vx, vy]（速度单位 px/秒）；P 初始给足，避免"第一帧不信检测"
        self.x = np.array([[(x1 + x2) / 2.0], [(y1 + y2) / 2.0], [0.0], [0.0]], float)
        self.P = np.diag([100.0, 100.0, 1e4, 1e4])
        self.w = max(x2 - x1, 1.0)
        self.h = max(y2 - y1, 1.0)
        self.last_time = None

    # ------------------------------------------------------------ 内部
    def _matrices(self, dt):
        """状态转移 F 与过程噪声 Q（按实际 dt 算，帧率飘了也不怕）。"""
        transition = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]],
                              float)
        noise = np.array([[dt ** 4 / 4, 0, dt ** 3 / 2, 0],
                          [0, dt ** 4 / 4, 0, dt ** 3 / 2],
                          [dt ** 3 / 2, 0, dt ** 2, 0],
                          [0, dt ** 3 / 2, 0, dt ** 2]], float) * self.std_acc ** 2
        return transition, noise

    def _advance(self, now, damp_velocity=False) -> None:
        """把状态推进到 now（dt<=0 就什么都不做，避免重复积分）。"""
        if self.last_time is None:
            self.last_time = now
            return
        dt = min(max(now - self.last_time, 0.0), self.dt_max)
        if dt <= 0:
            return
        self.last_time = now
        transition, noise = self._matrices(dt)
        self.x = transition @ self.x
        self.P = transition @ self.P @ transition.T + noise
        if damp_velocity:                      # 丢检：速度衰减 → 滑停而不是飞出去
            self.x[2] *= self.lost_damp
            self.x[3] *= self.lost_damp
        self._clamp()

    def _clamp(self) -> None:
        """速度上限 + 中心点夹在画面内（留 20% 余量），防止外推把框甩飞。"""
        speed = float((self.x[2, 0] ** 2 + self.x[3, 0] ** 2) ** 0.5)
        if speed > self.v_max > 0:
            scale = self.v_max / speed
            self.x[2, 0] *= scale
            self.x[3, 0] *= scale
        margin_x, margin_y = self.width * 0.2, self.height * 0.2
        self.x[0, 0] = min(max(float(self.x[0, 0]), -margin_x), self.width + margin_x)
        self.x[1, 0] = min(max(float(self.x[1, 0]), -margin_y), self.height + margin_y)

    # ------------------------------------------------------------ 对外
    @property
    def center(self) -> tuple:
        return float(self.x[0, 0]), float(self.x[1, 0])

    @property
    def box(self) -> tuple:
        cx, cy = self.center
        return (cx - self.w / 2, cy - self.h / 2, cx + self.w / 2, cy + self.h / 2)

    def predict(self, now, damp_velocity=False) -> tuple:
        """推进到 now 并返回预测中心（丢检时传 damp_velocity=True）。"""
        self._advance(now, damp_velocity=damp_velocity)
        return self.center

    def update(self, box, cls_id, name, conf, now) -> tuple:
        """用一次检测校正：先推进到 now，再做标准卡尔曼校正。"""
        self._advance(now)
        measure = np.array([[(box[0] + box[2]) / 2.0], [(box[1] + box[3]) / 2.0]], float)
        observation = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], float)
        noise = np.eye(2) * self.meas_std ** 2
        gain = self.P @ observation.T @ np.linalg.inv(
            observation @ self.P @ observation.T + noise)
        self.x = self.x + gain @ (measure - observation @ self.x)
        identity = np.eye(4)
        self.P = (identity - gain @ observation) @ self.P
        self.miss = 0
        self.cls_id, self.name, self.conf = int(cls_id), name, float(conf)
        # 尺寸只做轻量平滑：不参与滤波，就不会"长大/缩小跑飞"
        self.w = (1 - self.size_alpha) * self.w + self.size_alpha * max(box[2] - box[0], 1.0)
        self.h = (1 - self.size_alpha) * self.h + self.size_alpha * max(box[3] - box[1], 1.0)
        self._clamp()
        return self.center

    @property
    def alive(self) -> bool:
        return self.miss <= self.age_max

def box_iou(a, b) -> float:
    """两个框的 IoU：用于轨迹关联/去重（保证一个零件只留一个框）。"""
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0
