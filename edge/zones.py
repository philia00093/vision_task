"""料盘区域的**图像算法**：暗色/绿色掩膜、料盘形状判据、粘连切分。

关键坑：两个黑色料盘挨着时，暗色掩膜会把它们连成一块，所以必须先找"缝"切开，再按料盘形状挑出左右两个。"""

from __future__ import annotations

import numpy as np
import cv2

from edge.constants import DEFAULT_DARK_V_MAX, DEFAULT_GREEN_HI, DEFAULT_GREEN_LO, DEFAULT_TRAY_MAX_SHARE, DEFAULT_TRAY_MIN_FILL, DEFAULT_TRAY_MIN_SHARE, DEFAULT_TRAY_SEAM_DROP, DEFAULT_TRAY_SHAPE_LIMIT, DEFAULT_TRAY_SPLIT_RATIO, DEFAULT_ZONE_SCAN_TOP, GREEN_MIN_AREA_RATIO, GREEN_MIN_FILL_RATIO
from edge.imaging import contours_to_boxes


def detect_green_regions(frame, lo, hi, min_area_ratio=GREEN_MIN_AREA_RATIO,
                         min_fill_ratio=GREEN_MIN_FILL_RATIO, morph=7, merge_gap=0,
                         mask=None):
    """HSV 分割绿色料盘，按 x 排序返回 [(x1, y1, x2, y2, area), ...]。

    这是 --zone-mode green 的老路径（料盘还是绿色时用）；
    默认走 dark_mask（料盘黑色）。左边第一块=装配区，第二块=物料区。
    mask 参数可以复用外部已经算好的掩码，避免重复做 HSV 转换。
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

def dark_mask(frame, v_max=DEFAULT_DARK_V_MAX, morph=7, scan_top=DEFAULT_ZONE_SCAN_TOP):
    """黑色料盘的二值掩码（0/255）——料盘改成黑色后用它替代绿色分割。

    思路：木桌亮、料盘暗，所以在 HSV 里只取"明度 V 足够低"的像素。
    两个坑：
      · 画面顶部那片黑色机器臂同样是暗的 → 用 scan_top 把上面一刀切掉；
      · 料盘上的反光/高光会把暗块打洞 → 闭运算补洞。
    HSV 而不是直接灰度，是因为灰度对黄木桌的亮度差更敏感，V 通道更稳。
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv[:, :, 2], 0, int(v_max))
    if scan_top and scan_top > 0:
        cut = int(round(frame.shape[0] * float(scan_top)))
        if cut > 0:
            mask[:cut, :] = 0                      # 上半屏（机器臂）不参与候选
    if morph > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph, morph))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    return mask

def zone_mask(frame, mode="dark", green_lo=DEFAULT_GREEN_LO, green_hi=DEFAULT_GREEN_HI,
              v_max=DEFAULT_DARK_V_MAX, scan_top=DEFAULT_ZONE_SCAN_TOP):
    """按 --zone-mode 选掩码：dark=黑色料盘（默认），green=老版绿色料盘。"""
    if mode == "green":
        return green_mask(frame, green_lo, green_hi)
    return dark_mask(frame, v_max=v_max, scan_top=scan_top)

def tray_like(box, frame_area, min_share=DEFAULT_TRAY_MIN_SHARE,
              max_share=DEFAULT_TRAY_MAX_SHARE, shape_limit=DEFAULT_TRAY_SHAPE_LIMIT) -> bool:
    """这个块像不像一个料盘：面积占比合适 + 不细长。"""
    w, h = max(box[2] - box[0], 1), max(box[3] - box[1], 1)
    ratio = max(w, h) / float(min(w, h))
    share = box[4] / max(float(frame_area), 1.0)
    return min_share <= share <= max_share and ratio <= shape_limit

def tray_pair_ok(left, right, frame_area, max_area_ratio=2.5) -> bool:
    """两个候选框像不像"并排放在桌上的两个料盘"。

    自动识别偶尔会把阴影/机器臂当成料盘，锁错了后面就一直判不出遮挡。
    这里加一道闸：两块大小别差太多、左右大致分开、竖直方向基本并排。
    不过闸就不锁定（继续自动识别），宁可显示"等待区域锁定"也不要锁错。

    比"大小"用的是**外接框面积**而不是轮廓面积：左盘中间有个大圆凹槽，
    它的暗色轮廓面积天然只有右盘的三分之一，用轮廓面积比会误杀（实测 3.3），
    而外接框面积比只有 1.0~1.8，才是"两个大小差不多的盘"的真实反映。
    """
    if left is None or right is None:
        return False
    if not (tray_like(left, frame_area) and tray_like(right, frame_area)):
        return False
    area_left = max((left[2] - left[0]) * (left[3] - left[1]), 1)
    area_right = max((right[2] - right[0]) * (right[3] - right[1]), 1)
    if max(area_left, area_right) / min(area_left, area_right) > max_area_ratio:
        return False
    if left[0] > right[0]:                       # 保证 left 在左边
        left, right = right, left
    right_w = right[2] - right[0]
    if left[2] > right[0] + right_w * 0.30:      # 左右重叠太多 → 像同一块被切开
        return False
    overlap_y = min(left[3], right[3]) - max(left[1], right[1])
    if overlap_y < min(left[3] - left[1], right[3] - right[1]) * 0.40:
        return False
    return True

def split_touching_trays(mask, boxes, split_ratio=DEFAULT_TRAY_SPLIT_RATIO,
                         seam_drop=DEFAULT_TRAY_SEAM_DROP,
                         min_area_ratio=DEFAULT_TRAY_MIN_SHARE,
                         min_fill_ratio=DEFAULT_TRAY_MIN_FILL):
    """两个料盘挨着时掩码连成一块，这里从"缝"处把它切成两块。

    缝的判据：最大块的中间带里**暗像素最少的一列**——两盘之间是阴影窄缝，
    没有料盘本体那么暗，所以那一列的暗像素数会明显低于两侧。
    只有该列确实明显更空（低于中间带中位数的 seam_drop 倍）才切，
    避免只有一块料盘时被硬切成两半。返回切过的框列表（没切就原样返回）。
    """
    if not boxes:
        return boxes
    big = max(boxes, key=lambda b: b[4])
    x1, y1, x2, y2 = big[0], big[1], big[2], big[3]
    box_w, box_h = x2 - x1, y2 - y1
    if box_w < box_h * split_ratio:                  # 不够宽：不是两个并排的盘
        return boxes
    low, high = int(box_w * 0.28), int(box_w * 0.72)  # 只在中段找缝
    if high - low < 6:
        return boxes
    profile = (mask[y1:y2, x1:x2] > 0).sum(axis=0).astype(np.float32)
    index = int(np.argmin(profile[low:high]))
    if profile[low + index] > float(np.median(profile[low:high])) * seam_drop:
        return boxes
    split = low + index + x1
    height, width = mask.shape[:2]
    left, right = mask.copy(), mask.copy()
    left[:, split:] = 0
    right[:, :split] = 0
    out = [b for b in boxes if b is not big]
    out += contours_to_boxes(left, width, height, min_area_ratio, min_fill_ratio)
    out += contours_to_boxes(right, width, height, min_area_ratio, min_fill_ratio)
    return out

def pick_tray_boxes(boxes, frame_area,
                    min_share=DEFAULT_TRAY_MIN_SHARE) -> list:
    """从候选块里挑出左右两个料盘：先按料盘特征过滤，再取面积最大的两块，按 x 排序。"""
    good = [b for b in boxes if tray_like(b, frame_area)]
    if len(good) < 2:
        return []
    good.sort(key=lambda b: -b[4])
    return sorted(good[:2], key=lambda b: b[0])
