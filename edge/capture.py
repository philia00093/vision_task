"""数据采集按钮：点一下/按数字键就把画面存进训练图目录。

默认存**干净帧**（没有任何叠加），免得把误识别的框烧进训练数据。"""

from __future__ import annotations

import cv2
import re
from pathlib import Path

from edge.constants import BUTTON_GAP, BUTTON_HEIGHT, BUTTON_MARGIN, BUTTON_RIGHT_RESERVE, CAPTURE_SPECS, LABEL_FONT_SIZE
from edge.drawing import _measure_label, draw_text_box, load_cjk_font
from edge.imaging import save_image


class CaptureDeck:
    """图像采集按钮：点一下 / 按 1~5，就把当前帧存进训练用 images 目录。

    编号规则：扫描目标目录里已有的 <前缀>_<数字>，从"最大编号 + 1"续起，
    所以接着已有的 compass_1..compass_30 往下存是 compass_31；
    中间的缺号（如 compass_14）不会被重新占用，避免和历史数据混淆。
    """

    def __init__(self, root, font=None, enabled=True, prefix_base="vision_yolo_train"):
        self.root = Path(root)
        self.prefix_base = prefix_base
        self.font = font
        self.enabled = enabled
        self.buttons = []          # [(name, prefix, dir, hotkey, rect)]
        self.hit_boxes = []        # [(rect, index)]
        self._next = {}            # (目录, 前缀) -> 下一个可用编号
        self._flash = {}           # index -> 剩余高亮帧数
        self.pending = None        # 由鼠标回调置位，主循环消费
        self.last_saved = None
        self.counts = {}           # 每个前缀本次已存数量

    # ------------------------------------------------------------ 编号
    @staticmethod
    def scan_highest(directory: Path, prefix: str) -> int:
        """扫描目录里已有的 <前缀>_<数字>，返回最大编号（没有则 0）。"""
        highest = 0
        if directory.is_dir():
            pattern = re.compile(rf"^{re.escape(prefix)}_(\d+)$", re.IGNORECASE)
            for item in directory.iterdir():
                if item.is_file():
                    match = pattern.match(item.stem)
                    if match:
                        highest = max(highest, int(match.group(1)))
        return highest

    def start_numbers(self) -> dict[str, int]:
        """本次会话每个前缀的起始编号（= 已有最大编号 + 1）。不产生副作用。"""
        starts = {}
        for _name, prefix, directory, _hotkey, _rect in self.buttons:
            starts[prefix] = self.scan_highest(directory, prefix) + 1
        return starts

    def next_index(self, directory: Path, prefix: str) -> int:
        """返回下一个可用编号：已有最大编号 + 1（不填缺号）。"""
        key = (str(directory), prefix)
        if key not in self._next:
            self._next[key] = self.scan_highest(directory, prefix)
        self._next[key] += 1
        return self._next[key]

    # ------------------------------------------------------------ 保存
    def capture(self, frame, index: int) -> Path | None:
        """把 frame 存成 <前缀>_<编号>.png，返回保存路径。"""
        if not self.enabled or frame is None or not (0 <= index < len(self.buttons)):
            return None
        name, prefix, directory, _hotkey, _rect = self.buttons[index]
        directory.mkdir(parents=True, exist_ok=True)
        number = self.next_index(directory, prefix)
        out = directory / f"{prefix}_{number}.png"
        while out.exists():                       # 极端情况下的兜底，绝不覆盖已有数据
            number = self.next_index(directory, prefix)
            out = directory / f"{prefix}_{number}.png"
        save_image(out, frame)
        self._flash[index] = 12                   # 高亮约 12 帧
        self.counts[prefix] = self.counts.get(prefix, 0) + 1
        self.last_saved = (name, out)
        return out

    # ------------------------------------------------------------ 布局与绘制
    def layout(self, image):
        """按画面尺寸算按钮位置（底部横排）。"""
        self.buttons = []
        self.hit_boxes = []
        height, width = image.shape[:2]
        count = len(CAPTURE_SPECS)
        total_gap = BUTTON_GAP * (count - 1)
        available = width - BUTTON_MARGIN * 2 - BUTTON_RIGHT_RESERVE
        button_w = max(78, min(150, (available - total_gap) // count))
        y = height - BUTTON_MARGIN - BUTTON_HEIGHT
        x = BUTTON_MARGIN
        for i, (name, prefix, rel, hotkey) in enumerate(CAPTURE_SPECS):
            rect = (x, y, x + button_w, y + BUTTON_HEIGHT)
            self.buttons.append((name, prefix, self.root / rel, hotkey, rect))
            self.hit_boxes.append((rect, i))
            x += button_w + BUTTON_GAP
        # 文字必须放进按钮里：按最宽标签自动选字号，否则会溢出压到隔壁按钮
        self.text_font = self._fit_font(button_w - 10)
        return self.buttons

    def _labels(self) -> list[str]:
        """每个按钮可能出现的最长文字（含"已存"闪烁态）。"""
        labels = []
        for _name, _p, _d, hotkey, _r in self.buttons:
            labels.append(f"{hotkey} {_name}")
            labels.append(f"{hotkey} 已存")
        return labels

    def _fit_font(self, max_width: int):
        """挑一个能让所有标签都塞进按钮宽度的字号。"""
        if self.font is None:
            return None
        base_size = getattr(self.font, "size", LABEL_FONT_SIZE)
        for size in range(base_size, 9, -1):
            font = load_cjk_font(size)
            if font is None:
                return None
            widest = max(_measure_label(text, font)[0] for text in self._labels())
            if widest <= max_width:
                return font
        return load_cjk_font(10)

    def draw(self, image):
        """把按钮画到画面底部；被采集的按钮短暂高亮并显示"已存"。"""
        if not self.enabled or not self.buttons:
            return image
        overlay = image.copy()
        for i, (_name, _prefix, _dir, _hotkey, (x1, y1, x2, y2)) in enumerate(self.buttons):
            flashing = self._flash.get(i, 0) > 0
            color = (0, 190, 0) if flashing else (45, 45, 45)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (230, 230, 230), 1)
        cv2.addWeighted(overlay, 0.55, image, 0.45, 0, image)      # 半透明，不挡画面

        font = self.text_font if getattr(self, "text_font", None) is not None else None
        for i, (name, _prefix, _dir, hotkey, (x1, y1, x2, y2)) in enumerate(self.buttons):
            flashing = self._flash.get(i, 0) > 0
            label = f"{hotkey} 已存" if flashing else f"{hotkey} {name}"
            color = (0, 255, 0) if flashing else (215, 215, 215)
            if font is not None:
                text_w, text_h = _measure_label(label, font)
                # 水平、垂直都居中，并保证不越出按钮
                tx = x1 + max(2, (x2 - x1 - text_w) // 2)
                ty = y1 + max(2, (y2 - y1 - text_h) // 2)
                draw_text_box(image, label, tx, ty, color, font=font)
            else:
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                cv2.putText(image, label,
                            (x1 + max(4, (x2 - x1 - tw) // 2),
                             y1 + (y2 - y1 + th) // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        # 递减高亮计数
        for key in list(self._flash):
            self._flash[key] -= 1
            if self._flash[key] <= 0:
                self._flash.pop(key, None)
        return image

    def handle_click(self, x: int, y: int) -> bool:
        """鼠标点击：命中按钮则置位待采集，返回是否命中。"""
        for (x1, y1, x2, y2), index in self.hit_boxes:
            if x1 <= x <= x2 and y1 <= y <= y2:
                self.pending = index
                return True
        return False

    def hotkey_index(self, key: int) -> int | None:
        """键盘按键 → 按钮下标（1~5）。"""
        for i, (_n, _p, _d, hotkey, _r) in enumerate(self.buttons):
            if key == ord(hotkey):
                return i
        return None

    def summary(self) -> str:
        if not self.counts:
            return "本次未采集"
        return "本次采集：" + "、".join(f"{k}×{v}" for k, v in sorted(self.counts.items()))

def on_mouse(event, x, y, flags, deck):
    """鼠标回调：左键点击命中按钮就记录一次待采集。"""
    if event == cv2.EVENT_LBUTTONDOWN and deck is not None:
        deck.handle_click(x, y)
