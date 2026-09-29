"""Кроп этикетки на фото запроса.

YOLOv8n, дообученный на разметке рамок этикеток по полевым фото кейса
(веса artifacts/models/label_yolov8n.pt). На полке в кадр попадают соседние
бутылки — кейсодержатель прямо сказал искать центральную, поэтому из всех
найденных рамок берётся лучшая по уверенности с поправкой на близость к
центру кадра.

Применяется только к запросам: эталоны каталога — студийные фото одной
бутылки, их кадрировать незачем.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

log = logging.getLogger(__name__)


@dataclass(slots=True)
class Box:
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float

    def expand(self, margin: float, width: int, height: int) -> Box:
        dx = int((self.x2 - self.x1) * margin)
        dy = int((self.y2 - self.y1) * margin)
        return Box(max(0, self.x1 - dx), max(0, self.y1 - dy), min(width, self.x2 + dx),
                   min(height, self.y2 + dy), self.confidence)


class LabelDetector:
    def __init__(self, weights: Path, confidence: float = 0.25, device: str = "auto",
                 center_weight: float = 0.3) -> None:
        # ultralytics подменяет PIL.Image.open и на первом битом файле ставит
        # pi-heif через pip прямо внутри запроса. HEIC и так открывает pillow-heif.
        os.environ.setdefault("YOLO_AUTOINSTALL", "False")
        from ultralytics import YOLO

        from ..features.embedder import pick_device

        self.model = YOLO(str(weights))
        self.confidence = confidence
        # На MPS ultralytics местами нестабилен, а детектор и так ~30 мс на CPU.
        device = pick_device(device)
        self.device = 0 if device == "cuda" else "cpu"
        self.center_weight = center_weight
        # Прогрев: первый вызов YOLO инициализирует граф — не на первом запросе.
        self.detect(Image.new("RGB", (320, 320)))

    def detect(self, image: Image.Image) -> Box | None:
        array = np.asarray(image.convert("RGB"))[:, :, ::-1]  # ultralytics ждёт BGR
        result = self.model.predict(source=array, conf=self.confidence, iou=0.45,
                                    verbose=False, device=self.device)
        if not result or result[0].boxes is None or len(result[0].boxes) == 0:
            return None
        width, height = image.size
        best, best_score = None, -1.0
        for xyxy, conf in zip(result[0].boxes.xyxy.tolist(), result[0].boxes.conf.tolist(),
                              strict=True):
            x1, y1, x2, y2 = (int(v) for v in xyxy)
            cx, cy = (x1 + x2) / 2 / width, (y1 + y2) / 2 / height
            off_center = ((cx - 0.5) ** 2 + (cy - 0.5) ** 2) ** 0.5  # 0..0.71
            score = float(conf) - self.center_weight * off_center
            if score > best_score:
                best, best_score = Box(x1, y1, x2, y2, float(conf)), score
        return best


def backdrop_box(image: Image.Image, tolerance: int = 30, min_uniform: float = 0.6,
                 min_fill: float = 0.03, min_area: float = 0.1) -> Box | None:
    """Рамка предмета на однотонном фоне (пэкшот, скриншот карточки) или None.

    Фон считается однотонным, если >= min_uniform пикселей узкой рамки кадра
    отличаются от их медианного цвета меньше чем на tolerance по каждому каналу.
    Тогда предмет — строки и столбцы, где отличающихся от фона пикселей больше
    min_fill. На фото с полки рамка кадра пёстрая (соседние бутылки, ценники),
    и функция возвращает None — дальше работает детектор этикетки.
    """
    array = np.asarray(image.convert("RGB")).astype(np.int16)
    height, width = array.shape[:2]
    band = max(2, min(height, width) // 100)
    border = np.concatenate([
        array[:band].reshape(-1, 3), array[-band:].reshape(-1, 3),
        array[:, :band].reshape(-1, 3), array[:, -band:].reshape(-1, 3),
    ])
    background = np.median(border, axis=0)
    if (np.abs(border - background).max(axis=1) < tolerance).mean() < min_uniform:
        return None
    mask = np.abs(array - background).max(axis=2) >= tolerance
    rows = np.flatnonzero(mask.mean(axis=1) > min_fill)
    cols = np.flatnonzero(mask.mean(axis=0) > min_fill)
    if rows.size == 0 or cols.size == 0:
        return None
    x1, y1, x2, y2 = int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1
    area = (x2 - x1) * (y2 - y1) / (width * height)
    if area < min_area or area > 0.999:
        return None
    return Box(x1, y1, x2, y2, 1.0)


def crop(image: Image.Image, box: Box | None, margin: float) -> Image.Image:
    if box is None:
        return image
    box = box.expand(margin, *image.size)
    return image.crop((box.x1, box.y1, box.x2, box.y2))


def gray_world(image: Image.Image) -> Image.Image:
    """Баланс белого «серый мир»: магазинные лампы против студийного света."""
    array = np.asarray(image.convert("RGB")).astype(np.float32)
    means = array.reshape(-1, 3).mean(axis=0)
    gains = np.clip(means.mean() / np.maximum(means, 1e-6), 0.5, 2.0)
    return Image.fromarray(np.clip(array * gains, 0, 255).astype(np.uint8))
