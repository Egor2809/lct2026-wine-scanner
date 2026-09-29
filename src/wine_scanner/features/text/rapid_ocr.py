"""Локальный кроссплатформенный OCR: RapidOCR (PP-OCR на ONNX Runtime, CPU).

Работает на Linux/macOS/Windows без сети и без платных API — это текстовый
канал режима «в контуре». Распознавание — модель PP-OCRv5 `eslav`
(кириллица + латиница), детектор — штатный из пакета.

Кроме кадра целиком читаются его верхняя и нижняя части, увеличенные в 1,6
раза: мелкий шрифт (сорт, «полусладкое», название над этикеткой) детектор на
целом кадре теряет. Проходы идут параллельно, у каждого потока свой движок.
Замер 29.09 на полевых фото: 59 -> 61/63, macro F1 0,891 -> 0,945.
"""

from __future__ import annotations

import re
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image

from . import TextLine

_TILES = ((0.0, 0.6), (0.4, 1.0))
_TILE_SCALE = 1.6
_DEDUP = re.compile(r"\W+")


class RapidOCRExtractor:
    name = "rapidocr"

    def __init__(self, max_side: int = 1600, lang: str = "eslav", threads: int = 2,
                 tiles: bool = True) -> None:
        self.max_side = max_side
        self.lang = lang
        self.threads = threads
        self.tiles = tiles
        self._local = threading.local()
        self._pool = ThreadPoolExecutor(1 + len(_TILES) if tiles else 1,
                                        thread_name_prefix="rapidocr")
        # Первый движок — сразу: модели грузятся при старте приложения, а не на запросе.
        self._pool.submit(self._engine).result()

    @staticmethod
    def available() -> bool:
        try:
            import rapidocr  # noqa: F401
        except ImportError:
            return False
        return True

    def _engine(self):
        """Движок RapidOCR текущего потока (сессии ONNX не делим между потоками)."""
        engine = getattr(self._local, "engine", None)
        if engine is None:
            from rapidocr import RapidOCR
            from rapidocr.utils.typings import LangRec, ModelType, OCRVersion

            engine = RapidOCR(params={
                "Rec.lang_type": LangRec(self.lang),
                "Rec.ocr_version": OCRVersion.PPOCRV5,
                "Rec.model_type": ModelType.MOBILE,
                "EngineConfig.onnxruntime.intra_op_num_threads": self.threads,
                "Global.log_level": "error",
            })
            self._local.engine = engine
        return engine

    def _read(self, image: Image.Image, frame_height: float) -> list[TextLine]:
        result = self._engine()(np.asarray(image))
        if result.txts is None:
            return []
        boxes = np.asarray(result.boxes)
        return [
            TextLine(text=text, confidence=float(score),
                     height=float((box[:, 1].max() - box[:, 1].min()) / frame_height))
            for text, score, box in zip(result.txts, result.scores, boxes, strict=True)
            if text.strip()
        ]

    def extract(self, image: Image.Image) -> list[TextLine]:
        image = image.convert("RGB")
        if max(image.size) > self.max_side:
            image = image.copy()
            image.thumbnail((self.max_side, self.max_side))
        width, height = image.size
        jobs = [self._pool.submit(self._read, image, height)]
        if self.tiles:
            for top, bottom in _TILES:
                tile = image.crop((0, int(top * height), width, int(bottom * height)))
                tile = tile.resize((int(tile.width * _TILE_SCALE),
                                    int(tile.height * _TILE_SCALE)), Image.BICUBIC)
                jobs.append(self._pool.submit(self._read, tile, height * _TILE_SCALE))
        seen: set[str] = set()
        lines = []
        for job in jobs:  # порядок: кадр целиком, затем части; дубли строки — один раз
            for line in job.result():
                key = _DEDUP.sub("", line.text.lower())
                if key and key not in seen:
                    seen.add(key)
                    lines.append(line)
        return lines
