"""Локальный OCR через Apple Vision (macOS).

Выбран по замеру на полевых фото: кириллицу этикеток читает почти без
ошибок за 0.3–0.8 c на CPU, тогда как RapidOCR из коробки её не знает.
Работает офлайн — поэтому он же запасной канал для облачного OCR.
"""

from __future__ import annotations

import sys

from PIL import Image

from . import TextLine


class AppleVisionExtractor:
    name = "apple-vision"

    def __init__(self, max_side: int = 1600) -> None:
        from ocrmac import ocrmac

        self._ocr = ocrmac
        self.max_side = max_side

    @staticmethod
    def available() -> bool:
        if sys.platform != "darwin":
            return False
        try:
            import ocrmac  # noqa: F401
        except ImportError:
            return False
        return True

    def extract(self, image: Image.Image) -> list[TextLine]:
        image = image.convert("RGB")
        if max(image.size) > self.max_side:
            image = image.copy()
            image.thumbnail((self.max_side, self.max_side))
        results = self._ocr.OCR(
            image, language_preference=["ru-RU", "en-US"], recognition_level="accurate"
        ).recognize()
        # bbox — (x, y, w, h) в долях кадра
        return [
            TextLine(text=text, confidence=float(conf), height=float(box[3]))
            for text, conf, box in results
            if text.strip()
        ]
