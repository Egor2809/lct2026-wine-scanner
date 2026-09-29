"""Нормализация фото запроса: одно фото -> виды для каждого канала.

Визуальному эмбеддингу нужна этикетка крупно и с выровненным цветом, OCR —
та же этикетка с запасом по краям и без цветокоррекции (баланс белого
может «съесть» тонкий цветной шрифт). Детектор этикетки вызывается один
раз, оба вида режутся из одного кадра.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from .image import _clahe, _resize, decode
from .label_crop import Box, LabelDetector, backdrop_box, crop, gray_world


@dataclass(slots=True)
class QueryViews:
    embed: Image.Image
    ocr: Image.Image
    box: Box | None
    # Бутылка целиком на однотонном фоне (пэкшот, фото с экрана) — кадр того же
    # вида, что эталоны каталога. Эмбеддер сравнивает его и без адаптера запросов.
    packshot: bool = False


class QueryNormalizer:
    def __init__(self, config: dict, root: Path) -> None:
        self.config = config
        crop_cfg = config.get("crop") or {}
        self.detector: LabelDetector | None = None
        if crop_cfg.get("enabled"):
            weights = Path(crop_cfg.get("weights", "artifacts/models/label_yolov8n.pt"))
            weights = weights if weights.is_absolute() else root / weights
            if weights.exists():
                self.detector = LabelDetector(
                    weights,
                    confidence=float(crop_cfg.get("confidence", 0.25)),
                    center_weight=float(crop_cfg.get("center_weight", 0.3)),
                )
            else:
                raise FileNotFoundError(
                    f"normalize.crop.enabled, но нет весов детектора: {weights}"
                )
        self.embed_margin = float(crop_cfg.get("margin", 0.08))
        self.ocr_margin = float(crop_cfg.get("ocr_margin", 0.15))
        self.white_balance = bool(config.get("white_balance", True))
        self.backdrop = config.get("backdrop") or {}

    def __call__(self, payload: bytes) -> QueryViews:
        # JPEG уменьшается при декодировании, только если обе стороны >= 2x рабочего
        # размера: обычное 12 Мп фото (4032x3024) декодируется как раньше.
        return self.from_image(decode(payload, int(self.config.get("max_side", 1600))))

    def from_image(self, image: Image.Image) -> QueryViews:
        # Детектору и эмбеддеру хватает 1280 px; OCR получает кроп из того же
        # кадра — этого достаточно, чтобы читать название, а не мелкий шрифт.
        image = _resize(image, int(self.config.get("max_side", 1600)))
        if self.backdrop.get("enabled", True):
            # Однотонный фон: детектор этикетки (обучен на полках) ловит на нём
            # полосы фона, а эталон каталога — вся бутылка. Режем по фону, без
            # баланса белого: «серый мир» по кадру бутылки уводит цвета этикетки.
            found = backdrop_box(image, int(self.backdrop.get("tolerance", 30)),
                                 float(self.backdrop.get("min_uniform", 0.6)))
            if found is not None:
                bottle = crop(image, found, 0.0)
                return QueryViews(embed=bottle, ocr=bottle, box=None, packshot=True)
        box = self.detector.detect(image) if self.detector is not None else None

        embed = crop(image, box, self.embed_margin)
        if self.white_balance:
            embed = gray_world(embed)
        clahe = self.config.get("clahe") or {}
        if clahe.get("enabled", False):
            embed = _clahe(embed, float(clahe.get("clip_limit", 2.0)),
                           int(clahe.get("tile_grid", 8)))

        return QueryViews(embed=embed, ocr=crop(image, box, self.ocr_margin), box=box)
