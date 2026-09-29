"""Нормализация входного изображения.

Кейсодержатель прямым текстом назвал этот слой главным резервом точности:
эталоны каталога студийные, запрос — фото с телефона в магазине. Всё, что
сближает два распределения, живёт здесь.

Сейчас реализованы обязательные вещи, которые ломают пайплайн молча, если их
нет: EXIF-ориентация, HEIC, единое цветовое пространство и размер, CLAHE.
Кроп этикетки и выпрямление по гомографии придут сюда же — интерфейс
`normalize()` при этом не меняется.
"""

from __future__ import annotations

import io
import logging
import warnings

import cv2
import numpy as np
from PIL import Image, ImageFile, ImageOps

log = logging.getLogger(__name__)

try:  # iPhone снимает в HEIC, а оценочный набор — реальные пользовательские фото
    import pillow_heif

    pillow_heif.register_heif_opener()
    HEIC_SUPPORTED = True
except ImportError:  # pragma: no cover - зависит от окружения
    HEIC_SUPPORTED = False
    log.warning("pillow-heif не установлен: HEIC-файлы не будут открываться")


# Обрезанный при передаче JPEG лучше распознать по верхней части, чем отвергнуть.
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Декомпрессионная бомба: PNG 12000x10000 весит килобайты, а в памяти — сотни МБ.
# Предел с запасом над 48 Мп iPhone и самым большим фото датасета кейса (80 Мп).
Image.MAX_IMAGE_PIXELS = 100_000_000
# Выше предела Pillow лишь предупреждает (ошибка — только с двойного): делаем ошибкой.
warnings.simplefilter("error", Image.DecompressionBombWarning)


class ImageDecodeError(ValueError):
    """Файл не является изображением или повреждён."""


def decode(payload: bytes, max_side: int | None = None) -> Image.Image:
    """Байты запроса -> RGB-изображение с учётом EXIF-ориентации.

    `exif_transpose` обязателен: без него часть фото уходит в модель повёрнутой
    на 90°, и визуальный эмбеддинг деградирует тихо, без единой ошибки в логах.

    `max_side` — JPEG декодируется сразу уменьшенным в 2/4/8 раз (DCT-scaling),
    но обе стороны остаются не меньше max_side. Фото на 50 МБ иначе съедает
    секунды SLA на одно декодирование; 12 Мп фото при 1600 не меняются.
    """
    try:
        image = Image.open(io.BytesIO(payload))
        # Фильтр warnings снимается через -W или pytest — размер проверяем и явно.
        if image.width * image.height > Image.MAX_IMAGE_PIXELS:
            raise Image.DecompressionBombError(f"{image.width}x{image.height} больше предела")
        if max_side and image.format == "JPEG":
            image.draft("RGB", (max_side, max_side))
        image.load()
    except Exception as exc:  # Pillow бросает широкий спектр исключений
        raise ImageDecodeError(f"не удалось декодировать изображение: {exc}") from exc

    image = ImageOps.exif_transpose(image)
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image


def _resize(image: Image.Image, max_side: int) -> Image.Image:
    width, height = image.size
    longest = max(width, height)
    if longest <= max_side:
        return image
    scale = max_side / longest
    return image.resize((round(width * scale), round(height * scale)), Image.LANCZOS)


def _clahe(image: Image.Image, clip_limit: float, tile_grid: int) -> Image.Image:
    """Выравнивание освещённости в LAB: магазинный свет против студийного.

    Работаем только по каналу L — иначе поедет цвет этикетки, а он несёт сигнал.
    """
    lab = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2LAB)
    light, a, b = cv2.split(lab)
    equalizer = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid, tile_grid))
    merged = cv2.merge((equalizer.apply(light), a, b))
    return Image.fromarray(cv2.cvtColor(merged, cv2.COLOR_LAB2RGB))


def normalize(image: Image.Image, config: dict | None = None) -> Image.Image:
    """Приводит фото запроса и эталон каталога к одному виду.

    Вызывается и на индексации, и на инференсе — строго одной и той же
    функцией. Любое расхождение между этими двумя путями означает, что мы
    ищем в одном пространстве, а индексировали в другом.
    """
    config = config or {}
    image = _resize(image, int(config.get("max_side", 1024)))

    clahe_cfg = config.get("clahe") or {}
    if clahe_cfg.get("enabled", True):
        image = _clahe(
            image,
            float(clahe_cfg.get("clip_limit", 2.0)),
            int(clahe_cfg.get("tile_grid", 8)),
        )

    # TODO(accuracy): crop.enabled -> детекция бутылки/этикетки + гомография.
    # Включается только после замера на hold-out, не раньше.
    return image


def load_and_normalize(payload: bytes, config: dict | None = None) -> Image.Image:
    return normalize(decode(payload), config)
