"""Текст с этикетки.

Второй сигнал поверх визуального эмбеддинга. Near-duplicates одной серии
(«Жемчужная 9 Цитрон, Шардоне» против «Жемчужная 9 Алиготе, Цитрон»)
визуально почти неотличимы, а различаются ровно текстом — его и читаем.

Контракт — протокол `TextExtractor`. Реализации взаимозаменяемы и выбираются
в config.yaml (`text.kind`):

- `yandex` — Yandex Vision OCR по API, ключ из ENV (YANDEX_API_KEY);
- `rapid`  — RapidOCR (PP-OCRv5 eslav, ONNX, CPU) — локальный OCR «в контуре»
             на любой ОС, без сети и без платных API;
- `apple`  — Apple Vision на macOS, офлайн, ~0.3–0.8 c;
- `local`  — Apple Vision на macOS, иначе RapidOCR;
- `auto`   — облако, если есть ключ, иначе локальный OCR (`local`); при сбое
             облака (сеть, 401, 429, таймаут) запрос обслуживает локальный. Это
             и есть страховка на созвон: отвалилась сеть — сервис отвечает;
- `none`   — только визуальный канал.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from PIL import Image

log = logging.getLogger(__name__)


@dataclass(slots=True)
class TextLine:
    text: str
    confidence: float = 1.0
    # Высота строки в долях кадра — размер шрифта. Крупный текст — название
    # вина, мелкий — юридическая информация и соседние бутылки.
    height: float = 0.0


@runtime_checkable
class TextExtractor(Protocol):
    name: str

    def extract(self, image: Image.Image) -> list[TextLine]: ...


class NullExtractor:
    name = "none"

    def extract(self, image: Image.Image) -> list[TextLine]:
        return []


class FallbackExtractor:
    """Основной экстрактор + запасной на случай исключения (сеть, квоты)."""

    def __init__(self, primary: TextExtractor, fallback: TextExtractor) -> None:
        self.primary = primary
        self.fallback = fallback
        self.name = f"{primary.name}|{fallback.name}"

    def extract(self, image: Image.Image) -> list[TextLine]:
        try:
            return self.primary.extract(image)
        except Exception as exc:
            log.warning("%s недоступен (%s), читаю через %s", self.primary.name, exc,
                        self.fallback.name)
            return self.fallback.extract(image)


def _rapid(config: dict) -> TextExtractor:
    from .rapid_ocr import RapidOCRExtractor

    if RapidOCRExtractor.available():
        return RapidOCRExtractor(max_side=int(config.get("max_side", 1600)),
                                 tiles=bool(config.get("rapid_tiles", True)),
                                 threads=int(config.get("rapid_threads", 2)))
    log.warning("RapidOCR не установлен: локальный текстовый канал выключен")
    return NullExtractor()


def _local(config: dict) -> TextExtractor:
    from .apple_vision import AppleVisionExtractor

    if (config.get("local") or "auto") != "rapid" and AppleVisionExtractor.available():
        return AppleVisionExtractor(max_side=int(config.get("max_side", 1600)))
    return _rapid(config)


def build_text_extractor(config: dict, api_key: str = "", folder_id: str = "") -> TextExtractor:
    kind = (config.get("kind") or "auto").lower()
    if kind == "none":
        return NullExtractor()
    if kind == "rapid":
        return _rapid(config)
    if kind in {"apple", "local"}:
        return _local(config)

    if kind in {"yandex", "auto"}:
        if not api_key:
            # Без ключа сервис обязан подняться: на проверке ключ могут не передать.
            log.warning("YANDEX_API_KEY не задан: облачный OCR выключен, работаю без него")
            return _local(config)
        from .yandex_ocr import YandexOCRExtractor

        cloud = YandexOCRExtractor(
            api_key=api_key,
            folder_id=folder_id,
            deadline=float(config.get("deadline", 1.6)),
            attempt_timeout=float(config.get("attempt_timeout", 1.2)),
            cooldown=float(config.get("cooldown", 30.0)),
            trip_after=int(config.get("trip_after", 2)),
            min_interval=float(config.get("min_interval", 1.0)),
            max_side=int(config.get("max_side", 1600)),
        )
        return FallbackExtractor(cloud, _local(config))

    raise ValueError(f"Неизвестный text.kind: {kind}")
