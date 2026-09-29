"""Нормализованное представление позиции каталога."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

# Год урожая в slug: ...-2021, ...-2021-2 (переиздание), ...-nv (non vintage).
_VINTAGE_SUFFIX = re.compile(r"-(?:19|20)\d{2}(?:-\d+)?$|-nv$", re.IGNORECASE)
# Год урожая внутри названия: «Каберне Совиньон 2020».
_YEAR_IN_TITLE = re.compile(r"\b(?:19|20)\d{2}\b")


def slug_stem(slug: str) -> str:
    """Slug без хвоста винтажа.

    Кейсодержатель сравнивает slug нечётко: «если мы найдём то вино только
    другого винтажа, мы считаем, что это правильное попадание». Все сравнения
    в сервисе и в evaluate.py идут через эту функцию.
    """
    return _VINTAGE_SUFFIX.sub("", (slug or "").strip().lower())


def same_wine(a: str, b: str) -> bool:
    """Нечёткое равенство slug по правилам оценки."""
    return bool(a) and slug_stem(a) == slug_stem(b)


@dataclass(slots=True)
class Wine:
    slug: str
    title: str = ""
    winery: str = ""
    region: str = ""
    grape: str = ""
    style: str = ""
    color: str = ""
    vintage: str = ""
    sugar: str = ""
    description: str = ""
    pairing: list[str] = field(default_factory=list)
    # Откуда build_index.py берёт эталон: локальный путь или внешний URL.
    image: str = ""
    # Что уходит в карточку браузеру. Для локальных файлов это HTTP-путь:
    # абсолютный путь файловой системы браузер запросит от корня сервера
    # и получит 404.
    image_url: str = ""
    url: str = ""
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def stem(self) -> str:
        return slug_stem(self.slug)

    @property
    def identity(self) -> str:
        """Ключ «то же вино»: одинаковые название и винодельня.

        В каталоге есть карточки-близнецы, отличающиеся только хвостом slug
        («…-shardone» и «…-shardone-1»): другой винтаж или дубль. Для
        уверенности ответа это одно вино.
        """
        # Год в названии («Каберне Совиньон 2020» / «… 2021») — это винтаж, а другой
        # винтаж того же вина кейсодержатель засчитывает как попадание.
        title = " ".join(_YEAR_IN_TITLE.sub(" ", self.title.lower()).split())
        return f"{title}|{' '.join(self.winery.lower().split())}" if title else self.stem

    def card(self) -> dict[str, Any]:
        """Карточка в том виде, в каком её ждёт фронтенд.

        Ключи совпадают с frontend/app.js: title, winery, region, grape,
        style, color, vintage, note, pairing, image, url.
        """
        return {
            "title": self.title,
            "winery": self.winery,
            "region": self.region,
            "grape": self.grape,
            "style": self.style,
            "color": self.color,
            "vintage": self.vintage,
            "note": self.description,
            "pairing": self.pairing,
            "image": self.image_url or (
                self.image if self.image.startswith(("http://", "https://")) else ""),
            "url": self.url,
        }
