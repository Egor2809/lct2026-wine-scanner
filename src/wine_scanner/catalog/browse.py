"""Просмотр каталога: поиск, фильтры, фасеты — для страниц каталога,
винодельни и сомелье в UI. Распознавание сюда не ходит."""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass

from .loader import Catalog
from .models import Wine

MAX_LIMIT = 60
# В дампе нет отдельного поля «игристое»: признак живёт в названии и описании.
# В описании корень «игрист» не годится: «премия российских игристых вин»,
# «пикантная игристость» у тихих вин. Там ищем только перляж и метод.
_SPARKLING_TITLE = ("игрист", "брют", "brut", "петнат", "пет-нат", "пет нат", "petnat", "pet-nat",
                    "pet nat", "шампанск", "frizzante", "фризанте", "spumante", "спуманте")
_SPARKLING_NOTE = ("перляж", "пузырьк", "метод шарма", "игристое вино")


def _fold(value: str) -> str:
    return " ".join((value or "").lower().replace("ё", "е").split())


def is_sparkling(wine: Wine) -> bool:
    title, note = _fold(wine.title), _fold(wine.description)
    return any(w in title for w in _SPARKLING_TITLE) or any(w in note for w in _SPARKLING_NOTE)


def _haystack(wine: Wine) -> str:
    return _fold(f"{wine.title} {wine.winery} {wine.grape} {wine.region}")


@dataclass(slots=True)
class Page:
    total: int
    items: list[Wine]
    styles: list[tuple[str, int]]
    regions: list[tuple[str, int]]


def browse(
    catalog: Catalog,
    q: str = "",
    style: str = "",
    region: str = "",
    winery: str = "",
    sparkling: bool | None = None,
    seed: int | None = None,
    limit: int = 24,
    offset: int = 0,
) -> Page:
    """Все слова запроса должны найтись в названии, винодельне, сорте или регионе.

    Фасеты считаются без собственного фильтра: иначе выбранный стиль
    оставляет в списке стилей только себя.
    """
    words = _fold(q).split()
    style_f, region_f, winery_f = _fold(style), _fold(region), _fold(winery)

    base = [
        w for w in catalog
        if all(word in _haystack(w) for word in words)
        and (not winery_f or _fold(w.winery) == winery_f)
        and (sparkling is None or is_sparkling(w) == sparkling)
    ]
    by_region = [w for w in base if not region_f or _fold(w.region) == region_f]
    by_style = [w for w in base if not style_f or _fold(w.style) == style_f]
    matched = [w for w in by_region if not style_f or _fold(w.style) == style_f]
    matched.sort(key=lambda w: (_fold(w.winery), _fold(w.title)))
    if seed is not None:
        # Подборка сомелье: первая страница — случайная выборка по всем
        # хозяйствам, а не первые по алфавиту.
        random.Random(seed).shuffle(matched)

    limit = max(1, min(int(limit), MAX_LIMIT))
    offset = max(0, int(offset))
    return Page(
        total=len(matched),
        items=matched[offset:offset + limit],
        styles=Counter(w.style for w in by_region if w.style).most_common(),
        regions=Counter(w.region for w in by_style if w.region).most_common(),
    )
