"""«Цифровой сомелье»: блюдо -> вина разных хозяйств и почему они подходят.

Стоит на том же словаре блюд, что и «к чему подать» в карточке: вино
подходит к блюду, если блюдо есть в его гастрономии (совет винодельни или
правило по сорту и стилю). Поэтому «сыры» — это и выдержанное красное,
и сладкое белое к голубым сырам, и оранжевое, а не только 16 оранжевых вин.
Если сочетание фильтров пустое, они ослабляются по одному — и ответ
прямо говорит об этом.
"""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass, field

from ..catalog.loader import Catalog
from ..catalog.models import Wine
from .pairing import DISHES, pairing_for
from .traits import traits

# id совпадают с FOOD в frontend/app.js.
FOODS: dict[str, tuple[str, frozenset[str]]] = {
    "meat": ("Красное мясо, дичь", frozenset({"red_meat", "lamb", "game", "bbq", "meat"})),
    "poultry": ("Птица, паста, овощи", frozenset({"poultry", "duck", "white_meat", "veal_pork",
                                                  "pasta", "risotto", "vegetables", "mushrooms"})),
    "fish": ("Рыба, морепродукты", frozenset({"fish", "seafood", "oysters"})),
    "cheese": ("Сыры, пряная кухня", frozenset({"hard_cheese", "soft_cheese", "blue_cheese",
                                                "goat_cheese", "cheese", "spicy", "asian"})),
    "aperitif": ("Аперитив, праздник", frozenset({"aperitif", "snacks", "salads"})),
    "dessert": ("Десерты, фрукты", frozenset({"dessert", "fruit", "foie"})),
}
PICKS = 6
# Не больше двух вин одного стиля и сорта, на первом проходе — и одного стиля:
# «сыры» должны показать и красное, и белое, и оранжевое.
PER_KIND = 2


@dataclass(slots=True)
class Pick:
    wine: Wine
    score: float
    why: str


@dataclass(slots=True)
class Selection:
    items: list[Pick]
    total: int
    relaxed: bool
    used: dict[str, object]
    regions: list[tuple[str, int]] = field(default_factory=list)


def _match(wine: Wine, dishes: frozenset[str]) -> tuple[float, str]:
    """-> (сила пары, «почему подходит»). 0 — вино к блюду не подходит."""
    pairing = pairing_for(wine)
    hits = [(i, d, b) for i, (d, b) in enumerate(zip(pairing.dishes, pairing.bases,
                                                      strict=True)) if d in dishes]
    if not hits:
        return 0.0, ""
    # Первое блюдо в гастрономии — самое характерное для вина.
    score = sum(1.0 / (1 + i) for i, _, _ in hits)
    if pairing.source == "description":
        score += 0.3  # совет самой винодельни весомее общего правила
    basis = hits[0][2]
    labels = [DISHES[d][0].lower() for _, d, b in hits if b == basis]
    return score, f"{basis[:1].upper()}{basis[1:]}: {', '.join(labels[:3])}"


def pick(
    catalog: Catalog,
    food: str,
    region: str = "",
    sparkling: bool | None = None,
    seed: int | None = None,
    limit: int = PICKS,
) -> Selection:
    dishes = FOODS[food][1] if food in FOODS else frozenset()
    matched: list[Pick] = []
    for wine in catalog:
        score, why = _match(wine, dishes) if dishes else (1.0, "")
        if score > 0:
            matched.append(Pick(wine, score, why))

    tries: list[dict[str, object]] = [
        {"region": region, "sparkling": sparkling},
        {"sparkling": sparkling},
        {"region": region},
        {},
    ]
    used: dict[str, object] = {}
    chosen: list[Pick] = []
    for params in tries:
        chosen = [p for p in matched if _passes(p.wine, params)]
        if chosen:
            used = {k: v for k, v in params.items() if v not in ("", None)}
            break

    regions = Counter(p.wine.region for p in matched
                      if p.wine.region and _passes(p.wine, {"sparkling": sparkling}))
    requested = {k: v for k, v in tries[0].items() if v not in ("", None)}
    return Selection(
        items=_diverse(chosen, seed, limit),
        total=len(chosen),
        relaxed=used != requested,
        used=used,
        regions=regions.most_common(),
    )


def _passes(wine: Wine, params: dict[str, object]) -> bool:
    region, sparkling = params.get("region"), params.get("sparkling")
    if region and " ".join(wine.region.lower().split()) != " ".join(str(region).lower().split()):
        return False
    return sparkling is None or traits(wine).sparkling == sparkling


def _diverse(picks: list[Pick], seed: int | None, limit: int) -> list[Pick]:
    """Сильные пары вперёд, но с разбросом: «Подобрать заново» даёт другие
    вина. Одно вино на хозяйство, не больше PER_KIND на стиль и сорт.
    Предел на стиль снимается вторым проходом, если других стилей к блюду
    нет (к стейку — только красные)."""
    rng = random.Random(seed)
    ranked = sorted(picks, key=lambda p: (-(p.score + rng.random() * 0.6), p.wine.slug))
    wineries: set[str] = set()
    kinds: Counter[tuple[str, str]] = Counter()
    styles: Counter[str] = Counter()
    result: list[Pick] = []
    for style_cap in (max(1, limit // 3), limit):
        for p in ranked:
            t = traits(p.wine)
            winery = (p.wine.winery or p.wine.slug).lower()
            kind = (t.style, t.grape_names[0][0] if t.grape_names else "")
            if winery in wineries or kinds[kind] >= PER_KIND or styles[t.style] >= style_cap:
                continue
            wineries.add(winery)
            kinds[kind] += 1
            styles[t.style] += 1
            result.append(p)
            if len(result) == limit:
                return result
    return result
