"""Подбор российских аналогов.

Кейсодержатель отметил этот сценарий отдельно: вариант с аналогами
«оценивается особенно высоко». Работает в обе стороны — и когда вино
найдено уверенно (похожие позиции от других виноделен), и когда уверенности
нет (замена вина, которого нет в каталоге).

Подбор идёт по карточке, а не по вектору: пользователю нужен близкий вкус,
а не похожая этикетка. Сорта сравниваются множествами (Jaccard по купажу):
«Каберне Фран» — не «Каберне Совиньон», а купаж из семи сортов — не чистый
сорт. У каждого аналога есть причины — их показывает интерфейс.

Без уверенного ответа лучший кандидат заведомо неверен, поэтому аналоги
строятся от профиля: взвешенное голосование топ-кандидатов плюс то, что
прочитано на этикетке (сорт, стиль, регион на этикетке надёжнее соседей
по картинке).
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from ..catalog.loader import Catalog
from ..catalog.models import Wine
from .traits import SWEET, Traits, label_traits, traits

# Веса близости. Стиль, игристость и сладость — рамка (красное к красному,
# брют к брюту), сорт —
# главное внутри рамки, аромат различает вина одного сорта, регион — довесок:
# в каталоге всего девять регионов, половина вин — Кубань.
W_STYLE, W_STYLE_MISS = 3.0, -3.0
W_SPARKLING_MISS = -4.0
# Сладость так же определяет вино, как цвет: к сладкому рислингу ближе
# сладкий мускат, чем сухой рислинг. Сахар не указан — значит сухое.
W_SUGAR_MISS = -4.5
W_GRAPES = 4.0
W_AROMA = 2.0
AROMA_FLOOR = 4
W_REGION = 0.75
# Сколько кандидатов голосуют за профиль и какая доля голосов нужна полю.
PROFILE_TOP = 5
PROFILE_SHARE = 0.4

REASON_GRAPE = "тот же сорт"
REASON_BLEND = "тот же купаж"
REASON_STYLE = "тот же стиль"
REASON_REGION = "тот же регион"


@dataclass(slots=True)
class Analog:
    wine: Wine
    score: float
    reasons: list[str] = field(default_factory=list)


@dataclass(slots=True)
class Profile:
    """Каким было вино на фото, если его нет в каталоге."""

    grapes: frozenset[str] = frozenset()
    grape_names: tuple[tuple[str, str], ...] = ()
    style: str = ""
    sparkling: bool | None = None
    sugar: str = ""
    region: str = ""
    # Какие поля взяты с этикетки, а не у кандидатов — для честной подписи в UI.
    from_label: list[str] = field(default_factory=list)

    def traits(self) -> Traits:
        return Traits(self.grapes, self.grape_names, self.style, bool(self.sparkling),
                      self.sugar, self.region, frozenset())

    def as_dict(self) -> dict[str, Any]:
        return {
            "style": self.style,
            "grapes": [name for _, name in self.grape_names],
            "region": self.region,
            "sparkling": self.sparkling,
            "sugar": self.sugar,
            "from_label": self.from_label,
        }


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


def similarity(target: Traits, other: Traits,
               sparkling_known: bool = True) -> tuple[float, list[str]]:
    """Близость по карточке -> (скор, причины для интерфейса)."""
    score, reasons = 0.0, []

    same_style = bool(target.style) and target.style == other.style
    if target.style and other.style:
        score += W_STYLE if same_style else W_STYLE_MISS
    same_fizz = not sparkling_known or target.sparkling == other.sparkling
    if not same_fizz:
        score += W_SPARKLING_MISS
    same_sugar = (target.sugar in SWEET) == (other.sugar in SWEET)
    if not same_sugar:
        score += W_SUGAR_MISS

    grapes = _jaccard(target.grapes, other.grapes)
    if grapes:
        score += W_GRAPES * grapes
        if grapes == 1.0:
            reasons.append(REASON_GRAPE if len(target.grapes) == 1 else REASON_BLEND)
        else:
            common = [name for key, name in other.grape_names if key in target.grapes]
            reasons.append("общий сорт: " + ", ".join(common[:2]))
    if same_style and same_fizz and same_sugar:
        reasons.append(REASON_STYLE)

    if target.region and target.region == other.region:
        score += W_REGION
        reasons.append(REASON_REGION)

    # Не чистый Jaccard: у двух вин с единственным дескриптором «вишня» он
    # равен 1 и перевешивает сорт. Знаменатель не меньше AROMA_FLOOR.
    common = sorted(target.aromas & other.aromas)
    if common:
        score += W_AROMA * len(common) / max(AROMA_FLOOR, len(target.aromas | other.aromas))
        if len(common) >= 2:
            reasons.append("в аромате: " + ", ".join(common[:2]))

    return score, reasons


def _pick(
    scored: Iterable[Analog],
    count: int,
    skip_wineries: Iterable[str] = (),
    seed: str = "",
) -> list[Analog]:
    """Лучшие по убыванию, не больше одной позиции на винодельню — иначе
    выдача вырождается в четыре вина одного хозяйства.

    Равный скор у десятков вин одного сорта — обычное дело. Ничья решается
    хэшем от seed и slug: порядок стабилен для одного запроса, но не
    алфавитный — иначе любой «Мерло» получал бы одни и те же хозяйства на «А».
    """
    def tiebreak(slug: str) -> bytes:
        return hashlib.blake2b(f"{seed}|{slug}".encode(), digest_size=8).digest()

    ranked = sorted((a for a in scored if a.score > 0),
                    key=lambda a: (-round(a.score, 6), tiebreak(a.wine.slug)))
    seen = set(skip_wineries)
    result: list[Analog] = []
    for analog in ranked:
        key = (analog.wine.winery or analog.wine.slug).lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(analog)
        if len(result) == count:
            break
    return result


def match_analogs(catalog: Catalog, wine: Wine, count: int = 4) -> list[Analog]:
    """Аналоги найденного вина: только другие винодельни, с причинами."""
    target = traits(wine)
    scored = []
    for other in catalog:
        if other.slug == wine.slug or other.stem == wine.stem or other.identity == wine.identity:
            continue  # другой винтаж того же вина — это не аналог
        scored.append(Analog(other, *similarity(target, traits(other))))
    own = {(wine.winery or "").lower()} if wine.winery else set()
    return _pick(scored, count, own, seed=wine.stem)


def find_analogs(catalog: Catalog, wine: Wine, count: int = 4) -> list[Wine]:
    return [a.wine for a in match_analogs(catalog, wine, count)]


def build_profile(candidates: list[tuple[Wine, float]], label_text: str = "") -> Profile:
    """Профиль по топ-кандидатам (взвешенно по скору) и тексту этикетки."""
    top = [(w, s) for w, s in candidates if w is not None][:PROFILE_TOP]
    weights = [max(s, 0.0) for _, s in top]
    if not sum(weights):
        weights = [1.0 / (rank + 1) for rank in range(len(top))]
    total = sum(weights) or 1.0

    styles: dict[str, float] = defaultdict(float)
    regions: dict[str, float] = defaultdict(float)
    sugars: dict[str, float] = defaultdict(float)
    grapes: dict[str, float] = defaultdict(float)
    names: dict[str, str] = {}
    sparkling = 0.0
    for (wine, _), weight in zip(top, weights, strict=True):
        t = traits(wine)
        styles[t.style] += weight
        regions[t.region] += weight
        sugars[t.sugar] += weight
        sparkling += weight if t.sparkling else 0.0
        for key, name in t.grape_names:
            grapes[key] += weight
            names.setdefault(key, name)

    def leader(votes: dict[str, float]) -> str:
        votes = {k: v for k, v in votes.items() if k}
        if not votes:
            return ""
        best = max(votes, key=lambda k: (votes[k], k))
        return best if votes[best] / total >= PROFILE_SHARE else ""

    # Сорт: всё, за что проголосовала заметная доля; купаж соседей не
    # размывает профиль до «всего понемногу».
    grape_keys = [k for k, v in sorted(grapes.items(), key=lambda kv: -kv[1])
                  if v / total >= PROFILE_SHARE]
    profile = Profile(
        grapes=frozenset(grape_keys),
        grape_names=tuple((k, names[k]) for k in grape_keys),
        style=leader(styles),
        sparkling=sparkling / total >= 0.5 if top else None,
        sugar=leader(sugars),
        region=leader(regions),
    )

    label = label_traits(label_text)
    if label.get("grapes"):
        keys = list(label["grapes"])
        profile.grapes = frozenset(keys)
        profile.grape_names = tuple((k, names.get(k, k.capitalize())) for k in keys)
        profile.from_label.append("grapes")
    for key in ("style", "sparkling", "sugar", "region"):
        if key in label:
            setattr(profile, key, label[key])
            profile.from_label.append(key)
    return profile


def profile_analogs(
    catalog: Catalog,
    profile: Profile,
    count: int = 4,
    exclude: Iterable[Wine] = (),
    skip_wineries: Iterable[str] = (),
) -> list[Analog]:
    """Аналоги по профилю. Кандидаты из выдачи «возможно, одно из них»
    сюда не попадают — они уже показаны выше."""
    skip = {w.stem for w in exclude} | {w.identity for w in exclude}
    target = profile.traits()
    scored = []
    for other in catalog:
        if other.stem in skip or other.identity in skip:
            continue
        scored.append(Analog(other, *similarity(target, traits(other),
                                                sparkling_known=profile.sparkling is not None)))
    return _pick(scored, count, {w.lower() for w in skip_wineries if w},
                 seed=repr(profile.as_dict()))


def out_of_catalog(
    catalog: Catalog,
    candidates: list[tuple[Wine | None, float]],
    label_text: str,
    count: int = 4,
) -> tuple[Profile, list[Analog]]:
    """Вина нет в каталоге: профиль топ-кандидатов и этикетки -> аналоги.

    Кандидаты уже показаны в выдаче, а вино, скорее всего, той же винодельни,
    что и лучший из них: блок называется «других виноделен».
    """
    top = [(w, s) for w, s in candidates[:PROFILE_TOP] if w is not None]
    profile = build_profile(top, label_text)
    own = [top[0][0].winery] if top and top[0][0].winery else []
    found = profile_analogs(catalog, profile, count, exclude=[w for w, _ in top],
                            skip_wineries=own)
    return profile, found
