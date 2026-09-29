"""«К чему подать» — детерминированно, без модели.

В дампе нет колонки гастрономии, а ТЗ требует её в карточке. Источники по
убыванию доверия:
  1. колонка pairing дампа (если в новой выгрузке появится);
  2. совет винодельни в описании: «сочетается с…», «рекомендуется к…» —
     из предложения берутся только известные блюда, в именительном падеже;
  3. правила по сахару и игристости, затем по основному сорту, затем по стилю.
Один словарь блюд на всё: те же подписи видит сомелье, когда ищет вина к блюду.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

from ..catalog.models import Wine
from .traits import SWEET, Traits, fold, traits

MAX_DISHES = 4

# id -> (подпись, как блюдо пишут в описаниях). Порядок = порядок в выдаче
# при равенстве. Общие «мясо» и «сыры» — в конце: они нужны, только если
# конкретнее ничего не нашлось.
DISHES: dict[str, tuple[str, str]] = {
    "aperitif": ("Аперитив", r"аперитив|welcome"),
    "red_meat": ("Стейк из говядины", r"говя|стейк|ростбиф|красн\w* мяс"),
    "lamb": ("Баранина", r"баран|ягн[её]н|ягнят"),
    "game": ("Дичь", r"\bдич"),
    "bbq": ("Шашлык и гриль", r"шашлык|барбекю|на углях|мяс\w* на гриле"),
    "duck": ("Утка", r"\bутк|\bутин"),
    "poultry": ("Птица", r"птиц|курин|куриц|индейк|цыпл"),
    "white_meat": ("Белое мясо", r"бел\w* мяс"),
    "veal_pork": ("Телятина и свинина", r"телят|свин"),
    "fish": ("Рыба", r"рыб|лосос|судак|тунц|форел"),
    "seafood": ("Морепродукты", r"морепрод|кревет|мидии|кальмар|осьмин"),
    "oysters": ("Устрицы", r"устриц"),
    "asian": ("Азиатская кухня", r"суши|азиат|восточн\w* кухн|тайск|карри"),
    "spicy": ("Пряная кухня", r"прян\w* (?:блюд|кухн)|остр\w* (?:блюд|закус)|кавказ"),
    "pasta": ("Паста и пицца", r"паст[аеуы]\b|пастой|пицц"),
    "risotto": ("Ризотто", r"ризотто"),
    "mushrooms": ("Грибы", r"гриб|трюфел|жюльен"),
    "vegetables": ("Овощные блюда", r"овощ"),
    "salads": ("Лёгкие салаты", r"салат"),
    "snacks": ("Закуски", r"закуск|тартар"),
    "foie": ("Фуа-гра", r"фуа"),
    "goat_cheese": ("Козий сыр", r"козь"),
    "soft_cheese": ("Мягкие сыры",
                    r"мягк\w* сыр|\bбри\b|камамбер|молод\w* сыр|нежирн\w* сыр|л[её]гк\w* сыр"),
    "hard_cheese": ("Выдержанные сыры",
                    r"тв[её]рд\w* сыр|выдерж\w* сыр|зрел\w* сыр|пармез|сыр\w* средн"),
    "blue_cheese": ("Голубые сыры", r"голуб\w* сыр|плесен"),
    "dessert": ("Десерты", r"десерт|выпечк|пирож|торт|тарт(?!ар)|безе|бисквит"),
    "fruit": ("Фрукты и ягоды", r"фрукт|ягод"),
    "meat": ("Мясные блюда", r"мяс"),
    "cheese": ("Сыры", r"сыр"),
}
_DISH_RE = {dish: re.compile(pattern) for dish, (_, pattern) in DISHES.items()}
_GENERIC = {
    "meat": {"red_meat", "lamb", "game", "bbq", "duck", "poultry", "white_meat", "veal_pork"},
    "cheese": {"goat_cheese", "soft_cheese", "hard_cheese", "blue_cheese"},
}

# Предложение с советом по подаче. «Сочетает кислотность с лёгкостью» — про
# вкус, а не про еду: нужен именно возвратный «сочетается».
_ADVICE = re.compile(
    r"сочета\w*ся|рекоменд|подава|гастрономи\w* сочетан|в гастрономии|гастрономия:|"
    r"аперитив|идеальн\w* к\b|дополн\w* (?:к|блюд)|в паре с|составит пару|^к\s"
)
_SENTENCE = re.compile(r"[.!?;\n]+")

# Правила по основному сорту: классические пары, по 2–3 блюда.
_BY_GRAPE: dict[str, tuple[str, ...]] = {
    "каберне совиньон": ("red_meat", "lamb", "hard_cheese"),
    "мерло": ("duck", "veal_pork", "pasta"),
    "каберне фран": ("veal_pork", "vegetables", "game"),
    "пино нуар": ("duck", "mushrooms", "fish"),
    "сира": ("bbq", "lamb", "spicy"),
    "саперави": ("bbq", "lamb", "hard_cheese"),
    "красностоп": ("red_meat", "game", "hard_cheese"),
    "цимлянский черный": ("game", "bbq", "hard_cheese"),
    "мальбек": ("red_meat", "bbq"),
    "марселан": ("lamb", "red_meat", "spicy"),
    "темпранильо": ("pasta", "veal_pork", "hard_cheese"),
    "санджовезе": ("pasta", "veal_pork", "hard_cheese"),
    "пти вердо": ("game", "red_meat"),
    "бастардо магарачский": ("game", "hard_cheese"),
    "шардоне": ("poultry", "fish", "risotto"),
    "совиньон блан": ("goat_cheese", "seafood", "salads"),
    "рислинг": ("asian", "fish", "veal_pork"),
    "алиготе": ("oysters", "fish", "snacks"),
    "ркацители": ("fish", "snacks", "soft_cheese"),
    "вионье": ("poultry", "asian"),
    "пино блан": ("fish", "pasta", "salads"),
    "пино гри": ("fish", "pasta", "salads"),
    "мускат": ("aperitif", "asian", "fruit"),
    "кокур": ("fish", "soft_cheese"),
    "траминер": ("asian", "foie", "blue_cheese"),
    "цитронный магарача": ("seafood", "aperitif", "salads"),
    "шенен блан": ("veal_pork", "soft_cheese"),
    "семильон": ("foie", "fish"),
    "сибирьковый": ("fish", "vegetables"),
    "бианка": ("fish", "salads"),
}
# Какой стиль у сорта в чистом виде: красный сорт в розовом вине или
# blanc de noirs правило сорта не применяем — там решает стиль.
_RED_GRAPES = frozenset({"каберне совиньон", "мерло", "каберне фран", "пино нуар", "сира",
                         "саперави", "красностоп", "цимлянский черный", "мальбек", "марселан",
                         "темпранильо", "санджовезе", "пти вердо", "бастардо магарачский"})

_BY_STYLE: dict[str, tuple[str, ...]] = {
    "Красное": ("red_meat", "hard_cheese", "mushrooms"),
    "Белое": ("fish", "poultry", "salads"),
    "Розовое": ("salads", "seafood", "pasta"),
    "Оранжевое": ("hard_cheese", "spicy", "mushrooms"),
}


@dataclass(frozen=True, slots=True)
class Pairing:
    dishes: tuple[str, ...]
    # description — совет винодельни из описания; rules — вывод по сорту и стилю.
    source: str
    # Основание для каждого блюда, для строки «почему»: «Каберне Совиньон»,
    # «брют», «совет винодельни». Параллельно dishes.
    bases: tuple[str, ...]

    @property
    def labels(self) -> list[str]:
        return [DISHES[d][0] for d in self.dishes]

    @property
    def basis(self) -> str:
        return self.bases[0] if self.bases else ""


def pairing_for(wine: Wine) -> Pairing:
    return _pairing(wine.description, traits(wine))


@lru_cache(maxsize=8192)
def _pairing(description: str, t: Traits) -> Pairing:
    advised = from_description(description)
    if advised:
        advised = advised[:MAX_DISHES]
        return Pairing(advised, "description", ("совет винодельни",) * len(advised))
    return _by_rules(t)


def from_description(description: str) -> tuple[str, ...]:
    """Блюда из советов по подаче в описании, в порядке упоминания."""
    found: list[tuple[int, str]] = []
    for sentence in _SENTENCE.split(fold(description)):
        sentence = sentence.strip(" —–-:")
        if not _ADVICE.search(sentence):
            continue
        for dish, pattern in _DISH_RE.items():
            match = pattern.search(sentence)
            if match and dish not in (d for _, d in found):
                found.append((match.start(), dish))
    dishes = [d for _, d in sorted(found)]
    dishes = [d for d in dishes if not (d in _GENERIC and _GENERIC[d] & set(dishes))]
    # «Сочетается со спелыми ягодами ежевики» — это аромат, а не еда.
    return tuple(dishes) if set(dishes) - {"fruit"} else ()


def _by_rules(t: Traits) -> Pairing:
    dishes: list[str] = []
    bases: list[str] = []

    def add(items: tuple[str, ...], why: str) -> None:
        for dish in items:
            if dish not in dishes:
                dishes.append(dish)
                bases.append(why)

    if t.sparkling:
        if t.sugar in SWEET:
            rule = ("dessert", "fruit", "aperitif")
        elif t.style == "Красное":  # красное игристое — к закускам и мясу, не к устрицам
            rule = ("aperitif", "snacks", "meat")
        else:
            rule = ("aperitif", "oysters", "seafood", "snacks")
        add(rule, "брют" if t.sugar == "брют" else f"игристое {t.sugar}".strip())
    elif t.sugar == "сладкое":
        add(("dessert", "blue_cheese", "fruit"), "сладкое вино")
    elif t.sugar == "полусладкое":
        add(("asian", "fruit", "dessert"), "полусладкое вино")

    if t.style in ("Красное", "Белое") and not t.sparkling:
        for key, name in t.grape_names[:2]:
            rule = _BY_GRAPE.get(key)
            if rule and (key in _RED_GRAPES) == (t.style == "Красное"):
                add(rule, name)

    add(_BY_STYLE.get(t.style, _BY_STYLE["Белое"]), f"{t.style.lower() or 'белое'} вино")
    return Pairing(tuple(dishes[:MAX_DISHES]), "rules", tuple(bases[:MAX_DISHES]))
