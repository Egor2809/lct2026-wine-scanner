"""Признаки вина для функций после поиска: сорта, стиль, игристость, сахар, аромат.

Дамп не даёт ни сахара, ни гастрономии, а сорт лежит строкой купажа
(«Каберне Совиньон, Мерло»). Здесь это превращается в сравнимые множества —
на них стоят аналоги, «к чему подать» и сомелье. Тот же разбор умеет читать
текст этикетки (OCR): на нём строится профиль вина, которого нет в каталоге.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

from ..catalog.browse import is_sparkling
from ..catalog.models import Wine

# Стили — ровно как в дампе: по ним же работают фильтры каталога.
STYLES = ("Красное", "Белое", "Розовое", "Оранжевое")
_STYLE_WORDS = {
    "Красное": ("красн", "red", "rouge", "rosso", "tinto", "рубин", "гранат"),
    "Белое": ("бел", "white", "blanc", "bianco", "blanco", "солом", "золот"),
    "Розовое": ("розов", "rose", "rosé", "rosato", "rosado", "лосос"),
    "Оранжевое": ("оранж", "orange", "янтар"),
}

# Сахар — только из названия и описания: отдельной колонки в дампе нет.
# Порядок важен: брют раньше «сухого» (extra dry), «полусухое» раньше «сухого».
_SUGAR = (
    ("брют", r"брют|\bbrut\b"),
    ("полусухое", r"полусух|semi[\s-]?dry|demi[\s-]?sec|\bextra[\s-]?dry\b"),
    ("полусладкое", r"полуслад|semi[\s-]?sweet|moelleux|amabile"),
    ("сладкое", r"(?<!полу)сладк|десертн|\bsweet\b|\bdoux\b|dolce|ликёрн|ликерн|"
                r"ice[\s-]?wine|айсвайн|ледяное вино|поздн\w* сбор"),
    ("сухое", r"(?<!полу)сух|\bdry\b|\bsec\b|secco|trocken"),
)
_SUGAR_RE = [(name, re.compile(pattern)) for name, pattern in _SUGAR]
SWEET = frozenset({"полусладкое", "сладкое"})

_SPARKLING_LABEL = re.compile(
    r"игрист|брют|\bbrut\b|sparkling|spumante|frizzante|шампанск|champagne|cr[ée]mant|"
    r"pet[\s-]?nat|петнат|пет-нат|m[ée]thode|traditionnelle|классическ\w* метод"
)

# Синонимы и семейства сортов. Ключ — вид строки после чистки, значение —
# под каким именем сорт сравнивается. «Мускат белый» и «Мускат Оттонель»
# для гостя — один мускат; «Каберне Фран» и «Каберне Совиньон» — разные вина.
_GRAPE_SYNONYMS = {
    "шираз": "сира",
    "рислинг рейнский": "рислинг",
    "пино гриджио": "пино гри",
    "пино черный": "пино нуар",
    "пино менье": "менье",
    "кокур белый": "кокур",
    "гевюрцтраминер": "траминер",
    "траминер розовый": "траминер",
    "мцване кахетинский": "мцване",
    "ркацители оранж": "ркацители",
    "бьянка": "бианка",
    "бастардо": "бастардо магарачский",
    "цвайгельт таманский": "цвайгельт",
}
_GRAPE_FAMILIES = ("мускат", "красностоп")
# «Белые сорта винограда» — не сорт, сравнивать по нему нельзя.
_NOT_A_GRAPE = re.compile(r"сорта винограда|купаж|ассамбляж")
_GRAPE_SPLIT = re.compile(r"[,;/+]|\s+и\s+|\s+&\s+")

# Как сорт пишут на этикетке. Только известные сорта: короткие редкие имена
# («Олег», «Восторг», «Шабаш») на этикетке чаще оказываются просто словами.
_GRAPE_LABEL = {
    "каберне совиньон": r"каберне[\s-]совиньон|cabernet[\s-]sauvignon",
    "каберне фран": r"каберне[\s-]фран|cabernet[\s-]franc",
    "мерло": r"мерло|merlot",
    "пино нуар": r"пино[\s-]нуар|pinot[\s-]noir|пино черн",
    "пино гри": r"пино[\s-]гри|pinot[\s-]gri",
    "пино блан": r"пино[\s-]блан|pinot[\s-]blanc",
    "шардоне": r"шардоне|chardonnay",
    "рислинг": r"рислинг|riesling",
    "совиньон блан": r"совиньон[\s-]блан|sauvignon[\s-]blanc",
    "сира": r"\bсира\b|шираз|syrah|shiraz",
    "саперави": r"саперави|saperavi",
    "ркацители": r"ркацители|rkatsiteli|rkaciteli",
    "алиготе": r"алиготе|aligot",
    "вионье": r"вионье|viognier",
    "мускат": r"мускат|muscat|moscato",
    "красностоп": r"красностоп|krasnostop",
    "цимлянский черный": r"цимлянск|tsimlyansk",
    "сибирьковый": r"сибирьков|sibirkov",
    "кокур": r"кокур|kokur",
    "мальбек": r"мальбек|malbec",
    "марселан": r"марселан|marselan",
    "пти вердо": r"пти[\s-]вердо|petit[\s-]verdot",
    "санджовезе": r"санджовезе|sangiovese",
    "темпранильо": r"темпранильо|tempranillo",
    "траминер": r"траминер|traminer",
    "шенен блан": r"шенен|chenin",
    "семильон": r"семильон|s[ée]millon",
    "цитронный магарача": r"цитронн",
    "бастардо магарачский": r"бастардо|bastardo",
}
_GRAPE_LABEL_RE = {key: re.compile(pattern) for key, pattern in _GRAPE_LABEL.items()}

_REGION_LABEL = {
    "Крым": r"крым|crimea|севастопол|sevastopol|инкерман|массандр|балаклав",
    "Кубань": r"кубан|kuban|тамань|taman|анап|anapa|абрау|геленджик",
    "Долина Дона": r"долин\w* дона|don valley|ростов|цимлянск",
    "Дагестан": r"дагестан|dagestan|дербент|derbent",
    "Ставрополье": r"ставропол|stavropol",
    "Самара": r"самар|samara",
}
_REGION_LABEL_RE = {name: re.compile(pattern) for name, pattern in _REGION_LABEL.items()}

# Ароматические дескрипторы: основа -> подпись. Сравниваем по ним, а не по
# всем словам описания: «вкус», «послевкусие», «оттенки» есть у каждого вина.
_AROMAS = {
    r"вишн": "вишня", r"черешн": "черешня", r"ежевик": "ежевика",
    r"смородин": "смородина", r"малин": "малина", r"клубник|земляник": "клубника",
    r"чернослив": "чернослив", r"слив(?!оч|к)": "слива", r"инжир": "инжир",
    r"гранат(?!ов\w* цвет)": "гранат", r"яблок": "яблоко", r"груш": "груша",
    r"персик": "персик", r"абрикос|кураг": "абрикос", r"\bайв": "айва",
    r"лимон": "лимон", r"грейпфрут": "грейпфрут", r"цитрус|лайм|апельсин": "цитрусы",
    r"ананас|манго|маракуй|тропич": "тропические фрукты", r"\bдын": "дыня",
    r"\bм[её]д": "мёд", r"цветоч|\bцвет(?:ы|ов|ами)\b|акаци|липы|бузин": "цветы",
    r"фиалк": "фиалка", r"\bроз(?:а|ы|ой)\b": "роза", r"трав(?:ы|ян|ами|а\b)": "травы",
    r"перц|перец": "перец", r"пряност|специ": "пряности", r"ванил": "ваниль",
    r"\bдуб": "дуб", r"шоколад|какао": "шоколад", r"кофе": "кофе", r"табак": "табак",
    r"\bкож": "кожа", r"минерал": "минеральность", r"орех|миндал": "орехи",
    r"бриош|хлеб|выпечк|тост": "выпечка", r"карамел|ирис": "карамель",
    r"\bмят|эвкалипт": "мята", r"\bдым|копч": "дымок",
}
_AROMAS_RE = [(re.compile(pattern), name) for pattern, name in _AROMAS.items()]


def fold(value: str) -> str:
    return " ".join((value or "").lower().replace("ё", "е").split())


def grape_key(part: str) -> str:
    """Один сорт из строки купажа -> имя, под которым он сравнивается."""
    part = re.sub(r"\(.*?\)|\d+\s*%|%", " ", fold(part))
    part = " ".join(part.replace("-", " ").split())
    part = _GRAPE_SYNONYMS.get(part, part)
    for family in _GRAPE_FAMILIES:
        if part.startswith(family):
            return family
    return part


def split_grapes(grape: str) -> list[tuple[str, str]]:
    """«Каберне Совиньон, Мерло и Саперави» -> [(ключ, как в карточке), ...]."""
    pairs: list[tuple[str, str]] = []
    for part in _GRAPE_SPLIT.split(grape or ""):
        name = " ".join(part.split()).strip(" .")
        if not name or _NOT_A_GRAPE.search(fold(name)):
            continue
        key = grape_key(name)
        if key and key not in (k for k, _ in pairs):
            pairs.append((key, name))
    return pairs


def style_of(style: str, color: str = "") -> str:
    """Стиль из поля style, при пустом — по цвету («рубиновый» -> Красное)."""
    for text in (fold(style), fold(color)):
        for name, words in _STYLE_WORDS.items():
            if any(word in text for word in words):
                return name
    return ""


def sugar_of(*texts: str) -> str:
    for text in texts:
        text = fold(text)
        for name, pattern in _SUGAR_RE:
            if pattern.search(text):
                return name
    return ""


def aromas_of(text: str) -> frozenset[str]:
    text = fold(text)
    return frozenset(name for pattern, name in _AROMAS_RE if pattern.search(text))


@dataclass(frozen=True, slots=True)
class Traits:
    grapes: frozenset[str]
    # Ключ сорта -> как он написан в карточке: для подписей «общий сорт — Мерло».
    grape_names: tuple[tuple[str, str], ...]
    style: str
    sparkling: bool
    sugar: str
    region: str
    aromas: frozenset[str]

    def grape_name(self, key: str) -> str:
        return next((name for k, name in self.grape_names if k == key), key.capitalize())


def traits(wine: Wine) -> Traits:
    return _traits(wine.title, wine.grape, wine.style, wine.color, wine.region,
                   wine.description, wine.sugar)


@lru_cache(maxsize=8192)
def _traits(title: str, grape: str, style: str, color: str, region: str,
            description: str, sugar: str) -> Traits:
    pairs = split_grapes(grape)
    probe = Wine(slug="", title=title, description=description)
    return Traits(
        grapes=frozenset(k for k, _ in pairs),
        grape_names=tuple(pairs),
        style=style_of(style, color),
        sparkling=is_sparkling(probe),
        # Сахар в названии надёжнее описания: там «сладкие ноты» у сухих вин.
        sugar=sugar_of(sugar, title, style) or _sugar_from_note(description),
        region=" ".join((region or "").split()),
        aromas=aromas_of(description),
    )


def _sugar_from_note(description: str) -> str:
    """Из описания берём только явное «вино сухое/полусладкое», не «сладкие ноты»."""
    for found in re.finditer(r"вин\w*\s+(\w+)", fold(description)):
        sugar = sugar_of(found.group(1))
        if sugar:
            return sugar
    return ""


def label_traits(text: str) -> dict[str, object]:
    """Что удалось прочитать на этикетке: сорта, стиль, игристость, сахар, регион.

    Возвращает только найденное — пустые поля профиль добирает у кандидатов.
    """
    text = fold(text)
    if not text:
        return {}
    found: dict[str, object] = {}
    grapes = [key for key, pattern in _GRAPE_LABEL_RE.items() if pattern.search(text)]
    if grapes:
        found["grapes"] = grapes
    style = next((name for name, words in (
        ("Розовое", r"розовое|\bros[eé]\b|rosato|rosado"),
        ("Оранжевое", r"оранжев|\borange wine\b"),
        ("Красное", r"красное|\bred\b|\brouge\b|\brosso\b|\btinto\b"),
        ("Белое", r"белое|\bwhite\b|\bbianco\b|\bblanco\b"),
    ) if re.search(words, text)), "")
    if style:
        found["style"] = style
    if _SPARKLING_LABEL.search(text):
        found["sparkling"] = True
    sugar = sugar_of(text)
    if sugar:
        found["sugar"] = sugar
    region = next((name for name, pattern in _REGION_LABEL_RE.items() if pattern.search(text)), "")
    if region:
        found["region"] = region
    return found
