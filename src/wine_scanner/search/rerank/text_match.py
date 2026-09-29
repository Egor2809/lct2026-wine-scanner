"""Текстовый сигнал: насколько прочитанное с этикетки совпадает с карточкой.

Этикетки пишут то кириллицей, то латиницей («INKERMAN» против «Инкерман»),
а slug каталога — в своём транслите («czitron», «tsitronnyy»). Поэтому
обе стороны приводятся к одному «рыхлому» латинскому написанию, где
cz/ts/c, kh/h, й/j/i и удвоенные буквы неразличимы.

Скор кандидата — доля его «опознавательных» токенов, найденных на
этикетке, со взвешиванием по IDF каталога: «табия», «цитрон», «львица»
весят много, «белое», «сухое», «вино» — мало, но всё же различают
полусладкое и полусухое одной серии.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable

from rapidfuzz import fuzz, process

from ...catalog.models import Wine
from ...features.text import TextLine

_CYR2LAT = {
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "e",
    "ж": "zh", "з": "z", "и": "i", "й": "i", "к": "k", "л": "l", "м": "m",
    "н": "n", "о": "o", "п": "p", "р": "r", "с": "s", "т": "t", "у": "u",
    "ф": "f", "х": "h", "ц": "c", "ч": "ch", "ш": "sh", "щ": "sh",
    "ъ": "", "ы": "y", "ь": "", "э": "e", "ю": "yu", "я": "ya",
}
# Порядок важен: сначала длинные сочетания.
_LOOSE = [
    ("sch", "sh"), ("cz", "c"), ("tz", "c"), ("ts", "c"), ("kh", "h"), ("ph", "f"),
    ("yo", "e"), ("ye", "e"), ("iy", "i"), ("yy", "y"), ("ij", "i"), ("yj", "y"),
    ("j", "i"), ("w", "v"), ("x", "ks"), ("ck", "k"), ("q", "k"),
]
_DOUBLE = re.compile(r"(.)\1+")
_TOKEN = re.compile(r"[a-z0-9]+")
_YEAR = re.compile(r"^(19|20)\d{2}$")
_DEDUP_SUFFIX = re.compile(r"-\d$")
# Слова, которые есть почти на каждой этикетке и в каждой карточке.
_STOP = {
    "vino", "vine", "wine", "winery", "vinodelna", "vinodelnya", "vinodelnia", "i", "s", "v",
    "na", "ot", "iz", "the", "de", "la", "le", "del", "di", "and", "rossii", "rossia",
    "rossiia", "proizvedeno", "razlito", "sdelano", "god", "urozhai", "urozhaia",
}


def loose(token: str) -> str:
    token = "".join(_CYR2LAT.get(ch, ch) for ch in token.lower())
    for src, dst in _LOOSE:
        token = token.replace(src, dst)
    return _DOUBLE.sub(r"\1", token)


# Локальный OCR (RapidOCR) читает кириллицу латинскими двойниками:
# «CAMAPA» вместо «САМАРА», «Py6uH» вместо «Рубин». Для таких слов пробуем
# и кириллическое прочтение, оставляем то, что лучше ложится на словарь.
_HOMOGLYPH = str.maketrans({
    "A": "А", "a": "а", "B": "В", "C": "С", "c": "с", "E": "Е", "e": "е", "H": "Н",
    "K": "К", "k": "к", "M": "М", "O": "О", "o": "о", "P": "Р", "p": "р", "T": "Т",
    "X": "Х", "x": "х", "Y": "У", "y": "у", "u": "и", "n": "п", "r": "г", "6": "б",
    # стилизованные шрифты: «ОΛΕΓ» вместо «ОЛЕГ»
    "Α": "А", "Β": "В", "Γ": "Г", "Δ": "Д", "Ε": "Е", "Η": "Н", "Κ": "К", "Λ": "Л",
    "Μ": "М", "Ο": "О", "Π": "П", "Ρ": "Р", "Τ": "Т", "Φ": "Ф", "Χ": "Х", "Υ": "У",
    "α": "а", "ε": "е", "κ": "к", "λ": "л", "ο": "о", "π": "п", "ρ": "р", "τ": "т", "χ": "х",
})
_LATIN = re.compile(r"[A-Za-z]")
_GREEK = re.compile(r"[\u0370-\u03ff]")
_CYRILLIC = re.compile(r"[А-Яа-яЁё]")


def _as_cyrillic(word: str) -> str | None:
    """Кириллическое прочтение слова из латинских двойников; None — не похоже на двойник."""
    if not (_LATIN.search(word) or _GREEK.search(word)):
        return None
    alt = word.translate(_HOMOGLYPH)
    return None if _LATIN.search(alt) or _GREEK.search(alt) else alt


# Надписи с разрядкой («У С А Д Ь Б А»): OCR отдаёт их отдельными буквами.
_SPACED = re.compile(r"(?<!\S)(?:\w\s){2,}\w(?!\S)")


def _unspace(text: str) -> str:
    return _SPACED.sub(lambda m: m.group(0).replace(" ", ""), text)


def tokenize(text: str, *, keep_long_numbers: bool = False) -> list[str]:
    out = []
    for raw in _TOKEN.findall(loose(_unspace(text))):
        if raw in _STOP or _YEAR.match(raw):
            continue  # год урожая не различает вина: промах по винтажу засчитывается
        if raw.isdigit():
            # «9» из «Жемчужная 9» различает, «135» в slug — это крепость 13.5 %
            if len(raw) >= 3 and not keep_long_numbers:
                continue
        elif len(raw) < 2:
            continue
        out.append(raw)
    return out


# Токены из названия и винодельни — то, что напечатано на лицевой этикетке.
# Хвост slug и сорт (категория, сахар, крепость) печатают не всегда.
SECONDARY_WEIGHT = 0.5


def wine_tokens(wine: Wine) -> dict[str, float]:
    """Токен -> вес «опознавательности» внутри карточки (1.0 или SECONDARY_WEIGHT)."""
    weights: dict[str, float] = {}
    slug = _DEDUP_SUFFIX.sub("", wine.slug)  # «-1» у близнеца на этикетке не печатают
    for token in tokenize(slug.replace("-", " ").replace("_", " ")) + tokenize(wine.grape):
        weights[token] = SECONDARY_WEIGHT
    for token in tokenize(wine.title) + tokenize(wine.winery):
        weights[token] = 1.0
    return weights


def _similarity(ratio: float) -> float:
    """rapidfuzz ratio 0..100 -> вклад совпадения: точное 1.0, «мускат»≈«мускатель» ~0.5."""
    return (ratio / 100.0) ** 4


class TextMatcher:
    """Строится один раз по каталогу; score() — на каждый запрос.

    Скор — F-мера между двумя долями, обе взвешены IDF:
    - полнота: какая часть «опознавательных» слов карточки есть на этикетке;
    - точность: какая часть узнаваемых слов этикетки объясняется карточкой.
    Одна полнота награждает короткие названия («Массандра Мускат» обходит
    «Мускатель белый»), одна точность — длинные. beta < 1 сдвигает в сторону
    точности: на этикетке обычно напечатано больше, чем в названии карточки.
    """

    def __init__(self, wines: Iterable[Wine], min_ratio: float = 84.0, beta: float = 0.7
                 ) -> None:
        self.min_ratio = min_ratio
        self.beta2 = beta * beta
        self.docs: dict[str, dict[str, float]] = {w.slug: wine_tokens(w) for w in wines}
        df = Counter(tok for toks in self.docs.values() for tok in toks)
        n = max(len(self.docs), 1)
        self.idf = {tok: math.log((n + 1) / (cnt + 0.5)) for tok, cnt in df.items()}
        self.vocab = list(self.idf)
        self.postings: dict[str, list[str]] = {}
        for slug, toks in self.docs.items():
            for tok in toks:
                self.postings.setdefault(tok, []).append(slug)
        self._norm = {
            slug: sum(self.idf[t] * w for t, w in toks.items()) or 1.0
            for slug, toks in self.docs.items()
        }

    def _query_matches(self, query_tokens: set[str]) -> dict[str, dict[str, float]]:
        """Токен этикетки -> {словарный токен каталога: сходство 0..1} (с опечатками OCR)."""
        out: dict[str, dict[str, float]] = {}
        for token in query_tokens:
            found: dict[str, float] = {}
            if token in self.idf:
                found[token] = 1.0
            if len(token) >= 4:  # короткие токены — только точное совпадение
                for hit, ratio, _ in process.extract(
                    token, self.vocab, scorer=fuzz.ratio, score_cutoff=self.min_ratio, limit=8
                ):
                    if abs(len(hit) - len(token)) <= 3:
                        found[hit] = max(found.get(hit, 0.0), _similarity(ratio))
            if found:
                out[token] = found
        return out

    def _fit(self, word: str) -> float:
        tokens = tokenize(word)
        if not tokens:
            return 0.0
        total = 0.0
        for token in tokens:
            if token in self.idf:
                total += 100.0
            elif len(token) >= 4:
                hit = process.extractOne(token, self.vocab, scorer=fuzz.ratio,
                                         score_cutoff=self.min_ratio)
                total += hit[1] if hit else 0.0
        return total / len(tokens)

    def _pick_reading(self, word: str) -> str:
        alt = _as_cyrillic(word)
        if alt is None:
            return word
        if _GREEK.search(word) or _CYRILLIC.search(word):
            return alt  # греческих слов и смеси алфавитов на наших этикетках не бывает
        return alt if self._fit(alt) > self._fit(word) else word

    def query_tokens(self, lines: list[TextLine]) -> dict[str, float]:
        """Токены этикетки -> вес по размеру шрифта (0.35 для мелкого .. 1.0 для крупного).

        Добавляются и склейки соседних токенов строки: OCR рвёт слова на
        декоративных шрифтах («Па зорг» вместо «Пазори»).
        """
        tallest = max((line.height for line in lines), default=0.0)
        out: dict[str, float] = {}
        for line in lines:
            size = 0.35 + 0.65 * math.sqrt(line.height / tallest) if tallest > 0 else 1.0
            text = " ".join(self._pick_reading(w) for w in _unspace(line.text).split())
            raw = _TOKEN.findall(loose(text))
            glued = ["".join(raw[i : i + 2]) for i in range(len(raw) - 1)
                     if min(len(raw[i]), len(raw[i + 1])) <= 3]
            for token in tokenize(text) + [g for g in glued if len(g) >= 5]:
                out[token] = max(out.get(token, 0.0), size)
        return out

    def score(self, lines: list[TextLine], slugs: Iterable[str] | None = None) -> dict[str, float]:
        """-> {slug: F-мера совпадения этикетки и карточки}, 0..1."""
        if not lines:
            return {}
        sizes = self.query_tokens(lines)
        matches = self._query_matches(set(sizes))
        if not matches:
            return {}

        # Вес токена этикетки — IDF его лучшего словарного соответствия × размер шрифта.
        query_weight = {
            q: max(self.idf[v] * s for v, s in found.items()) * sizes[q]
            for q, found in matches.items()
        }
        query_total = sum(query_weight.values()) or 1.0

        if slugs is None:
            touched = {slug for found in matches.values() for v in found
                       for slug in self.postings.get(v, ())}
        else:
            touched = set(slugs)

        scores: dict[str, float] = {}
        evidence: dict[str, frozenset[str]] = {}
        for slug in touched:
            doc = self.docs.get(slug)
            if not doc:
                continue
            recall_mass = 0.0
            for token, weight in doc.items():
                best = max((found.get(token, 0.0) * sizes[q] for q, found in matches.items()),
                           default=0.0)
                recall_mass += self.idf[token] * weight * best
            precision_mass = 0.0
            explained = []
            for q, found in matches.items():
                best = max((s for v, s in found.items() if v in doc), default=0.0)
                precision_mass += query_weight[q] * best
                if best > 0:
                    explained.append(q)
            recall = recall_mass / self._norm[slug]
            precision = precision_mass / query_total
            if recall <= 0 or precision <= 0:
                continue
            scores[slug] = (1 + self.beta2) * precision * recall / (
                self.beta2 * precision + recall
            )
            evidence[slug] = frozenset(explained)
        # Карточки, которым этикетка дала одни и те же слова, текст не различает:
        # по «GOLUBITSKOE ESTATE» все вина винодельни равны, а F-мера иначе
        # отдала бы победу самому короткому названию. Выбор между ними — за картинкой.
        best_by_evidence: dict[frozenset[str], float] = {}
        for slug, value in scores.items():
            key = evidence[slug]
            best_by_evidence[key] = max(best_by_evidence.get(key, 0.0), value)
        return {slug: best_by_evidence[evidence[slug]] for slug in scores}

    def name_recall(self, lines: list[TextLine], wine: Wine) -> tuple[float, float]:
        """-> (доля собственного названия, прочитанная на этикетке; IDF его редчайшего слова).

        Собственное название — слова title без винодельни: «Олег», «Цитронный».
        Взвешено IDF и размером шрифта, как полнота в score(). Второе значение
        отличает опознающее название от общего: «Олег» 5.1, «Шардоне Резерв» 3.2,
        «Мерло» 2.4 — сумма по словам здесь не годится, общие слова набирают её числом.
        """
        winery = tokenize(wine.winery)
        # «Инкерман» в названии при винодельне «Инкерманский ЗМВ» — тоже винодельня.
        # Короткие обломки («ch» из «Château») по префиксу не сравниваются.
        name = [
            t for t in dict.fromkeys(tokenize(wine.title))
            if t in self.idf and t not in winery
            and not any(min(len(t), len(w)) >= 4 and (t.startswith(w) or w.startswith(t))
                        for w in winery)
        ]
        if not lines or not name:
            return 0.0, max((self.idf[t] for t in name), default=0.0)
        mass = sum(self.idf[t] for t in name)
        sizes = self.query_tokens(lines)
        matches = self._query_matches(set(sizes))
        found = sum(
            self.idf[t] * max((f.get(t, 0.0) * sizes[q] for q, f in matches.items()), default=0.0)
            for t in name
        )
        return found / mass, max(self.idf[t] for t in name)
