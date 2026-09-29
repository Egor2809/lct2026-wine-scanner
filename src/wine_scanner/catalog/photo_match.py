"""Сопоставление «имя фото из дампа» -> файл на диске.

В дампе лежит исходное имя, которое загрузил клиент («Агора_Мускат.webp»),
а на диске — имя после Strapi: транслит, пробелы и дефисы в `_`, хвост
`_<hash>` перед расширением («Agora_Muskat_4030bf4725.webp»). Прямого
соответствия нет, поэтому обе стороны приводятся к одному ключу:
транслит + нижний регистр + только буквы и цифры, без хэша и без мусорных
суффиксов фоторедакторов. Сначала точное совпадение ключей, потом нечёткое
с высоким порогом — промах по фото хуже пропуска: вино с чужой картинкой
в индексе не найдётся никогда.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from rapidfuzz import fuzz, process

IMAGE_EXTS = {".webp", ".jpg", ".jpeg", ".png", ".jfif", ".gif", ".avif"}

_CYR2LAT = {
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "e",
    "ж": "zh", "з": "z", "и": "i", "й": "j", "к": "k", "л": "l", "м": "m",
    "н": "n", "о": "o", "п": "p", "р": "r", "с": "s", "т": "t", "у": "u",
    "ф": "f", "х": "h", "ц": "c", "ч": "ch", "ш": "sh", "щ": "sch",
    "ъ": "", "ы": "y", "ь": "", "э": "e", "ю": "yu", "я": "ya",
}
# Хвосты, которые добавляют фоторедакторы и «копии» файлов. Разделители
# в дампе — пробелы и дефисы, на диске — подчёркивания, поэтому [-_ ]?.
_JUNK = re.compile(
    # «без фона» в дампе на диске становится транслитом bez_fona — убираем оба.
    r"photoroom|no[-_ ]?bg[-_ ]?preview|carve[._ ]?photos|ezremove|preview_0|без фона|bez[-_ ]?fona"
    r"|—\s*копия|копия|kopiya|\(\d{1,2}\)|\bcopy\b|\bcopie\b",
    re.IGNORECASE,
)
_STRAPI_HASH = re.compile(r"_[0-9a-f]{10}$")
# Strapi кладёт рядом уменьшенные копии того же файла — это не отдельные фото.
_STRAPI_FORMAT = re.compile(r"^(large|medium|small|thumbnail)_")
# Strapi по-разному транслитерирует й/ы/щ; для ключа сводим спорные пары.
_LOOSE = [("sch", "sh"), ("yj", "y"), ("ij", "i"), ("j", "i"), ("yy", "y")]


def _translit(text: str) -> str:
    return "".join(_CYR2LAT.get(ch, ch) for ch in text.lower())


def match_key(filename: str, *, strapi: bool = False) -> str:
    stem = Path(filename).stem if Path(filename).suffix.lower() in IMAGE_EXTS else filename
    if strapi:
        stem = _STRAPI_HASH.sub("", stem)
    stem = _JUNK.sub(" ", stem)
    key = re.sub(r"[^a-z0-9]+", "", _translit(stem))
    for src, dst in _LOOSE:
        key = key.replace(src, dst)
    return key


@dataclass(slots=True)
class PhotoMatch:
    # Все файлы, подходящие по ключу. Больше одного — когда клиенты грузили
    # разные фото под одинаковым именем («Шардоне.webp»): выбирать между ними
    # должен вызывающий код по содержимому, а не по имени.
    files: list[str]
    how: str


def match_photos(
    wanted: dict[str, str],
    disk_files: Iterable[str],
    min_score: float = 90.0,
) -> dict[str, PhotoMatch]:
    """{slug: имя фото из дампа} -> {slug: PhotoMatch}.

    Слаги без уверенного совпадения в результат не попадают — их список
    печатает prepare_catalog.py, чтобы решение принимал человек.
    """
    files = [
        f
        for f in disk_files
        if Path(f).suffix.lower() in IMAGE_EXTS and not _STRAPI_FORMAT.match(f)
    ]
    exact_name = {f.lower(): f for f in files}
    by_key: dict[str, list[str]] = {}
    for f in sorted(files):
        by_key.setdefault(match_key(f, strapi=True), []).append(f)
    keys = list(by_key)

    result: dict[str, PhotoMatch] = {}
    for slug, photo in wanted.items():
        if not photo:
            continue
        if photo.lower() in exact_name:
            result[slug] = PhotoMatch([exact_name[photo.lower()]], "exact")
            continue
        key = match_key(photo)
        if key in by_key:
            result[slug] = PhotoMatch(list(by_key[key]), "key")
            continue
        if len(key) < 6:  # на коротких ключах fuzzy угадывает, а не сопоставляет
            continue
        hit = process.extractOne(key, keys, scorer=fuzz.ratio, score_cutoff=min_score)
        if hit is not None:
            result[slug] = PhotoMatch(list(by_key[hit[0]]), f"fuzzy:{hit[1]:.0f}")
    return result
