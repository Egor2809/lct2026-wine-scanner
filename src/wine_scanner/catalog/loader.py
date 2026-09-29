"""Загрузка дампа каталога «Своё вино».

Дамп выдан в CSV, будет перевыгружен после чистки, и имена колонок заранее
неизвестны. Поэтому загрузчик не знает ни одного имени колонки в коде: он
берёт синонимы из config.yaml (catalog.columns) и сопоставляет их с шапкой
файла. Новый дамп с другими заголовками = правка YAML, не кода.
"""

from __future__ import annotations

import csv
import json
import logging
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any
from urllib.parse import quote

from ..settings import Config, Settings, get_config, get_settings
from .models import Wine, slug_stem

log = logging.getLogger(__name__)

# Маршрут, по которому сервис отдаёт эталонные фотографии каталога.
# Держится здесь, потому что ссылку формирует загрузчик, а монтирует api/app.py.
CATALOG_IMAGES_ROUTE = "/catalog-images"
CATALOG_THUMBS_ROUTE = "/catalog-thumbs"

_SPLIT_LIST = re.compile(r"[;,|]|\s/\s")
_YEAR = re.compile(r"(?:19|20)\d{2}")


def _norm_header(name: str) -> str:
    return re.sub(r"[^a-z0-9а-яё]+", "", (name or "").strip().lower())


def _build_mapping(headers: Iterable[str], synonyms: dict[str, list[str]]) -> dict[str, str]:
    """Целевое поле -> реальное имя колонки в файле."""
    normalized = {_norm_header(h): h for h in headers}
    mapping: dict[str, str] = {}
    for field, options in synonyms.items():
        for option in options:
            actual = normalized.get(_norm_header(option))
            if actual is not None:
                mapping[field] = actual
                break
    return mapping


def _as_list(value: str) -> list[str]:
    if not value:
        return []
    value = value.strip()
    if value.startswith("["):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return [str(x).strip() for x in parsed if str(x).strip()]
        except json.JSONDecodeError:
            pass
    return [part.strip() for part in _SPLIT_LIST.split(value) if part.strip()]


def _sniff_dialect(path: Path) -> csv.Dialect | type[csv.Dialect]:
    """Дампы из Strapi/Postgres приезжают то с запятой, то с точкой с запятой.

    От Sniffer берём только разделитель: экранирование кавычек он угадывает
    неверно на описаниях с `""` внутри, и строки молча разъезжаются
    (на выданном дампе 2103 позиции превращались в 2108).
    """
    with path.open(encoding="utf-8-sig", newline="") as fh:
        sample = fh.read(8192)
    try:
        delimiter = csv.Sniffer().sniff(sample, delimiters=",;\t").delimiter
    except csv.Error:
        return csv.excel

    class Dialect(csv.excel):
        pass

    Dialect.delimiter = delimiter
    return Dialect


class Catalog:
    """Каталог в памяти. 1500–2000 позиций — грузится целиком, без БД."""

    def __init__(self, wines: list[Wine]) -> None:
        self.wines = wines
        self._by_slug = {w.slug: w for w in wines}
        self._by_stem: dict[str, list[Wine]] = {}
        for wine in wines:
            self._by_stem.setdefault(wine.stem, []).append(wine)

    def __len__(self) -> int:
        return len(self.wines)

    def __iter__(self):
        return iter(self.wines)

    def get(self, slug: str) -> Wine | None:
        wine = self._by_slug.get(slug)
        if wine is not None:
            return wine
        # Нечёткое попадание: тот же стем, другой винтаж.
        candidates = self._by_stem.get(slug_stem(slug))
        return candidates[0] if candidates else None

    def siblings(self, wine: Wine) -> list[Wine]:
        """Другие винтажи того же вина — главный источник near-duplicates."""
        return [w for w in self._by_stem.get(wine.stem, []) if w.slug != wine.slug]


def load_catalog(
    path: Path | None = None,
    settings: Settings | None = None,
    config: Config | None = None,
) -> Catalog:
    settings = settings or get_settings()
    config = config or get_config()
    path = path or settings.resolve(settings.catalog_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Дамп каталога не найден: {path}. Положите CSV в data/catalog.csv "
            f"или задайте CATALOG_PATH в .env (см. .env.example)."
        )

    synonyms: dict[str, list[str]] = config.get_path("catalog.columns", {})
    site_base = config.get_path("catalog.site_base_url", "")
    images_dir = settings.resolve(settings.catalog_images_dir)
    thumbs_dir = settings.resolve(settings.catalog_thumbs_dir)

    dialect = _sniff_dialect(path)
    wines: list[Wine] = []
    skipped = 0

    with path.open(encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh, dialect=dialect)
        mapping = _build_mapping(reader.fieldnames or [], synonyms)

        if "slug" not in mapping:
            raise ValueError(
                f"В дампе не найдена колонка slug. Заголовки: {reader.fieldnames}. "
                f"Допишите синоним в config/config.yaml -> catalog.columns.slug"
            )
        missing = sorted(set(synonyms) - set(mapping))
        if missing:
            log.warning("Колонки дампа не распознаны, поля останутся пустыми: %s", missing)

        for row in reader:
            wine = _row_to_wine(row, mapping, images_dir, thumbs_dir, site_base)
            if wine is None:
                skipped += 1
                continue
            wines.append(wine)

    if skipped:
        log.warning("Пропущено строк без slug: %d", skipped)
    if not wines:
        raise ValueError(f"Дамп {path} не содержит ни одной валидной позиции")

    log.info("Каталог загружен: %d позиций из %s", len(wines), path)
    return Catalog(wines)


def _row_to_wine(
    row: dict[str, Any],
    mapping: dict[str, str],
    images_dir: Path,
    thumbs_dir: Path,
    site_base: str,
) -> Wine | None:
    def field(name: str) -> str:
        column = mapping.get(name)
        if column is None:
            return ""
        return (row.get(column) or "").strip()

    slug = field("slug")
    if not slug:
        return None

    vintage = field("vintage")
    if not vintage:
        # В части дампов года нет отдельной колонкой — он только в slug/названии.
        found = _YEAR.search(slug) or _YEAR.search(field("title"))
        vintage = found.group(0) if found else ""

    image = field("image")
    image_url = image
    if image and not image.startswith(("http://", "https://")):
        # Отдаётся сервисом (см. api/app.py): превью, иначе исходник. Имя файла
        # без префикса браузер запросил бы от корня и получил 404.
        name = quote(image)
        local = images_dir / image
        if (thumbs_dir / image).exists():
            image_url = f"{CATALOG_THUMBS_ROUTE}/{name}"
        elif local.exists():
            image_url = f"{CATALOG_IMAGES_ROUTE}/{name}"
        else:
            image_url = ""
        if local.exists():
            image = str(local)

    url = field("url") or (f"{site_base.rstrip('/')}/{slug}" if site_base else "")

    return Wine(
        slug=slug,
        title=field("title"),
        winery=field("winery"),
        region=field("region"),
        grape=field("grape"),
        style=field("style"),
        color=field("color"),
        vintage=vintage,
        sugar=field("sugar"),
        description=field("description"),
        pairing=_as_list(field("pairing")),
        image=image,
        image_url=image_url,
        url=url,
        raw=dict(row),
    )
