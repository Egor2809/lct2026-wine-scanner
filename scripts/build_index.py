#!/usr/bin/env python3
"""Сборка векторного индекса по эталонным фотографиям каталога.

Заказчик обещал перевыгрузку дампа, поэтому пересборка — две команды:

    python scripts/prepare_catalog.py   # дамп -> data/catalog.csv с путями к фото
    python scripts/build_index.py       # data/catalog.csv -> artifacts/index.npz

Эталоны кодируются с role="catalog": базовой моделью, без LoRA-адаптера и
без кропа — это студийные фото одной бутылки. Ровно так же каталог кодирует
стенд оценки (scripts/eval_offline.py), иначе цифры стенда ничего не значат.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import httpx
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wine_scanner.catalog.loader import load_catalog  # noqa: E402
from wine_scanner.catalog.models import Wine  # noqa: E402
from wine_scanner.features.embedder import build_embedder  # noqa: E402
from wine_scanner.normalize.image import decode  # noqa: E402
from wine_scanner.search.index import build_index  # noqa: E402
from wine_scanner.settings import get_config, get_settings  # noqa: E402

log = logging.getLogger("build_index")

BATCH = 32


def fetch_image(wine: Wine, client: httpx.Client, cache_dir: Path) -> Image.Image | None:
    """Локальный файл или URL. Скачанное кэшируется: дамп индексируется не раз."""
    if not wine.image:
        return None

    local = Path(wine.image)
    if local.exists():
        return decode(local.read_bytes())

    if not wine.image.startswith(("http://", "https://")):
        return None

    cached = cache_dir / f"{wine.slug}.bin"
    if cached.exists():
        return decode(cached.read_bytes())

    try:
        response = client.get(wine.image, timeout=20.0, follow_redirects=True)
        response.raise_for_status()
    except Exception as exc:
        log.warning("%s: не удалось скачать %s (%s)", wine.slug, wine.image, exc)
        return None

    cache_dir.mkdir(parents=True, exist_ok=True)
    cached.write_bytes(response.content)
    return decode(response.content)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=None, help="путь к data/catalog.csv")
    parser.add_argument("--out", type=Path, default=None, help="куда писать индекс")
    parser.add_argument("--limit", type=int, default=0, help="обработать только N позиций")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    settings = get_settings()
    config = get_config()

    catalog = load_catalog(path=args.catalog, settings=settings, config=config)
    embedder = build_embedder(config.get("embedder") or {})

    wines = list(catalog)[: args.limit] if args.limit else list(catalog)
    cache_dir = settings.resolve(settings.artifacts_dir) / "image_cache"

    vectors: list[np.ndarray] = []
    slugs: list[str] = []
    missing: list[str] = []
    batch_images: list[Image.Image] = []
    batch_slugs: list[str] = []

    def flush() -> None:
        if batch_images:
            vectors.append(embedder.encode(batch_images, role="catalog"))
            slugs.extend(batch_slugs)
            batch_images.clear()
            batch_slugs.clear()

    started = time.perf_counter()
    with httpx.Client(headers={"User-Agent": "wine-scanner/0.1"}) as client:
        for position, wine in enumerate(wines, start=1):
            try:
                image = fetch_image(wine, client, cache_dir)
            except Exception as exc:  # битый файл не должен ронять всю сборку
                log.warning("%s: не открылось фото (%s)", wine.slug, exc)
                image = None
            if image is None:
                missing.append(wine.slug)
                continue
            batch_images.append(image)
            batch_slugs.append(wine.slug)
            if len(batch_images) == BATCH:
                flush()
            if position % 200 == 0:
                log.info("Обработано %d/%d", position, len(wines))
        flush()

    if not vectors:
        log.error(
            "Ни одной эталонной фотографии не получено. Проверьте колонку image "
            "в дампе и CATALOG_IMAGES_DIR."
        )
        return 1

    index = build_index(
        kind=config.get_path("index.kind", "numpy"),
        dim=embedder.dim,
        meta={
            "embedder": embedder.name,
            "dim": embedder.dim,
            "catalog_size": len(catalog),
            "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
    )
    index.add(np.concatenate(vectors), slugs)

    out = args.out or settings.index_path
    index.save(out)

    log.info(
        "Готово за %.1f c: %d позиций в индексе, %d без фотографии",
        time.perf_counter() - started,
        len(index),
        len(missing),
    )
    if missing:
        report = settings.resolve(settings.artifacts_dir) / "missing_images.txt"
        report.write_text("\n".join(missing), encoding="utf-8")
        log.warning(
            "Позиции без фотографии не найдутся по картинке — только по тексту этикетки. "
            "Список: %s",
            report,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
