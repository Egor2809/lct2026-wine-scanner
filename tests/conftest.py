"""Фикстуры: каталог и индекс собираются в памяти, без сети и без artifacts/."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wine_scanner.catalog.loader import load_catalog  # noqa: E402
from wine_scanner.features.embedder import build_embedder  # noqa: E402
from wine_scanner.normalize.image import normalize  # noqa: E402
from wine_scanner.pipeline import ScannerPipeline  # noqa: E402
from wine_scanner.search.index import build_index  # noqa: E402
from wine_scanner.settings import Config, Settings  # noqa: E402

CATALOG_ROWS = [
    # slug, name, producer, region, grape, category, color, year
    ("vedernikov-krasnostop-2021", "Красностоп 2021", "Ведерниковъ", "Долина Дона", "Красностоп", "Красное сухое", "red", "2021"),
    ("vedernikov-krasnostop-2022", "Красностоп 2022", "Ведерниковъ", "Долина Дона", "Красностоп", "Красное сухое", "red", "2022"),
    ("vedernikov-sibirkovy-2022", "Сибирьковый 2022", "Ведерниковъ", "Долина Дона", "Сибирьковый", "Белое сухое", "white", "2022"),
    ("sikory-riesling-2023", "Рислинг 2023", "Сикоры", "Семигорье", "Рислинг", "Белое сухое", "white", "2023"),
    ("uppa-winery-pinot-noir-2021", "Пино Нуар 2021", "UPPA Winery", "Севастополь", "Пино нуар", "Красное сухое", "red", "2021"),
    ("galitsky-galitsky-kosogor-2022", "Косогор 2022", "Галицкий и Галицкий", "Крымск", "Красностоп", "Красное сухое", "red", "2022"),
]

COLORS = [(120, 30, 40), (124, 34, 44), (230, 220, 180), (200, 210, 160), (150, 60, 70), (110, 40, 50)]


@pytest.fixture(scope="session")
def catalog_csv(tmp_path_factory) -> Path:
    data = tmp_path_factory.mktemp("data")
    images = data / "catalog_images"
    images.mkdir()

    rows = []
    for (slug, name, producer, region, grape, category, color, year), rgb in zip(CATALOG_ROWS, COLORS, strict=False):
        Image.new("RGB", (200, 300), rgb).save(images / f"{slug}.png")
        rows.append(
            {
                "slug": slug, "name": name, "producer": producer, "region": region,
                "grape": grape, "category": category, "color": color, "year": year,
                "description": f"{name} от «{producer}».", "food_pairing": "Мясо; Сыры",
                "image_url": f"{slug}.png", "url": f"https://vino-svoe.ru/wines/{slug}",
            }
        )

    path = data / "catalog.csv"
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


@pytest.fixture(scope="session")
def config() -> Config:
    import yaml

    with (ROOT / "config" / "config.yaml").open(encoding="utf-8") as fh:
        config = Config(yaml.safe_load(fh))
    # Контрактные тесты гоняются на гистограммном эмбеддере: синтетические
    # однотонные «этикетки» SigLIP не различает, а тестам нужны скорость,
    # детерминизм и отсутствие скачивания весов. Качество модели меряет
    # scripts/eval_offline.py на настоящих полевых фото.
    config["embedder"]["kind"] = "histogram"
    config["normalize"]["crop"]["enabled"] = False
    config["normalize"]["white_balance"] = False
    return config


@pytest.fixture(scope="session")
def pipeline(catalog_csv: Path, config: Config) -> ScannerPipeline:
    settings = Settings(
        catalog_path=catalog_csv,
        catalog_images_dir=catalog_csv.parent / "catalog_images",
    )
    catalog = load_catalog(path=catalog_csv, settings=settings, config=config)
    embedder = build_embedder(config.get("embedder"))
    norm_cfg = config.get("normalize")

    index = build_index(config.get_path("index.kind"), embedder.dim, {"embedder": embedder.name})
    vectors, slugs = [], []
    for wine in catalog:
        image = Image.open(wine.image)
        vectors.append(embedder.encode([normalize(image, norm_cfg)])[0])
        slugs.append(wine.slug)
    index.add(np.stack(vectors), slugs)

    return ScannerPipeline(catalog, embedder, index, config)


@pytest.fixture()
def client(pipeline: ScannerPipeline) -> TestClient:
    from wine_scanner.api.app import app

    app.state.pipeline = pipeline
    app.state.startup_error = None
    # Без lifespan: пайплайн уже собран фикстурой, повторная загрузка не нужна.
    return TestClient(app)
