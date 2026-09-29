"""Тесты загрузчика дампа, нормализации и деградации сигналов."""

from __future__ import annotations

import csv
import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from wine_scanner.catalog.loader import Catalog, load_catalog
from wine_scanner.catalog.models import Wine
from wine_scanner.enrich.analogs import find_analogs
from wine_scanner.features.embedder import HistogramEmbedder, l2_normalize
from wine_scanner.normalize.image import ImageDecodeError, decode, normalize
from wine_scanner.pipeline import assert_index_matches
from wine_scanner.search.confidence import estimate
from wine_scanner.search.index import NumpyIndex
from wine_scanner.search.rerank.fusion import Candidate, fuse
from wine_scanner.settings import Settings

# ── загрузчик дампа ───────────────────────────────────────────────────────


def _write_csv(path: Path, headers: list[str], rows: list[list[str]], delimiter: str = ",") -> Path:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, delimiter=delimiter)
        writer.writerow(headers)
        writer.writerows(rows)
    return path


def test_loader_maps_unknown_column_names(tmp_path: Path, config) -> None:
    """Дамп будет перевыгружен, имена колонок могут отличаться — маппинг
    идёт по синонимам из config.yaml, а не по хардкоду."""
    path = _write_csv(
        tmp_path / "dump.csv",
        ["Наименование", "Производитель", "wine_slug", "Регион"],
        [["Рислинг", "Сикоры", "sikory-riesling-2023", "Семигорье"]],
    )
    catalog = load_catalog(path=path, settings=Settings(), config=config)
    wine = catalog.get("sikory-riesling-2023")
    assert wine.title == "Рислинг"
    assert wine.winery == "Сикоры"
    assert wine.region == "Семигорье"


def test_loader_handles_semicolon_dialect(tmp_path: Path, config) -> None:
    path = _write_csv(
        tmp_path / "dump.csv",
        ["slug", "name", "producer"],
        [["a-b-2021", "Вино", "Хозяйство"]],
        delimiter=";",
    )
    assert load_catalog(path=path, settings=Settings(), config=config).get("a-b-2021").title == "Вино"


def test_loader_infers_vintage_from_slug(tmp_path: Path, config) -> None:
    path = _write_csv(tmp_path / "dump.csv", ["slug", "name"], [["sikory-riesling-2023", "Рислинг"]])
    assert load_catalog(path=path, settings=Settings(), config=config).get("sikory-riesling-2023").vintage == "2023"


def test_loader_rejects_dump_without_slug(tmp_path: Path, config) -> None:
    path = _write_csv(tmp_path / "dump.csv", ["name", "producer"], [["Вино", "Хозяйство"]])
    with pytest.raises(ValueError, match="slug"):
        load_catalog(path=path, settings=Settings(), config=config)


def test_loader_skips_rows_without_slug(tmp_path: Path, config) -> None:
    path = _write_csv(tmp_path / "dump.csv", ["slug", "name"], [["", "Пусто"], ["a-b-2021", "Вино"]])
    assert len(load_catalog(path=path, settings=Settings(), config=config)) == 1


def test_loader_parses_pairing_list(tmp_path: Path, config) -> None:
    path = _write_csv(
        tmp_path / "dump.csv",
        ["slug", "food_pairing"],
        [["a-b-2021", "Баранина; Выдержанные сыры"]],
    )
    wine = load_catalog(path=path, settings=Settings(), config=config).get("a-b-2021")
    assert wine.pairing == ["Баранина", "Выдержанные сыры"]


def test_card_image_is_http_path_not_filesystem_path(tmp_path: Path, config) -> None:
    """В карточку обязан уходить HTTP-путь.

    Абсолютный путь файловой системы браузер запрашивает от корня сервера
    и получает 404 — в интерфейсе это битые картинки во всей выдаче.
    """
    images = tmp_path / "catalog_images"
    images.mkdir()
    (images / "a-b-2021.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    path = _write_csv(tmp_path / "dump.csv", ["slug", "image_url"], [["a-b-2021", "a-b-2021.png"]])

    settings = Settings(catalog_images_dir=images)
    wine = load_catalog(path=path, settings=settings, config=config).get("a-b-2021")

    assert wine.card()["image"] == "/catalog-images/a-b-2021.png"
    # А индексатору по-прежнему нужен реальный файл на диске.
    assert Path(wine.image).exists()


def test_card_image_prefers_thumbnail(tmp_path: Path, config) -> None:
    images, thumbs = tmp_path / "photos", tmp_path / "thumbs"
    images.mkdir()
    thumbs.mkdir()
    (images / "a.webp").write_bytes(b"RIFF")
    (thumbs / "a.webp").write_bytes(b"RIFF")
    path = _write_csv(tmp_path / "dump.csv", ["slug", "image_url"], [["a-b-2021", "a.webp"]])

    settings = Settings(catalog_images_dir=images, catalog_thumbs_dir=thumbs)
    wine = load_catalog(path=path, settings=settings, config=config).get("a-b-2021")

    assert wine.card()["image"] == "/catalog-thumbs/a.webp"
    assert Path(wine.image) == images / "a.webp"


def test_card_image_empty_when_file_missing(tmp_path: Path, config) -> None:
    """Голое имя файла браузер запросит от корня и получит 404: пусто -> силуэт в UI."""
    path = _write_csv(tmp_path / "dump.csv", ["slug", "image_url"], [["a-b-2021", "a.webp"]])
    settings = Settings(catalog_images_dir=tmp_path / "no", catalog_thumbs_dir=tmp_path / "no")
    wine = load_catalog(path=path, settings=settings, config=config).get("a-b-2021")
    assert wine.card()["image"] == ""


def test_card_image_passes_through_external_url(tmp_path: Path, config) -> None:
    path = _write_csv(
        tmp_path / "dump.csv",
        ["slug", "image_url"],
        [["a-b-2021", "https://vino-svoe.ru/photo.jpg"]],
    )
    wine = load_catalog(path=path, settings=Settings(), config=config).get("a-b-2021")
    assert wine.card()["image"] == "https://vino-svoe.ru/photo.jpg"


def test_catalog_get_falls_back_to_other_vintage(config) -> None:
    catalog = Catalog([Wine(slug="sikory-riesling-2021"), Wine(slug="sikory-riesling-2023")])
    assert catalog.get("sikory-riesling-2022") is not None


# ── нормализация ──────────────────────────────────────────────────────────


def test_decode_rejects_garbage() -> None:
    with pytest.raises(ImageDecodeError):
        decode(b"definitely not an image")


def test_decode_applies_exif_orientation() -> None:
    """Без exif_transpose часть фото уходит в модель повёрнутой, и точность
    падает молча — ошибок в логах при этом не будет."""
    buffer = io.BytesIO()
    image = Image.new("RGB", (100, 50), (200, 100, 50))
    exif = image.getexif()
    exif[274] = 6  # ориентация «повернуть на 90°»
    image.save(buffer, format="JPEG", exif=exif)

    assert decode(buffer.getvalue()).size == (50, 100)


def test_normalize_is_deterministic_and_bounded(config) -> None:
    """Индексация и инференс обязаны звать одну и ту же функцию с одним
    результатом — иначе ищем не в том пространстве, где индексировали."""
    image = Image.new("RGB", (3000, 2000), (180, 60, 60))
    first = normalize(image, config.get("normalize"))
    second = normalize(image, config.get("normalize"))

    assert max(first.size) <= config.get_path("normalize.max_side")
    assert np.array_equal(np.asarray(first), np.asarray(second))


# ── эмбеддер и индекс ─────────────────────────────────────────────────────


def test_embeddings_are_l2_normalized() -> None:
    embedder = HistogramEmbedder(bins=4, grid=2)
    vectors = embedder.encode([Image.new("RGB", (64, 64), (10, 200, 30))] * 3)
    assert vectors.shape == (3, embedder.dim)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)


def test_l2_normalize_survives_zero_vector() -> None:
    assert not np.isnan(l2_normalize(np.zeros((1, 8)))).any()


def test_index_search_ranks_exact_match_first() -> None:
    index = NumpyIndex(dim=4)
    index.add(np.eye(4, dtype=np.float32), ["a", "b", "c", "d"])
    hits = index.search(np.array([[0.0, 1.0, 0.0, 0.0]], dtype=np.float32), k=3)

    assert hits[0][0] == "b"
    assert len(hits) == 3
    assert [score for _, score in hits] == sorted([s for _, s in hits], reverse=True)


def test_index_roundtrip(tmp_path: Path) -> None:
    index = NumpyIndex(dim=4, meta={"embedder": "histogram"})
    index.add(np.eye(4, dtype=np.float32), ["a", "b", "c", "d"])
    index.save(tmp_path / "index.npz")

    restored = NumpyIndex.load(tmp_path / "index.npz")
    assert restored.slugs == ["a", "b", "c", "d"]
    assert restored.meta["embedder"] == "histogram"


def test_index_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError, match="Рассинхрон"):
        NumpyIndex(dim=4).add(np.eye(4, dtype=np.float32), ["a", "b"])


def test_index_search_on_empty_returns_nothing() -> None:
    assert NumpyIndex(dim=4).search(np.ones((1, 4), dtype=np.float32), k=5) == []


# ── слияние и уверенность ─────────────────────────────────────────────────


def test_fusion_ignores_signals_with_zero_weight() -> None:
    """Текстовый и геометрический сигналы ещё не подключены: их веса нулевые,
    и мусор в этих полях не должен влиять на порядок."""
    candidates = [Candidate("a", visual=0.9, textual=0.0), Candidate("b", visual=0.5, textual=99.0)]
    assert [c.slug for c in fuse(candidates, {"visual": 1.0, "textual": 0.0})] == ["a", "b"]


def test_fusion_applies_textual_weight_when_enabled() -> None:
    candidates = [Candidate("a", visual=0.9, textual=0.0), Candidate("b", visual=0.5, textual=1.0)]
    assert [c.slug for c in fuse(candidates, {"visual": 1.0, "textual": 1.0})] == ["b", "a"]


def test_confidence_is_ordered_and_bounded() -> None:
    confident = estimate([0.95, 0.40, 0.35], temperature=0.07)
    ambiguous = estimate([0.95, 0.94, 0.93], temperature=0.07)

    assert confident["confidence"] > ambiguous["confidence"]
    assert confident["margin"] > ambiguous["margin"]
    assert 0.0 <= confident["top1_f1"] <= confident["top5_f1"] <= 1.0


def test_confidence_handles_empty_scores() -> None:
    assert estimate([], temperature=0.07)["confidence"] == 0.0


# ── аналоги ───────────────────────────────────────────────────────────────


def test_analogs_exclude_same_wine_and_other_vintages() -> None:
    catalog = Catalog(
        [
            Wine(slug="a-krasnostop-2021", winery="А", grape="Красностоп", color="red", style="Красное сухое"),
            Wine(slug="a-krasnostop-2022", winery="А", grape="Красностоп", color="red", style="Красное сухое"),
            Wine(slug="b-krasnostop-2021", winery="Б", grape="Красностоп", color="red", style="Красное сухое"),
            Wine(slug="c-riesling-2021", winery="В", grape="Рислинг", color="white", style="Белое сухое"),
        ]
    )
    target = catalog.get("a-krasnostop-2021")
    slugs = [w.slug for w in find_analogs(catalog, target, count=4)]

    assert "a-krasnostop-2021" not in slugs
    assert "a-krasnostop-2022" not in slugs, "другой винтаж того же вина — не аналог"
    assert slugs[0] == "b-krasnostop-2021", "ближайший по сорту и цвету от другой винодельни"


def test_analogs_do_not_repeat_winery() -> None:
    catalog = Catalog(
        [Wine(slug="a-x-2021", winery="А", grape="Мерло", color="red")]
        + [Wine(slug=f"b-{i}-2021", winery="Б", grape="Мерло", color="red") for i in range(5)]
    )
    analogs = find_analogs(catalog, catalog.get("a-x-2021"), count=4)
    assert len({w.winery for w in analogs}) == len(analogs)


# ── пайплайн ──────────────────────────────────────────────────────────────


def test_pipeline_finds_exact_reference_photo(pipeline) -> None:
    """Эталон каталога, поданный как запрос, обязан найти сам себя.
    Если этот тест падает — рассинхрон нормализации между индексацией и
    инференсом, а не проблема модели."""
    wine = pipeline.catalog.get("sikory-riesling-2023")
    result = pipeline.scan(Path(wine.image).read_bytes())

    assert result["slug"] == "sikory-riesling-2023"
    assert result["elapsed_ms"] >= 0


def test_startup_rejects_index_built_by_another_embedder(pipeline) -> None:
    """Смена эмбеддера без пересборки индекса — тихая катастрофа точности,
    поэтому сервис обязан падать громко при старте."""
    index = NumpyIndex(dim=4, meta={"embedder": "siglip"})
    index.add(np.eye(4, dtype=np.float32), list("abcd"))

    with pytest.raises(RuntimeError, match="Пересоберите индекс"):
        assert_index_matches(index, pipeline.embedder)


def test_startup_rejects_empty_index(pipeline) -> None:
    with pytest.raises(RuntimeError, match="Индекс пуст"):
        assert_index_matches(NumpyIndex(dim=4), pipeline.embedder)
