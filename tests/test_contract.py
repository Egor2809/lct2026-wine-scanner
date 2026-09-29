"""Тесты контракта ответа.

Главный риск кейса — не низкая точность, а ответ, который не парсится
скриптом кейсодержателя. Это обнуляет 50 баллов из 100 при идеальной
модели, поэтому форма ответа закреплена тестами.
"""

from __future__ import annotations

import io

import pytest
from PIL import Image

from wine_scanner import response
from wine_scanner.catalog.models import Wine, same_wine, slug_stem
from wine_scanner.response import ScanResponse


def png_bytes(color: tuple[int, int, int] = (120, 30, 40)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (240, 360), color).save(buffer, format="PNG")
    return buffer.getvalue()


# ── нечёткое сравнение slug ───────────────────────────────────────────────


@pytest.mark.parametrize(
    "slug,expected",
    [
        ("vedernikov-krasnostop-2021", "vedernikov-krasnostop"),
        ("vedernikov-krasnostop-2021-2", "vedernikov-krasnostop"),
        ("uppa-winery-cuvee-nv", "uppa-winery-cuvee"),
        ("sikory-riesling", "sikory-riesling"),
        # Год в середине slug — не винтажный хвост, трогать нельзя.
        ("winery-2021-special-edition", "winery-2021-special-edition"),
    ],
)
def test_slug_stem_strips_only_trailing_vintage(slug: str, expected: str) -> None:
    assert slug_stem(slug) == expected


def test_same_wine_accepts_other_vintage() -> None:
    """Кейсодержатель: «если мы найдём то вино только другого винтажа —
    это правильное попадание»."""
    assert same_wine("sikory-riesling-2023", "sikory-riesling-2021")
    assert not same_wine("sikory-riesling-2023", "sikory-chardonnay-2023")
    assert not same_wine("", "sikory-riesling-2023")


# ── карточка ──────────────────────────────────────────────────────────────


def test_card_has_keys_frontend_expects() -> None:
    wine = Wine(slug="x-y-2021", title="Рислинг", winery="Сикоры", vintage="2021")
    card = wine.card()
    for key in ("title", "winery", "region", "grape", "style", "vintage", "note", "pairing", "url"):
        assert key in card, f"frontend/app.js читает card.{key}"


# ── HTTP ──────────────────────────────────────────────────────────────────


def test_scan_returns_slug_at_root(client) -> None:
    response = client.post("/scan", files={"image": ("photo.png", png_bytes(), "image/png")})
    assert response.status_code == 200

    body = response.json()
    # Ровно это читает скрипт кейсодержателя. Любая обёртка ломает разбор.
    assert "slug" in body, "ключ 'slug' обязан лежать в корне ответа"
    assert isinstance(body["slug"], str) and body["slug"]
    assert body["slug"] is not None, "сервис обязан всегда возвращать лучший slug"

    for key in ("confidence", "top1_f1", "top5_f1", "elapsed_ms", "candidates", "card", "analogs"):
        assert key in body


def test_scan_never_returns_null_slug_on_noise(client) -> None:
    """Даже на мусорном изображении ответ обязан быть валидным slug:
    в приватном наборе все вина есть в каталоге, null там — всегда ошибка."""
    noise = png_bytes((0, 0, 0))
    body = client.post("/scan", files={"image": ("noise.png", noise, "image/png")}).json()
    assert body["slug"]


def test_scan_accepts_field_with_any_name(client) -> None:
    """Имя поля формы у скрипта кейсодержателя мы не контролируем."""
    response = client.post("/scan", files={"file": ("photo.png", png_bytes(), "image/png")})
    assert response.status_code == 200
    assert response.json()["slug"]


def test_scan_accepts_raw_body(client) -> None:
    response = client.post(
        "/scan", content=png_bytes(), headers={"content-type": "image/png"}
    )
    assert response.status_code == 200
    assert response.json()["slug"]


def test_scan_answers_slug_on_non_image(client) -> None:
    """Не-изображение — не повод для 4xx: скрипт оценки может упасть на нём целиком."""
    response = client.post("/scan", files={"image": ("a.txt", b"not an image", "text/plain")})
    assert response.status_code == 200
    body = response.json()
    assert body["slug"] and body["error"]


def test_candidates_are_sorted_and_capped(client) -> None:
    body = client.post("/scan", files={"image": ("photo.png", png_bytes(), "image/png")}).json()
    scores = [c["score"] for c in body["candidates"]]
    assert scores == sorted(scores, reverse=True)
    assert len(scores) <= 5, "return_k=5 — больше фронтенду не нужно"


def test_health_reports_sizes(client) -> None:
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["catalog_size"] > 0
    assert body["index_size"] > 0


def test_wines_lists_catalog_with_facets(client) -> None:
    body = client.get("/wines", params={"limit": 2}).json()
    assert body["total"] > 2 and len(body["items"]) == 2
    assert all(item["slug"] and item["card"]["title"] for item in body["items"])
    assert sum(s["count"] for s in body["styles"]) == body["total"]


def test_wines_filters_by_style_winery_and_query(client) -> None:
    red = client.get("/wines", params={"style": "Красное сухое"}).json()
    assert red["total"] > 0
    assert all(i["card"]["style"] == "Красное сухое" for i in red["items"])
    assert {s["name"] for s in red["styles"]} >= {"Красное сухое", "Белое сухое"}

    winery = client.get("/wines", params={"winery": "сикоры"}).json()
    assert winery["total"] > 0
    assert all(i["card"]["winery"] == "Сикоры" for i in winery["items"])

    found = client.get("/wines", params={"q": "рислинг сикоры"}).json()
    assert found["total"] > 0
    assert client.get("/wines", params={"q": "несуществующее"}).json()["total"] == 0


def test_wines_rejects_bad_paging(client) -> None:
    assert client.get("/wines", params={"limit": 0}).status_code == 422
    assert client.get("/wines", params={"limit": 1000}).status_code == 422
    assert client.get("/wines", params={"offset": -1}).status_code == 422


def test_browse_sparkling_region_offset_and_seed() -> None:
    from wine_scanner.catalog.browse import browse
    from wine_scanner.catalog.loader import Catalog

    catalog = Catalog([
        Wine(slug="a", title="Brut Zero Dosage", winery="A", region="Крым", style="Белое"),
        Wine(slug="b", title="Пет-Нат Рубин", winery="B", region="Кубань", style="Красное"),
        Wine(slug="c", title="Pinot Noir", winery="C", region="Крым", style="Красное",
             description="Аромат напоминает Шампань."),
        Wine(slug="d", title="Рислинг", winery="D", region="Кубань", style="Белое"),
    ])

    assert {w.slug for w in browse(catalog, sparkling=True).items} == {"a", "b"}
    assert {w.slug for w in browse(catalog, sparkling=False).items} == {"c", "d"}

    crimea = browse(catalog, region="крым")
    assert {w.slug for w in crimea.items} == {"a", "c"}
    assert dict(crimea.regions) == {"Крым": 2, "Кубань": 2}

    assert [w.slug for w in browse(catalog, limit=2, offset=2).items] == ["c", "d"]
    assert browse(catalog, offset=10).items == []

    shuffled = [w.slug for w in browse(catalog, seed=7).items]
    assert shuffled == [w.slug for w in browse(catalog, seed=7).items]
    assert sorted(shuffled) == ["a", "b", "c", "d"]


def test_scan_keeps_slug_when_response_has_nan(client, monkeypatch) -> None:
    """Starlette не сериализует NaN: без защиты это 500, а у скрипта оценки — null."""
    pipeline = client.app.state.pipeline
    slug = next(iter(pipeline.catalog)).slug
    monkeypatch.setattr(pipeline, "scan", lambda payload: {"slug": slug, "confidence": float("nan")})
    response = client.post("/scan", files={"image": ("photo.png", png_bytes(), "image/png")})
    assert response.status_code == 200
    assert response.json()["slug"] == slug
    assert response.json()["degraded"] and "error" not in response.json()


def test_scan_keeps_found_slug_when_enrichment_fails(client, monkeypatch) -> None:
    """Сбой карточки или аналогов после поиска не должен подменять найденное вино."""
    pipeline = client.app.state.pipeline
    expected = client.post("/scan", files={"image": ("photo.png", png_bytes(), "image/png")}).json()

    def boom(*args, **kwargs):
        raise RuntimeError("enrich")

    monkeypatch.setattr(pipeline, "analogs", boom)
    monkeypatch.setattr(pipeline, "profile_analogs", boom)
    body = client.post("/scan", files={"image": ("photo.png", png_bytes(), "image/png")}).json()
    assert body["slug"] == expected["slug"]
    assert body["degraded"] and "error" not in body, "с error фронт покажет «Фото не прочитано»"
    assert body["confident"] is False and body["candidates"][0]["slug"] == expected["slug"]


def test_every_scan_shape_matches_published_schema(client, monkeypatch) -> None:
    """Схема в /docs — это контракт: обычный ответ, fallback и degraded ей соответствуют."""
    ok = client.post("/scan", files={"image": ("photo.png", png_bytes(), "image/png")}).json()
    broken = client.post("/scan", files={"image": ("a.txt", b"not an image", "text/plain")}).json()
    pipeline = client.app.state.pipeline
    monkeypatch.setattr(pipeline, "analogs", lambda *a: 1 / 0)
    monkeypatch.setattr(pipeline, "profile_analogs", lambda *a: 1 / 0)
    degraded = client.post("/scan", files={"image": ("photo.png", png_bytes(), "image/png")}).json()
    for body in (ok, broken, degraded):
        ScanResponse.model_validate(body)


def test_openapi_documents_scan_contract(client) -> None:
    spec = client.get("/openapi.json").json()
    scan = spec["paths"]["/scan"]["post"]
    assert "multipart/form-data" in scan["requestBody"]["content"]
    schema = scan["responses"]["200"]["content"]["application/json"]["schema"]["$ref"]
    assert schema.endswith("/ScanResponse")
    assert "slug" in spec["components"]["schemas"]["ScanResponse"]["required"]


def test_schema_holds_when_index_slug_is_missing_from_catalog(pipeline, monkeypatch) -> None:
    """Каталог обновили, индекс нет: карточки нет — card: null, а не {} вне схемы."""
    monkeypatch.setattr(pipeline.catalog, "get", lambda slug: None)
    assert pipeline._card(pipeline.catalog.get("x")) is None
    ScanResponse.model_validate(response.degraded("x", None, "r"))
