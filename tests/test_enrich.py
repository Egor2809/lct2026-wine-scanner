"""Функции после поиска: гастрономия, демо-рейтинг, аналоги, профиль, сомелье.

Всё детерминированно и без моделей — поэтому проверяется на маленьких
каталогах в памяти.
"""

from __future__ import annotations

from wine_scanner.catalog.loader import Catalog
from wine_scanner.catalog.models import Wine
from wine_scanner.enrich.analogs import (
    REASON_GRAPE,
    REASON_REGION,
    REASON_STYLE,
    build_profile,
    match_analogs,
    out_of_catalog,
    profile_analogs,
)
from wine_scanner.enrich.card import card_for
from wine_scanner.enrich.pairing import from_description, pairing_for
from wine_scanner.enrich.rating import HIGH, LOW, demo_rating
from wine_scanner.enrich.sommelier import pick
from wine_scanner.enrich.traits import label_traits, split_grapes, sugar_of, traits


def red(slug: str, winery: str, grape: str, region: str = "Крым", **kw) -> Wine:
    return Wine(slug=slug, title=kw.pop("title", slug), winery=winery, grape=grape,
                style="Красное", region=region, **kw)


# ── признаки ──────────────────────────────────────────────────────────────


def test_grapes_split_into_normalized_sets() -> None:
    keys = [k for k, _ in split_grapes("Каберне Совиньон, Мерло и Шираз")]
    assert keys == ["каберне совиньон", "мерло", "сира"]
    assert [k for k, _ in split_grapes("Рислинг Рейнский")] == ["рислинг"]
    assert [k for k, _ in split_grapes("Пино черный (Пино Нуар)")] == ["пино нуар"]
    assert split_grapes("Белые сорта винограда") == [], "это не сорт"
    assert {k for k, _ in split_grapes("Каберне Фран")} != {k for k, _ in split_grapes("Каберне Совиньон")}


def test_sugar_from_title() -> None:
    assert sugar_of("Мускат белое полусладкое") == "полусладкое"
    assert sugar_of("Tête de Cheval сладкое") == "сладкое"
    assert sugar_of("Кокур белое полусухое") == "полусухое"
    assert sugar_of("5 elements Брют белое") == "брют"
    assert sugar_of("Каберне Совиньон") == ""


def test_label_text_gives_grapes_style_and_region() -> None:
    found = label_traits("AGORA | Cabernet Franc | red dry wine | Crimea")
    assert found["grapes"] == ["каберне фран"]
    assert found["style"] == "Красное"
    assert found["sugar"] == "сухое"
    assert found["region"] == "Крым"
    assert label_traits("") == {}


# ── к чему подать ─────────────────────────────────────────────────────────


def test_pairing_reads_winery_advice_in_nominative() -> None:
    note = ("Вкус насыщенный. Хорошо сочетается с белым мясом, телятиной, "
            "печёными овощами, сырами бри и камамбер.")
    wine = red("x", "А", "Каберне Фран", description=note)
    pairing = pairing_for(wine)
    assert pairing.source == "description"
    assert pairing.labels[:2] == ["Белое мясо", "Телятина и свинина"]
    assert "Мягкие сыры" in pairing.labels


def test_pairing_ignores_taste_phrases() -> None:
    assert from_description("Вкус сочетает кислотность с лёгкостью.") == ()
    assert from_description("Аромат сочетается со спелыми ягодами ежевики.") == (), \
        "ягоды в аромате — не блюдо"


def test_pairing_rules_by_sugar_grape_and_style() -> None:
    brut = Wine(slug="b", title="Игристое брют", grape="Шардоне", style="Белое")
    assert pairing_for(brut).labels[0] == "Аперитив"
    assert pairing_for(brut).basis == "брют"

    cabernet = red("c", "А", "Каберне Совиньон")
    assert pairing_for(cabernet).labels[0] == "Стейк из говядины"
    assert pairing_for(cabernet).basis == "Каберне Совиньон"

    sweet = Wine(slug="s", title="Мускат сладкое", grape="Мускат", style="Белое")
    assert pairing_for(sweet).labels[0] == "Десерты"

    # Красный сорт в розовом вине — правило сорта (стейк) не применяется.
    rose = Wine(slug="r", title="Розе", grape="Каберне Совиньон", style="Розовое")
    assert "Стейк из говядины" not in pairing_for(rose).labels


def test_every_card_gets_pairing_and_demo_rating() -> None:
    wine = red("agora-cabernet-franc-2021", "AGORA", "Каберне Фран")
    card = card_for(wine)
    for key in wine.card():
        assert key in card, "ключи Wine.card() читает фронтенд — их нельзя терять"
    assert card["pairing"] and card["pairing_source"] == "rules"
    assert card["rating"]["demo"] is True, "рейтинг выдуман — пометка обязательна"
    assert LOW <= card["rating"]["value"] <= HIGH


def test_catalog_pairing_wins_over_rules() -> None:
    wine = red("x", "А", "Мерло", pairing=["Баранина"])
    card = card_for(wine)
    assert card["pairing"] == ["Баранина"]
    assert card["pairing_source"] == "catalog"


def test_demo_rating_is_stable_and_shared_by_vintages() -> None:
    assert demo_rating("sikory-riesling-2021") == demo_rating("sikory-riesling-2023")
    values = {demo_rating(f"wine-{i}") for i in range(300)}
    assert min(values) >= LOW and max(values) <= HIGH
    assert len(values) > 8, "рейтинг должен различать вина"


# ── аналоги ───────────────────────────────────────────────────────────────


def test_cabernet_franc_is_not_cabernet_sauvignon() -> None:
    catalog = Catalog([
        red("a-franc", "А", "Каберне Фран"),
        red("b-sauvignon", "Б", "Каберне Совиньон"),
        red("c-franc", "В", "Каберне Фран", region="Кубань"),
        red("d-blend", "Г", "Каберне Фран, Каберне Совиньон, Мерло, Саперави, Сира, Марселан, Пти Вердо"),
    ])
    analogs = match_analogs(catalog, catalog.get("a-franc"), count=3)
    assert analogs[0].wine.slug == "c-franc", "тот же сорт важнее того же региона"
    assert REASON_GRAPE in analogs[0].reasons and REASON_STYLE in analogs[0].reasons
    blend = next(a for a in analogs if a.wine.slug == "d-blend")
    assert REASON_GRAPE not in blend.reasons, "купаж из семи сортов — не чистый сорт"
    assert blend.reasons[0].startswith("общий сорт")


def test_analogs_skip_own_winery_and_keep_one_per_winery() -> None:
    catalog = Catalog(
        [red("a-merlot", "А", "Мерло"), red("a-merlot-reserve", "А", "Мерло")]
        + [red(f"b-merlot-{i}", "Б", "Мерло") for i in range(3)]
        + [red("c-merlot", "В", "Мерло")]
    )
    analogs = match_analogs(catalog, catalog.get("a-merlot"), count=4)
    wineries = [a.wine.winery for a in analogs]
    assert "А" not in wineries, "аналоги — только других виноделен"
    assert sorted(wineries) == ["Б", "В"]
    assert all(REASON_REGION in a.reasons for a in analogs)


def test_sweet_wine_prefers_sweet_analogs() -> None:
    catalog = Catalog([
        Wine(slug="a", title="Рислинг сладкое", winery="А", grape="Рислинг", style="Белое"),
        Wine(slug="b", title="Рислинг", winery="Б", grape="Рислинг", style="Белое"),
        Wine(slug="c", title="Мускат полусладкое", winery="В", grape="Мускат", style="Белое"),
    ])
    assert match_analogs(catalog, catalog.get("a"), 2)[0].wine.slug == "c"


# ── вина нет в каталоге: профиль ──────────────────────────────────────────


def test_profile_votes_over_candidates_and_trusts_label() -> None:
    catalog = Catalog([
        red("a-franc", "А", "Каберне Фран"),
        red("b-franc", "Б", "Каберне Фран"),
        Wine(slug="c-riesling", winery="В", grape="Рислинг", style="Белое", region="Кубань"),
        red("d-saperavi", "Г", "Саперави", region="Кубань"),
        red("e-franc", "Д", "Каберне Фран"),
        red("f-saperavi", "Е", "Саперави", region="Кубань"),
    ])
    top = [(catalog.get("a-franc"), 0.6), (catalog.get("b-franc"), 0.5), (catalog.get("c-riesling"), 0.2)]

    profile = build_profile(top)
    assert profile.style == "Красное"
    assert profile.grapes == frozenset({"каберне фран"})
    assert profile.region == "Крым"
    analogs = profile_analogs(catalog, profile, 4, exclude=[w for w, _ in top])
    assert analogs[0].wine.slug == "e-franc", "кандидаты уже показаны — их в аналогах нет"
    assert not {"a-franc", "b-franc"} & {a.wine.slug for a in analogs}

    by_label = build_profile(top, "SAPERAVI · Kuban")
    assert by_label.grapes == frozenset({"саперави"})
    assert by_label.region == "Кубань"
    assert set(by_label.from_label) == {"grapes", "region"}
    slugs = [a.wine.slug for a in profile_analogs(catalog, by_label, 2, exclude=[w for w, _ in top])]
    assert set(slugs) == {"d-saperavi", "f-saperavi"}


def test_out_of_catalog_skips_winery_of_best_candidate() -> None:
    catalog = Catalog([
        red("a-franc", "А", "Каберне Фран"),
        red("a-franc-reserve", "А", "Каберне Фран"),
        red("b-franc", "Б", "Каберне Фран"),
    ])
    profile, analogs = out_of_catalog(catalog, [(catalog.get("a-franc"), 0.6), (None, 0.1)], "")
    assert profile.grapes == frozenset({"каберне фран"})
    assert [a.wine.slug for a in analogs] == ["b-franc"]


def test_profile_survives_zero_scores() -> None:
    wine = red("a", "А", "Мерло")
    assert build_profile([(wine, 0.0), (None, 0.0)]).style == "Красное"
    assert build_profile([]).style == ""


# ── сомелье ───────────────────────────────────────────────────────────────


def test_sommelier_offers_several_styles_with_reasons() -> None:
    catalog = Catalog([
        Wine(slug="o", title="Оранж", winery="А", grape="Ркацители", style="Оранжевое"),
        red("r", "Б", "Каберне Совиньон"),
        Wine(slug="w", title="Мускат сладкое", winery="В", grape="Мускат", style="Белое"),
        Wine(slug="f", title="Алиготе", winery="Г", grape="Алиготе", style="Белое"),
    ])
    found = pick(catalog, "cheese", seed=1)
    styles = {traits(p.wine).style for p in found.items}
    assert {"Оранжевое", "Красное", "Белое"} <= styles, "сыры — не только оранжевые вина"
    assert all(p.why for p in found.items)
    assert "f" not in {p.wine.slug for p in found.items}, "Алиготе к сырам правила не дают"


def test_sommelier_relaxes_filters_and_says_so() -> None:
    catalog = Catalog([red("r", "Б", "Каберне Совиньон", region="Кубань")])
    found = pick(catalog, "meat", region="Дагестан", sparkling=False)
    assert found.items and found.relaxed
    assert found.used == {"sparkling": False}


# ── HTTP ──────────────────────────────────────────────────────────────────


def test_api_cards_and_analogs_carry_new_fields(client) -> None:
    body = client.get("/wines/vedernikov-krasnostop-2021").json()
    assert body["card"]["rating"]["demo"] is True
    assert body["card"]["pairing"]
    assert all("reasons" in a for a in body["analogs"])
    slugs = {a["slug"] for a in body["analogs"]}
    assert "vedernikov-krasnostop-2022" not in slugs, "другой винтаж — не аналог"


def test_api_sommelier(client) -> None:
    body = client.get("/sommelier", params={"food": "meat", "seed": 3}).json()
    assert body["items"] and all(item["why"] for item in body["items"])
    assert {"total", "relaxed", "used", "regions"} <= set(body)
    assert client.get("/sommelier", params={"food": "pizza"}).status_code == 400


def test_scan_builds_analogs_from_profile_when_not_confident(client) -> None:
    import io

    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (240, 360), (10, 200, 30)).save(buffer, format="PNG")
    body = client.post("/scan", files={"image": ("x.png", buffer.getvalue(), "image/png")}).json()
    assert isinstance(body["slug"], str) and body["slug"]
    assert "analogs_profile" in body
    if body["confident"]:
        assert body["analogs_profile"] is None
    else:
        assert isinstance(body["analogs_profile"], dict)
        top = {c["slug"] for c in body["candidates"][:5]}
        assert not top & {a["slug"] for a in body["analogs"]}
