"""Текстовый канал, сопоставление фото каталога, индекс с несколькими эталонами."""

from __future__ import annotations

import numpy as np

from wine_scanner.catalog.models import Wine
from wine_scanner.catalog.photo_match import match_key, match_photos
from wine_scanner.features.text import TextLine
from wine_scanner.search.confidence import decide, estimate
from wine_scanner.search.index import NumpyIndex
from wine_scanner.search.rerank.text_match import TextMatcher, loose, tokenize

SERIES = [
    Wine(slug="zhemchuzhnaya-9-czitron-shardone", title="Жемчужная 9 Цитрон, Шардоне", winery="АРАТТИ"),
    Wine(slug="zhemchuzhnaya-9-aligote-czitron", title="Жемчужная 9 Алиготе, Цитрон", winery="АРАТТИ"),
    Wine(slug="czitronnyj-magaracha-sovinon-blan", title="Жемчужная 9 Цитрон, Совиньон блан", winery="АРАТТИ"),
    Wine(slug="massandra-muskat", title="Массандра Мускат", winery="Массандра"),
    Wine(slug="massandra-muskatel-belyy-belye-sorta-vinograda-beloe-sladkoe-16", title="Мускатель белый", winery="Массандра"),
    Wine(slug="inkerman-riesling", title="Инкерман Рислинг", winery="Инкерманский ЗМВ"),
]


def lines(*texts: str, height: float = 0.05) -> list[TextLine]:
    return [TextLine(text=t, height=height) for t in texts]


def best(scores: dict[str, float]) -> str:
    return max(scores, key=scores.get)


def test_loose_transliteration_unifies_label_and_slug_spelling():
    # Этикетка кириллицей, slug в своём транслите — токены должны совпасть.
    assert loose("Цитрон") == loose("czitron") == loose("tsitron")
    assert tokenize("INKERMAN") == tokenize("Инкерман")


def test_year_and_alcohol_do_not_become_tokens():
    assert "2024" not in tokenize("Урожай 2024")
    assert "135" not in tokenize("merlo krasnoe suhoe 135")
    assert "9" in tokenize("Жемчужная 9")  # номер серии различает вина


def test_text_separates_near_duplicates_of_one_series():
    matcher = TextMatcher(SERIES)
    scores = matcher.score(lines("ЖЕМЧУЖНАЯ", "9", "ЦИТРОН", "ШАРДОНЕ", "БЕЛОЕ ПОЛУСЛАДКОЕ"))
    assert best(scores) == "zhemchuzhnaya-9-czitron-shardone"


def test_precision_beats_short_titles():
    # Одна полнота выбрала бы «Массандра Мускат»: у неё короче название.
    matcher = TextMatcher(SERIES)
    scores = matcher.score(lines("МАССАНДРА", "МУСКАТЕЛЬ", "БЕЛЫЙ", "ВИНО РОССИИ"))
    assert best(scores) == "massandra-muskatel-belyy-belye-sorta-vinograda-beloe-sladkoe-16"


def test_latin_label_matches_cyrillic_card():
    matcher = TextMatcher(SERIES)
    assert best(matcher.score(lines("INKERMAN", "RIESLING"))) == "inkerman-riesling"


def test_no_text_gives_no_scores():
    assert TextMatcher(SERIES).score([]) == {}


def test_name_recall_counts_own_name_not_winery():
    matcher = TextMatcher(SERIES)
    wine = SERIES[5]  # «Инкерман Рислинг» от «Инкерманский ЗМВ»: название — «Рислинг»
    own, _ = matcher.name_recall(lines("РИСЛИНГ"), wine)
    winery_only, _ = matcher.name_recall(lines("ИНКЕРМАНСКИЙ", "ЗМВ"), wine)
    assert own > 0.9
    assert winery_only == 0.0
    assert matcher.name_recall([], wine)[0] == 0.0


def test_name_idf_is_rarest_word_not_sum():
    # Пять общих слов (каждое в 10 карточках из 100) по сумме IDF обошли бы
    # одно редкое, но вино не опознают.
    generic = [Wine(slug=f"g{i}", title="Каберне Совиньон Красное Сухое Резерв", winery=f"W{i}")
               for i in range(10)]
    rare = [Wine(slug=f"r{i}", title=f"Уникум{i}", winery=f"R{i}") for i in range(90)]
    matcher = TextMatcher(generic + rare)
    _, generic_idf = matcher.name_recall(lines("КАБЕРНЕ СОВИНЬОН"), generic[0])
    _, rare_idf = matcher.name_recall(lines("УНИКУМ0"), rare[0])
    assert generic_idf < 4.0 <= rare_idf


def test_strapi_names_match_dump_names():
    assert match_key("Агора_Мускат.webp") == match_key("Agora_Muskat_4030bf4725.webp", strapi=True)
    found = match_photos(
        {"agora-muskat": "Агора_Мускат.webp", "katharon": "Katharon - Рислинг (2025).webp"},
        ["Agora_Muskat_4030bf4725.webp", "medium_Agora_Muskat_4030bf4725.webp",
         "Katharon_Risling_2025_983f02b959.webp"],
    )
    assert found["agora-muskat"].files == ["Agora_Muskat_4030bf4725.webp"]  # без medium_-копии
    assert found["katharon"].files == ["Katharon_Risling_2025_983f02b959.webp"]


def test_background_removed_suffix_matches_in_both_scripts():
    """«без фона» в дампе на диске превращается в bez_fona — три вина каталога
    из-за этого остались без эталона и не находились по картинке."""
    found = match_photos(
        {"lorio": "Игристое Розовое (Пино Нуар) без фона.webp", "tamagne": "тамань шато без фона.webp"},
        ["Igristoe_Rozovoe_Pino_Nuar_bez_fona_0156c8af31.webp", "taman_shato_bez_fona_d2c12c80a3.webp"],
    )
    assert found["lorio"].files == ["Igristoe_Rozovoe_Pino_Nuar_bez_fona_0156c8af31.webp"]
    assert found["tamagne"].files == ["taman_shato_bez_fona_d2c12c80a3.webp"]


def test_same_name_uploads_are_all_returned_for_disambiguation():
    found = match_photos({"oleg": "Screenshot_9.webp"},
                         ["Screenshot_9_7d27b4ab94.webp", "Screenshot_9_78adaf4455.webp"])
    assert len(found["oleg"].files) == 2


def test_twin_cards_share_confidence():
    # Два близнеца по полвероятности — это одно уверенное вино.
    metrics = estimate([0.90, 0.90, 0.50], temperature=0.05, groups=["a", "a", "b"])
    assert metrics["confidence"] > 0.99
    alone = estimate([0.90, 0.90, 0.50], temperature=0.05)
    assert alone["confidence"] < 0.51


def test_index_aggregates_rows_per_slug_and_excludes_sources():
    index = NumpyIndex(dim=2)
    index.add(np.array([[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]]), ["a", "b", "a"],
              sources=["catalog", "catalog", "field.jpg"])
    assert len(index) == 2 and index.rows == 3
    hits = index.search(np.array([[0.6, 0.8]]), k=5)
    assert [slug for slug, _ in hits] == ["a", "b"]  # «a» найдено по второму эталону
    hits = index.search(np.array([[0.6, 0.8]]), k=5, exclude={"field.jpg"})
    assert hits[0][0] == "b"
    assert set(index.score(np.array([[1.0, 0.0]]), ["b"])) == {"b"}


def test_decide_needs_gap_and_absolute_match_or_rare_name() -> None:
    config = {"threshold": 0.5, "absolute": {"text_weight": 0.5, "threshold": 0.6},
              "name": {"recall": 0.6, "min_idf": 4.0}}
    assert decide(config, 0.9, 0.5, 0.4, 0.0, 0.0) == (True, 0.7)
    assert decide(config, 0.4, 0.9, 1.0, 1.0, 9.0)[0] is False, "нет отрыва от второго"
    assert decide(config, 0.9, 0.2, 0.0, 0.0, 0.0)[0] is False, "вино похоже, но не то"
    assert decide(config, 0.9, 0.2, 0.0, 0.8, 5.1)[0] is True, "прочитано редкое название"
    assert decide(config, 0.9, 0.2, 0.0, 0.8, 2.4)[0] is False, "«Мерло» вино не опознаёт"
