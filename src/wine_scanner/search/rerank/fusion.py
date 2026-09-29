"""Слияние сигналов ре-ранкинга.

Сейчас активен один сигнал — визуальный, остальные весят 0 и подключатся
после замеров (текстовый через VLM/OCR, геометрический через инлайеры
локальных признаков). Смысл модуля в том, чтобы это подключение было
изменением весов в config.yaml, а не переписыванием поиска.

Веса подбираются ЗАМЕРАМИ на hold-out. Любое значение, поставленное «на
глаз», обязано быть помечено как временное.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class Candidate:
    slug: str
    visual: float = 0.0
    textual: float = 0.0
    geometric: float = 0.0
    score: float = 0.0
    signals: dict[str, float] = field(default_factory=dict)


def fuse(
    candidates: list[Candidate],
    weights: dict[str, float],
) -> list[Candidate]:
    """Линейное слияние. Возвращает список, отсортированный по убыванию score.

    Линейная комбинация, а не обученный ранкер, — сознательно: пока сигнал
    один, LightGBM обучать не на чем, а неинтерпретируемый скор мешает
    разбирать ошибки.
    """
    w_visual = float(weights.get("visual", 1.0))
    w_textual = float(weights.get("textual", 0.0))
    w_geometric = float(weights.get("geometric", 0.0))

    for candidate in candidates:
        candidate.signals = {
            "visual": round(candidate.visual, 4),
            "textual": round(candidate.textual, 4),
            "geometric": round(candidate.geometric, 4),
        }
        candidate.score = (
            w_visual * candidate.visual
            + w_textual * candidate.textual
            + w_geometric * candidate.geometric
        )

    return sorted(candidates, key=lambda c: c.score, reverse=True)


def rank_candidates(
    index,
    vector,
    text_scores: dict[str, float],
    weights: dict[str, float],
    top_k: int = 50,
    text_pool: int = 20,
) -> list[Candidate]:
    """Визуальный пул top_k + текстовый пул text_pool -> слияние.

    Единая точка ранжирования для сервиса и стендов оценки: любое улучшение
    здесь сразу видно и в scripts/eval_offline.py, и в ответе сервиса.
    """
    hits = index.search(vector, k=top_k)
    if not hits:
        raise RuntimeError("Индекс не вернул кандидатов — проверьте сборку индекса")
    visual = dict(hits)
    text_top = sorted(text_scores, key=text_scores.get, reverse=True)[:text_pool]
    missing = [slug for slug in text_top if slug not in visual]
    if missing:
        visual.update(index.score(vector, missing))
    floor = min(visual.values())
    return fuse(
        [
            Candidate(slug=slug, visual=visual.get(slug, floor),
                      textual=text_scores.get(slug, 0.0))
            for slug in dict.fromkeys([*visual, *text_top])
        ],
        weights,
    )
