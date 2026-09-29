"""Уверенность и F1-метрики ответа.

ТЗ требует отдавать «метрику уверенности (F1) для топ-1 и топ-5». F1 на
единичном запросе не определён в классическом смысле, поэтому под ним
понимается откалиброванная вероятность правильности top-1 и вероятность
того, что правильный ответ попал в top-5 — именно так это подаётся в UI и
в скрипт оценки.
"""

from __future__ import annotations

import numpy as np


def softmax_scores(scores: list[float], temperature: float) -> list[float]:
    """Softmax с температурой по косинусным скорам.

    Косинусы близких кандидатов различаются в третьем знаке, поэтому без
    низкой температуры распределение вырождается в равномерное и confidence
    перестаёт что-либо различать.
    """
    if not scores:
        return []
    values = np.asarray(scores, dtype=np.float64) / max(temperature, 1e-6)
    values -= values.max()
    exponent = np.exp(values)
    return (exponent / exponent.sum()).tolist()


def estimate(
    scores: list[float],
    temperature: float,
    groups: list[str] | None = None,
) -> dict[str, float]:
    """-> confidence, top1_f1, top5_f1, margin.

    `groups` — ключ «то же вино» для каждого кандидата (одинаковые название и
    винодельня: другой винтаж или дубль карточки). Вероятности одной группы
    складываются: два близнеца по 0.45 — это уверенный ответ 0.9, а не два
    сомнительных. Иначе UI показывал бы экран выбора там, где выбирать не из чего.

    `margin` (разрыв между первым и вторым кандидатом другой группы) полезнее
    самой уверенности при разборе ошибок: он сразу показывает, спутаны ли
    near-duplicates одной серии.
    """
    if not scores:
        return {"confidence": 0.0, "top1_f1": 0.0, "top5_f1": 0.0, "margin": 0.0}

    groups = groups or [str(i) for i in range(len(scores))]
    probabilities = softmax_scores(scores, temperature)
    mass: dict[str, float] = {}
    for key, probability in zip(groups, probabilities, strict=True):
        mass[key] = mass.get(key, 0.0) + probability
    ordered = list(dict.fromkeys(groups))  # группы в порядке ранга
    top1 = mass[ordered[0]]
    top5 = float(sum(mass[key] for key in ordered[:5]))
    runner_up = next((s for s, g in zip(scores, groups, strict=True) if g != groups[0]), None)
    margin = float(scores[0] - runner_up) if runner_up is not None else float(scores[0])

    return {
        "confidence": round(top1, 4),
        "top1_f1": round(top1, 4),
        "top5_f1": round(min(top5, 1.0), 4),
        "margin": round(margin, 4),
    }


def decide(
    config: dict,
    confidence: float,
    visual: float,
    textual: float,
    name_recall: float,
    name_idf: float,
) -> tuple[bool, float]:
    """-> (показать карточку сразу, match_score). Только сигнал для UI.

    `config` — секция `confidence` из config.yaml.
    """
    absolute = config.get("absolute") or {}
    name = config.get("name") or {}
    # Абсолютное совпадение: softmax видит только разрыв между кандидатами,
    # а вино вне каталога часто родственник вина из каталога той же серии.
    match_score = visual + float(absolute.get("text_weight", 0.5)) * textual
    # Эталон каталога бывает скриншотом с другим светом и ракурсом: картинка
    # слабая (visual ~0.2), но редкое название вина прочитано на этикетке.
    name_match = (name_recall >= float(name.get("recall", 1.1))
                  and name_idf >= float(name.get("min_idf", 0.0)))
    confident = confidence >= float(config.get("threshold", 0.5)) and (
        match_score >= float(absolute.get("threshold", 0.0)) or name_match
    )
    return confident, match_score
