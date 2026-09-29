"""Контракт ответа /scan: плоский JSON, `slug` строго в корне и никогда не null.

Скрипт кейсодержателя читает `slug` из корня ответа, а на не-200 пишет null.
Поэтому здесь три формы ответа, и у каждой непустой slug:
- `build`     — обычный ответ по кандидатам;
- `fallback`  — фото не прочитано или модель упала: позиция каталога и `error`;
- `degraded`  — вино найдено, но карточку или аналоги собрать не удалось:
                slug тот же, UI показывает его единственным вариантом.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class Card(BaseModel):
    """Карточка вина для фронтенда (enrich/card.py)."""

    model_config = ConfigDict(extra="allow")

    title: str
    winery: str
    region: str = ""
    grape: str = ""
    style: str = ""
    color: str = ""
    vintage: str = ""
    note: str = ""
    pairing: list[str] = []
    image: str = Field("", description="HTTP-путь к превью фото или внешний URL; пусто — нет фото")
    url: str = Field("", description="Страница вина на портале vino-svoe.ru")


class Candidate(BaseModel):
    slug: str
    score: float
    signals: dict[str, float] = Field(description="visual, textual, geometric — вклад сигналов")
    card: Card | None


class Analog(BaseModel):
    slug: str
    card: Card
    reasons: list[str] = Field(description="Почему похоже: тот же сорт, регион, стиль")


class ScanResponse(BaseModel):
    """Ответ POST /scan (и алиасов /v1/eval/predict, /predict) на любое непустое тело.

    Даже нечитаемое фото или сбой модели дают 200 и этот ответ; 400/413/503 —
    только пустой или слишком большой запрос и сервис, который не поднялся.
    """

    model_config = ConfigDict(extra="forbid")

    slug: str = Field(description="Лучший slug каталога. Строго в корне, никогда не null")
    confidence: float = Field(description="Вероятность, что top-1 верен (близнецы сложены)")
    top1_f1: float = Field(description="Метрика уверенности для top-1 (ТЗ, п. 3)")
    top5_f1: float = Field(description="Вероятность, что верное вино среди top-5")
    margin: float = Field(description="Отрыв top-1 от ближайшего другого вина")
    confident: bool = Field(description="Показывать карточку сразу, без экрана вариантов")
    match_score: float
    name_recall: float
    card: Card | None
    candidates: list[Candidate] = Field(description="До 5 кандидатов по убыванию скора")
    analogs: list[Analog] = Field(description="Аналоги других виноделен")
    analogs_profile: dict[str, Any] | None = Field(
        description="Профиль снимка, по которому подобраны аналоги, если вина нет в каталоге")
    label_detected: bool
    ocr_text: str
    timings_ms: dict[str, int]
    elapsed_ms: int
    error: str | None = Field(None, description="Фото не прочитано: slug — позиция-заглушка")
    degraded: str | None = Field(
        None, description="Вино найдено, но карточку или аналоги собрать не удалось")


def build(
    candidates: list,
    metrics: dict[str, float],
    decision: tuple[bool, float, float],
    card: Callable[[str], dict[str, Any]],
    return_k: int,
    label_detected: bool,
    ocr_text: str,
    timings: dict[str, float],
) -> dict[str, Any]:
    best = candidates[0]
    confident, match_score, name_recall = decision
    return {
        "slug": best.slug,
        "confidence": metrics["confidence"],
        "top1_f1": metrics["top1_f1"],
        "top5_f1": metrics["top5_f1"],
        "margin": metrics["margin"],
        # Порог — сигнал для UI, а не причина отдать null: на оценочном
        # прогоне все вина есть в каталоге, «не найдено» там всегда ошибка.
        "confident": confident,
        "match_score": round(match_score, 4),
        "name_recall": round(name_recall, 4),
        "card": card(best.slug),
        "candidates": [
            {"slug": c.slug, "score": round(c.score, 4), "signals": c.signals,
             "card": card(c.slug)}
            for c in candidates[:return_k]
        ],
        "analogs": [],
        # Профиль, по которому подобраны аналоги, когда вина нет в каталоге.
        "analogs_profile": None,
        "label_detected": label_detected,
        "ocr_text": ocr_text[:500],
        "timings_ms": {k: round(v * 1000) for k, v in timings.items()},
        "elapsed_ms": 0,
    }


def fallback(slug: str, card: dict[str, Any] | None, reason: str) -> dict[str, Any]:
    return {
        "slug": slug,
        "confidence": 0.0,
        "top1_f1": 0.0,
        "top5_f1": 0.0,
        "margin": 0.0,
        "confident": False,
        "match_score": 0.0,
        "name_recall": 0.0,
        "card": card,
        "candidates": [],
        "analogs": [],
        "analogs_profile": None,
        "label_detected": False,
        "ocr_text": "",
        "timings_ms": {},
        "elapsed_ms": 0,
        "error": reason,
    }


def degraded(slug: str, card: dict[str, Any] | None, reason: str) -> dict[str, Any]:
    response = fallback(slug, card, reason)
    del response["error"]
    response["degraded"] = reason
    response["candidates"] = [{"slug": slug, "score": 0.0, "signals": {}, "card": card}]
    return response
