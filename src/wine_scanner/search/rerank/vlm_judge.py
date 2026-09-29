"""VLM-судья: финальный выбор среди нескольких финалистов.

Визуальный поиск и текст этикетки почти всегда приводят правильное вино в
топ-5, но внутри серии («Жемчужная 9 Цитрон, Шардоне» / «… Алиготе, Цитрон»)
иногда ставят первым соседа. Мультимодальная модель смотрит на этикетку и
на эталоны финалистов одновременно — как человек — и выбирает номер.

Судья не может придумать вино: он выбирает только среди переданных
кандидатов. Отказ, ошибка сети или непарсибельный ответ — вердикт None,
и ответ остаётся за основным пайплайном.

Протокол — OpenAI-совместимый /chat/completions: провайдер (DeepSeek,
Yandex AI Studio, любой другой) меняется в .env, не в коде.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path

import httpx
from PIL import Image, ImageOps

log = logging.getLogger(__name__)

SYSTEM = (
    "Ты эксперт по российским винам и винным этикеткам. Тебе показывают фото "
    "бутылки с полки магазина и несколько карточек каталога с эталонными фото. "
    "Определи, какая карточка — это именно то вино, что в центре фото. "
    "Смотри на название, винодельню, сорт, цвет вина и сахар (сухое/полусухое/"
    "полусладкое), дизайн этикетки. Год урожая не важен: другой винтаж того же "
    "вина — правильный ответ. Отвечай только JSON."
)


@dataclass(slots=True)
class JudgeCandidate:
    slug: str
    title: str
    winery: str
    style: str
    grape: str
    image: str  # локальный путь к эталонному фото


@dataclass(slots=True)
class Verdict:
    index: int | None  # номер выбранного кандидата (0-based) или None — «ни один»
    confidence: float
    elapsed: float
    raw: str = ""


def _data_url(image: Image.Image, side: int) -> str:
    image = image.convert("RGB")
    if max(image.size) > side:
        image = image.copy()
        image.thumbnail((side, side))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _parse(text: str, count: int) -> tuple[int | None, float]:
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        return None, 0.0
    try:
        data = json.loads(match.group(0))
        choice = int(data.get("choice", 0))
        confidence = float(data.get("confidence", 0.0))
    except (ValueError, TypeError):
        return None, 0.0
    if not 1 <= choice <= count:
        return None, confidence
    return choice - 1, confidence


class VLMJudge:
    def __init__(self, base_url: str, api_key: str, model: str, timeout: float = 20.0,
                 thinking: str = "disabled", query_side: int = 1024, thumb_side: int = 448
                 ) -> None:
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self.thinking = thinking
        self.query_side = query_side
        self.thumb_side = thumb_side
        self.name = f"{model}:{thinking}:json"
        self._client = httpx.Client(headers={"Authorization": f"Bearer {api_key}"},
                                    timeout=timeout)
        self._thumbs: dict[str, str] = {}  # эталоны каталога кодируются один раз

    def _thumb(self, path: str) -> str:
        if path not in self._thumbs:
            image = ImageOps.exif_transpose(Image.open(path))
            self._thumbs[path] = _data_url(image, self.thumb_side)
        return self._thumbs[path]

    def messages(self, query: Image.Image, candidates: list[JudgeCandidate], ocr_text: str
                 ) -> list[dict]:
        content: list[dict] = [
            {"type": "text", "text": "Фото с полки (этикетка в центре):"},
            {"type": "image_url", "image_url": {"url": _data_url(query, self.query_side)}},
        ]
        if ocr_text:
            content.append({"type": "text",
                            "text": f"Текст, распознанный на этикетке: {ocr_text[:400]}"})
        for number, candidate in enumerate(candidates, 1):
            details = ", ".join(x for x in (candidate.style, candidate.grape) if x)
            content.append({"type": "text", "text": (
                f"Кандидат {number}: «{candidate.title}» — {candidate.winery}"
                + (f" ({details})" if details else "")
            )})
            if candidate.image and Path(candidate.image).exists():
                content.append({"type": "image_url",
                                "image_url": {"url": self._thumb(candidate.image)}})
        content.append({"type": "text", "text": (
            f"Какой кандидат (1–{len(candidates)}) — вино на фото? Если ни один не "
            'подходит, choice = 0. Ответ строго JSON: {"choice": <номер>, '
            '"confidence": <0..1>}'
        )})
        return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": content}]

    def judge(self, query: Image.Image, candidates: list[JudgeCandidate], ocr_text: str = ""
              ) -> Verdict:
        started = time.perf_counter()
        body: dict = {
            "model": self.model,
            "messages": self.messages(query, candidates, ocr_text),
            "max_tokens": 300 if self.thinking == "disabled" else 3000,
            # Строгий JSON: без него модель иногда отвечает прозой, и вердикт теряется.
            "response_format": {"type": "json_object"},
        }
        if self.thinking == "disabled":
            body["thinking"] = {"type": "disabled"}
        else:
            body["reasoning_effort"] = self.thinking
        response = self._client.post(self.url, json=body)
        response.raise_for_status()
        text = response.json()["choices"][0]["message"].get("content") or ""
        index, confidence = _parse(text, len(candidates))
        return Verdict(index, confidence, time.perf_counter() - started, text)
