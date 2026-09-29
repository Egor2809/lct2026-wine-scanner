"""Облачный OCR: Yandex Vision OCR (AI Studio).

POST https://ocr.api.cloud.yandex.net/ocr/v1/recognizeText
Авторизация — API-ключ сервисного аккаунта с ролью ai.vision.user,
каталог — заголовок x-folder-id. Ключ только из ENV (YANDEX_API_KEY).
Тариф ~0.13 ₽ за изображение.

Устойчивость. SLA запроса — 3 с, облако не должно съедать его целиком:
- весь вызов укладывается в `deadline`, каждая попытка — в остаток дедлайна;
- квота облака ~1 запрос/с: запросы разносятся на `min_interval`, иначе
  серия фото подряд ловит 429 (замер: 14 из 20 без паузы);
- 429, 5xx, таймаут и обрыв сети повторяются с коротким бэкоффом;
- одиночный сбой обслуживает запасной экстрактор только для этого фото
  (случайный таймаут не должен отключать облако на десятки фото — было,
  -1.5 п.п. top-1). После `trip_after` сбоев подряд облако выключается на
  `cooldown` секунд (401/403 — сразу и до перезапуска), и запросы не платят
  дедлайн заново на каждом фото.
"""

from __future__ import annotations

import base64
import io
import logging
import math
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout

import httpx
from PIL import Image

from . import TextLine

log = logging.getLogger(__name__)

ENDPOINT = "https://ocr.api.cloud.yandex.net/ocr/v1/recognizeText"
RETRY_STATUSES = {429, 500, 502, 503, 504}
AUTH_STATUSES = {401, 403}
MIN_ATTEMPT = 0.2  # меньше этого на попытку не остаётся смысла стучаться


class OCRUnavailable(RuntimeError):
    """Облачный OCR не ответил в пределах дедлайна или выключен предохранителем."""


class YandexOCRExtractor:
    name = "yandex-ocr"

    def __init__(
        self,
        api_key: str,
        folder_id: str = "",
        deadline: float = 1.6,
        attempt_timeout: float = 1.2,
        backoff: tuple[float, ...] = (0.15, 0.3),
        cooldown: float = 30.0,
        trip_after: int = 2,
        min_interval: float = 1.0,
        max_side: int = 1600,
        transport: httpx.BaseTransport | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.max_side = max_side
        self.deadline = deadline
        self.attempt_timeout = attempt_timeout
        self.backoff = backoff
        self.cooldown = cooldown
        self.trip_after = trip_after
        self.min_interval = min_interval
        self._failures = 0
        self._last_start = -math.inf
        self._clock = clock
        self._sleep = sleep
        self._disabled_until = 0.0
        headers = {"Authorization": f"Api-Key {api_key}"}
        if folder_id:
            headers["x-folder-id"] = folder_id
        self._client = httpx.Client(headers=headers, transport=transport)
        self._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="yandex-ocr")

    def extract(self, image: Image.Image) -> list[TextLine]:
        if self._clock() < self._disabled_until:
            raise OCRUnavailable("облачный OCR временно выключен после сбоя")

        image = image.convert("RGB")
        if max(image.size) > self.max_side:
            image = image.copy()
            image.thumbnail((self.max_side, self.max_side))
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=90)
        body = {
            "mimeType": "JPEG",
            "languageCodes": ["ru", "en"],
            "model": "page",
            "content": base64.b64encode(buffer.getvalue()).decode("ascii"),
        }
        try:
            payload = self._post(body)
        except OCRUnavailable:
            raise
        except Exception as exc:  # неожиданный ответ: тоже сбой облака, а не падение
            self._fail("неожиданный ответ")
            raise OCRUnavailable(str(exc)) from exc
        self._failures = 0
        return _parse(payload, image.height)

    def _fail(self, problem: str) -> str:
        self._failures += 1
        if self._failures < self.trip_after:
            return problem
        # Квота (429) отпускает быстро, сеть — нет: разные паузы предохранителя.
        pause = min(self.cooldown, 3.0) if problem == "HTTP 429" else self.cooldown
        self._disabled_until = self._clock() + pause
        log.warning("Yandex OCR: %d сбоя подряд, облако выключено на %.0f c",
                    self._failures, pause)
        return f"{problem}; облако выключено на {pause:.0f} c"

    def _post(self, body: dict) -> dict:
        started = self._clock()
        problem = "дедлайн исчерпан"
        for attempt in range(len(self.backoff) + 1):
            wait = max(0.0, self._last_start + self.min_interval - self._clock())
            remaining = self.deadline - (self._clock() - started) - wait
            if remaining < MIN_ATTEMPT:
                break
            if wait:
                self._sleep(wait)
            self._last_start = self._clock()
            timeout = min(self.attempt_timeout, remaining)
            # Таймауты httpx не покрывают DNS: при сбое сети getaddrinfo висел
            # 30 с на запрос (замер на Mac). Поэтому ждём ответ из отдельного
            # потока ровно до таймаута, зависший поток просто бросаем.
            future = self._executor.submit(self._client.post, ENDPOINT, json=body,
                                           timeout=timeout)
            try:
                response = future.result(timeout=timeout + 0.05)
            except FuturesTimeout:
                problem = "нет ответа за таймаут (DNS или соединение)"
            except httpx.TransportError as exc:  # таймаут, обрыв соединения
                problem = f"{type(exc).__name__}: {exc}"
            else:
                if response.status_code in AUTH_STATUSES:
                    # Ключ не станет валидным сам: не тратим SLA ни на одном запросе.
                    self._disabled_until = math.inf
                    log.error("Yandex OCR отклонил ключ (%d): облако выключено до перезапуска",
                              response.status_code)
                    raise OCRUnavailable(f"HTTP {response.status_code}: ключ отклонён")
                if response.status_code not in RETRY_STATUSES:
                    response.raise_for_status()
                    return response.json()
                problem = f"HTTP {response.status_code}"
            if attempt < len(self.backoff):
                # После 429 следующую попытку всё равно сдвинет min_interval.
                pause = self.backoff[attempt]
                if self.deadline - (self._clock() - started) - pause < MIN_ATTEMPT:
                    break
                self._sleep(pause)

        raise OCRUnavailable(self._fail(problem))


def _parse(payload: dict, image_height: int) -> list[TextLine]:
    annotation = (payload.get("result") or {}).get("textAnnotation") or {}
    height = float(annotation.get("height") or image_height) or 1.0

    lines: list[TextLine] = []
    for block in annotation.get("blocks", []):
        for line in block.get("lines", []):
            text = (line.get("text") or "").strip()
            if not text:
                continue
            ys = [float(v.get("y", 0)) for v in line.get("boundingBox", {}).get("vertices", [])]
            size = (max(ys) - min(ys)) / height if ys else 0.0
            lines.append(TextLine(text=text, confidence=1.0, height=size))
    return lines
