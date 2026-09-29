"""Устойчивость к плохим входам и отказам облачного OCR.

Упавший или зависший запрос на проверке стоит дороже промаха: скрипт
кейсодержателя может оборвать прогон, и 50 баллов за точность обнулятся.
Правило: непустое тело -> 200 и slug в корне; 4xx — только невалидный запрос.
"""

from __future__ import annotations

import io
import logging
import time

import httpx
import numpy as np
import pytest
from PIL import Image

from wine_scanner.features.text import (
    FallbackExtractor,
    NullExtractor,
    TextLine,
    build_text_extractor,
)
from wine_scanner.features.text.yandex_ocr import OCRUnavailable, YandexOCRExtractor

SLA = 3.0


def encode(image: Image.Image, fmt: str, **kwargs) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format=fmt, **kwargs)
    return buffer.getvalue()


def photo(mode: str = "RGB") -> Image.Image:
    rng = np.random.default_rng(0)
    pixels = rng.integers(0, 255, (360, 240, 3), dtype=np.uint8)
    image = Image.fromarray(pixels)
    return image.convert(mode) if mode != "RGB" else image


def assert_slug(response: httpx.Response) -> dict:
    assert response.status_code == 200, response.text
    body = response.json()
    assert isinstance(body.get("slug"), str) and body["slug"]
    return body


# ── плохие входы ──────────────────────────────────────────────────────────


def test_heic(client) -> None:
    pillow_heif = pytest.importorskip("pillow_heif")
    buffer = io.BytesIO()
    pillow_heif.from_pillow(photo()).save(buffer, format="HEIF")
    body = assert_slug(client.post("/scan", files={"image": ("a.heic", buffer.getvalue(), "image/heic")}))
    assert "error" not in body


@pytest.mark.parametrize("mode", ["RGBA", "LA", "P", "L", "CMYK", "I;16"])
def test_unusual_color_modes(client, mode: str) -> None:
    fmt = "JPEG" if mode == "CMYK" else "PNG"
    payload = encode(photo(mode) if mode != "I;16" else photo("L").convert("I;16"), fmt)
    body = assert_slug(client.post("/scan", files={"image": (f"a.{fmt.lower()}", payload, "image/png")}))
    assert "error" not in body


def test_png_with_alpha(client) -> None:
    image = photo("RGBA")
    image.putalpha(0)  # полностью прозрачная
    body = assert_slug(client.post("/scan", files={"image": ("a.png", encode(image, "PNG"), "image/png")}))
    assert "error" not in body


def test_50mb_photo(client) -> None:
    # Хвост после маркера конца JPEG декодеры игнорируют: так файл весит
    # 50 МБ без минуты на генерацию шума. Время декодирования огромного
    # кадра проверяет test_huge_jpeg_decodes_fast.
    payload = encode(photo(), "JPEG") + b"\0" * (50 * 1024 * 1024)
    body = assert_slug(client.post("/scan", files={"image": ("big.jpg", payload, "image/jpeg")}))
    assert "error" not in body


def test_huge_jpeg_decodes_fast() -> None:
    from wine_scanner.normalize.image import decode

    payload = encode(Image.new("RGB", (8000, 6000), (120, 30, 40)), "JPEG", quality=95)
    started = time.perf_counter()
    image = decode(payload, max_side=1600)
    assert time.perf_counter() - started < 1.0
    assert min(image.size) >= 1600  # уменьшили, но не ниже рабочего размера


def test_normal_photo_is_not_downscaled_on_decode() -> None:
    """DCT-уменьшение не должно трогать обычные 12 Мп фото: иначе поедет точность."""
    from wine_scanner.normalize.image import decode

    payload = encode(Image.new("RGB", (4032, 3024), (120, 30, 40)), "JPEG")
    assert decode(payload, max_side=1600).size == (4032, 3024)


def test_truncated_jpeg_is_recognised(client) -> None:
    payload = encode(photo(), "JPEG")
    body = assert_slug(client.post("/scan", files={"image": ("cut.jpg", payload[: len(payload) // 2], "image/jpeg")}))
    assert "error" not in body


@pytest.mark.parametrize("payload", [b"not an image", b"\xff\xd8\xff" + b"\0" * 100, b"GIF89a"])
def test_garbage_gets_fallback_slug(client, payload: bytes) -> None:
    body = assert_slug(client.post("/scan", files={"image": ("x.jpg", payload, "image/jpeg")}))
    assert body["error"] and body["confidence"] == 0.0


def test_decompression_bomb_gets_fallback_slug(client) -> None:
    """120 Мп в PNG на ~15 КБ: между пределом и двойным Pillow лишь предупреждал."""
    payload = encode(Image.new("1", (12000, 10000)), "PNG")
    body = assert_slug(client.post("/scan", files={"image": ("bomb.png", payload, "image/png")}))
    assert body["error"] and body["confidence"] == 0.0


@pytest.mark.parametrize("size", [(8064, 6048), (8000, 10000)])
def test_big_real_photos_pass_bomb_limit(size) -> None:
    """48 Мп iPhone и самое большое фото датасета кейса (80 Мп) — не бомба."""
    from wine_scanner.normalize.image import decode

    payload = encode(Image.new("L", size, 128), "JPEG")
    assert min(decode(payload, max_side=1600).size) >= 1600


def test_photo_without_label(client) -> None:
    payload = encode(Image.new("RGB", (640, 480), (90, 90, 90)), "JPEG")
    assert_slug(client.post("/scan", files={"image": ("wall.jpg", payload, "image/jpeg")}))


def test_empty_body_is_4xx(client) -> None:
    assert client.post("/scan").status_code == 400
    assert client.post("/scan", content=b"", headers={"content-type": "image/jpeg"}).status_code == 400
    assert client.post("/scan", files={"image": ("e.jpg", b"", "image/jpeg")}).status_code == 400


@pytest.mark.parametrize(
    "content_type",
    ["application/json", "text/plain", "application/octet-stream", "image/webp"],
)
def test_wrong_content_type_raw_body(client, content_type: str) -> None:
    payload = encode(photo(), "JPEG")
    assert_slug(client.post("/scan", content=payload, headers={"content-type": content_type}))


def test_wrong_part_content_type(client) -> None:
    payload = encode(photo(), "JPEG")
    assert_slug(client.post("/scan", files={"image": ("photo", payload, "application/pdf")}))


def test_eval_alias(client) -> None:
    payload = encode(photo(), "JPEG")
    assert_slug(client.post("/v1/eval/predict", files={"image": ("a.jpg", payload, "image/jpeg")}))


def test_pipeline_crash_is_not_500(client, monkeypatch) -> None:
    def boom(payload: bytes) -> dict:
        raise RuntimeError("модель упала")

    monkeypatch.setattr(client.app.state.pipeline, "scan", boom)
    body = assert_slug(client.post("/scan", files={"image": ("a.jpg", encode(photo(), "JPEG"), "image/jpeg")}))
    assert body["error"] and "модель упала" not in body["error"]


# ── отказы Yandex OCR ─────────────────────────────────────────────────────

OK_BODY = {
    "result": {
        "textAnnotation": {
            "height": 100,
            "blocks": [{"lines": [{"text": "Красностоп", "boundingBox": {
                "vertices": [{"x": 0, "y": 10}, {"x": 50, "y": 30}]}}]}],
        }
    }
}


class FakeClock:
    """Виртуальное время: тесты таймаутов не спят по-настоящему."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


def cloud(handler, clock: FakeClock | None = None, **kwargs) -> tuple[YandexOCRExtractor, list, FakeClock]:
    clock = clock or FakeClock()
    calls: list[httpx.Request] = []

    def record(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return handler(request, clock)

    extractor = YandexOCRExtractor(
        api_key="test", folder_id="f", transport=httpx.MockTransport(record),
        clock=clock, sleep=clock.sleep, **{"min_interval": 0.0, **kwargs},
    )
    return extractor, calls, clock


IMAGE = Image.new("RGB", (64, 64), (200, 180, 160))


def test_ocr_success() -> None:
    extractor, calls, _ = cloud(lambda r, c: httpx.Response(200, json=OK_BODY))
    lines = extractor.extract(IMAGE)
    assert [line.text for line in lines] == ["Красностоп"]
    assert calls[0].headers["Authorization"] == "Api-Key test"


def test_ocr_network_down_stays_within_deadline_then_opens_circuit() -> None:
    def down(request, clock):
        clock.now += 0.05
        raise httpx.ConnectError("нет сети", request=request)

    extractor, calls, clock = cloud(down)
    started = clock.now
    with pytest.raises(OCRUnavailable):
        extractor.extract(IMAGE)
    assert len(calls) == 3  # попытка + два повтора
    assert clock.now - started <= extractor.deadline

    # Второй сбой подряд размыкает предохранитель.
    with pytest.raises(OCRUnavailable):
        extractor.extract(IMAGE)
    assert len(calls) == 6

    # Следующие запросы не платят дедлайн: сразу в запасной канал.
    with pytest.raises(OCRUnavailable):
        extractor.extract(IMAGE)
    assert len(calls) == 6

    # После cooldown облако пробуется снова.
    clock.now += extractor.cooldown + 1
    with pytest.raises(OCRUnavailable):
        extractor.extract(IMAGE)
    assert len(calls) == 9


def test_ocr_single_failure_does_not_disable_cloud() -> None:
    """Случайный таймаут — запасной OCR только для этого фото, а не на 30 с."""
    responses = iter([httpx.Response(503)] * 3 + [httpx.Response(200, json=OK_BODY)])
    extractor, calls, _ = cloud(lambda r, c: next(responses))
    with pytest.raises(OCRUnavailable):
        extractor.extract(IMAGE)
    assert extractor.extract(IMAGE)[0].text == "Красностоп"


def test_ocr_min_interval_spaces_requests() -> None:
    """Квота ~1 запрос/с: серия фото подряд не должна ловить 429."""
    starts: list[float] = []

    def ok(request, clock):
        starts.append(clock.now)
        clock.now += 0.3
        return httpx.Response(200, json=OK_BODY)

    extractor, _, _ = cloud(ok, min_interval=1.0)
    for _ in range(3):
        extractor.extract(IMAGE)
    assert [round(b - a, 6) for a, b in zip(starts, starts[1:], strict=False)] == [1.0, 1.0]


def test_ocr_timeout_stays_within_deadline() -> None:
    def hang(request, clock):
        timeout = request.extensions["timeout"]["read"]
        clock.now += timeout
        raise httpx.ReadTimeout("таймаут", request=request)

    extractor, calls, clock = cloud(hang)
    started = clock.now
    with pytest.raises(OCRUnavailable):
        extractor.extract(IMAGE)
    assert clock.now - started <= extractor.deadline + 1e-9
    # Вторая попытка получила только остаток дедлайна, а не полный таймаут.
    assert calls[1].extensions["timeout"]["read"] < extractor.attempt_timeout


@pytest.mark.parametrize("status", [401, 403])
def test_ocr_bad_key_disables_cloud_until_restart(status: int) -> None:
    extractor, calls, clock = cloud(lambda r, c: httpx.Response(status, json={"error": "x"}))
    with pytest.raises(OCRUnavailable):
        extractor.extract(IMAGE)
    assert len(calls) == 1  # без повторов: ключ сам не починится
    clock.now += 10_000
    with pytest.raises(OCRUnavailable):
        extractor.extract(IMAGE)
    assert len(calls) == 1


def test_ocr_429_retried_with_backoff() -> None:
    responses = iter([httpx.Response(429), httpx.Response(200, json=OK_BODY)])
    extractor, calls, clock = cloud(lambda r, c: next(responses))
    started = clock.now
    assert extractor.extract(IMAGE)[0].text == "Красностоп"
    assert len(calls) == 2
    assert clock.now - started == pytest.approx(extractor.backoff[0])


def test_ocr_429_persistent_short_cooldown() -> None:
    extractor, calls, clock = cloud(lambda r, c: httpx.Response(429))
    with pytest.raises(OCRUnavailable):
        extractor.extract(IMAGE)
    with pytest.raises(OCRUnavailable):
        extractor.extract(IMAGE)
    assert len(calls) == 6  # второй сбой подряд — предохранитель
    clock.now += 3.1  # квота отпускает быстро — и предохранитель короткий
    with pytest.raises(OCRUnavailable):
        extractor.extract(IMAGE)
    assert len(calls) == 9


def test_ocr_5xx_retried() -> None:
    responses = iter([httpx.Response(503), httpx.Response(200, json=OK_BODY)])
    extractor, calls, _ = cloud(lambda r, c: next(responses))
    assert extractor.extract(IMAGE)
    assert len(calls) == 2


class LocalOCR:
    name = "local"

    def extract(self, image):
        return [TextLine(text="локально")]


def test_fallback_to_local_ocr_on_cloud_failure() -> None:
    def down(request, clock):
        raise httpx.ConnectError("нет сети", request=request)

    extractor, _, _ = cloud(down)
    lines = FallbackExtractor(extractor, LocalOCR()).extract(IMAGE)
    assert [line.text for line in lines] == ["локально"]


def _linux(monkeypatch, rapid: bool) -> None:
    """Как на Linux: Apple Vision нет; RapidOCR — есть или нет."""
    from wine_scanner.features.text import rapid_ocr
    from wine_scanner.features.text.apple_vision import AppleVisionExtractor

    monkeypatch.setattr(AppleVisionExtractor, "available", staticmethod(lambda: False))
    monkeypatch.setattr(rapid_ocr.RapidOCRExtractor, "available", staticmethod(lambda: rapid))
    monkeypatch.setattr(rapid_ocr.RapidOCRExtractor, "__init__", lambda self, **kw: None)


def test_fallback_on_linux_is_local_ocr(monkeypatch) -> None:
    from wine_scanner.features.text.rapid_ocr import RapidOCRExtractor

    _linux(monkeypatch, rapid=True)
    extractor = build_text_extractor({"kind": "auto"}, api_key="k")
    assert isinstance(extractor, FallbackExtractor)
    assert isinstance(extractor.fallback, RapidOCRExtractor)


def test_fallback_without_local_ocr_is_image_only(monkeypatch) -> None:
    _linux(monkeypatch, rapid=False)
    extractor = build_text_extractor({"kind": "auto"}, api_key="k")
    assert isinstance(extractor.fallback, NullExtractor)


@pytest.mark.parametrize("kind", ["auto", "yandex"])
def test_no_key_starts_in_contour(monkeypatch, caplog, kind: str) -> None:
    """Без ключа сервис поднимается «в контуре» — на локальном OCR, с предупреждением."""
    from wine_scanner.features.text.rapid_ocr import RapidOCRExtractor

    _linux(monkeypatch, rapid=True)
    with caplog.at_level(logging.WARNING):
        extractor = build_text_extractor({"kind": kind}, api_key="")
    assert isinstance(extractor, RapidOCRExtractor)
    assert "YANDEX_API_KEY" in caplog.text


def test_scan_within_sla_when_cloud_hangs(client, monkeypatch) -> None:
    """Сквозной сценарий: облако висит по-настоящему, запрос всё равно < SLA."""

    def hang(request: httpx.Request) -> httpx.Response:
        time.sleep(request.extensions["timeout"]["read"])
        raise httpx.ReadTimeout("таймаут", request=request)

    extractor = YandexOCRExtractor(api_key="k", transport=httpx.MockTransport(hang))
    pipeline = client.app.state.pipeline
    monkeypatch.setattr(pipeline, "text_extractor", FallbackExtractor(extractor, NullExtractor()))

    payload = encode(photo(), "JPEG")
    for _ in range(3):  # третий запрос идёт через разомкнутый предохранитель
        started = time.perf_counter()
        assert_slug(client.post("/scan", files={"image": ("a.jpg", payload, "image/jpeg")}))
        assert time.perf_counter() - started < SLA


def test_raw_body_sent_as_urlencoded_form(client) -> None:
    """`curl --data-binary @photo.jpg` без заголовка помечает тело как форму."""
    payload = encode(photo(), "JPEG")
    response = client.post(
        "/scan", content=payload,
        headers={"content-type": "application/x-www-form-urlencoded"},
    )
    assert_slug(response)


def test_ocr_hard_timeout_when_client_ignores_timeout() -> None:
    """DNS-зависание: httpx не соблюдает таймаут, облако всё равно отпускает запрос."""

    def stuck(request: httpx.Request) -> httpx.Response:
        time.sleep(5)
        return httpx.Response(200, json=OK_BODY)

    extractor = YandexOCRExtractor(api_key="k", transport=httpx.MockTransport(stuck),
                                   deadline=0.6, attempt_timeout=0.5)
    started = time.perf_counter()
    with pytest.raises(OCRUnavailable):
        extractor.extract(IMAGE)
    assert time.perf_counter() - started < 1.0


def test_scan_within_sla_when_any_ocr_hangs(client, monkeypatch) -> None:
    """Запасной OCR тоже может зависнуть: бюджет текста в пайплайне держит SLA."""

    class Stuck:
        name = "stuck"

        def extract(self, image):
            time.sleep(5)
            return []

    pipeline = client.app.state.pipeline
    monkeypatch.setattr(pipeline, "text_extractor", Stuck())
    # Значение бюджета — из конфига (в контуре 6 с); проверяем механизм, не число.
    monkeypatch.setitem(pipeline.config["text"], "budget", 1.0)
    started = time.perf_counter()
    assert_slug(client.post("/scan", files={"image": ("a.jpg", encode(photo(), "JPEG"), "image/jpeg")}))
    assert time.perf_counter() - started < SLA
