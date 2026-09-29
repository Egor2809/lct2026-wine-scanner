"""Харднинг HTTP-слоя: предел тела, request-id, лог, заголовки, GZip, очередь.

Всё здесь — обвязка вокруг /scan, и главное, что проверяется, — что она
не трогает контракт: 200 и `slug` в корне на любое непустое тело.
"""

from __future__ import annotations

import asyncio
import importlib
import io
import logging
import threading
import time

import httpx
import pytest
from PIL import Image
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.routing import Mount
from starlette.staticfiles import StaticFiles
from starlette.testclient import TestClient

from wine_scanner.catalog.loader import CATALOG_IMAGES_ROUTE

app_module = importlib.import_module("wine_scanner.api.app")


def jpeg() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (240, 360), (120, 30, 40)).save(buffer, format="JPEG")
    return buffer.getvalue()


def assert_slug(response: httpx.Response) -> dict:
    assert response.status_code == 200, response.text
    body = response.json()
    assert isinstance(body.get("slug"), str) and body["slug"]
    return body


@pytest.fixture()
def scan_calls(client, monkeypatch) -> list[int]:
    """Считает вызовы модели: отказ по размеру не должен до неё доходить."""
    calls: list[int] = []
    original = client.app.state.pipeline.scan

    def counting(payload: bytes) -> dict:
        calls.append(len(payload))
        return original(payload)

    monkeypatch.setattr(client.app.state.pipeline, "scan", counting)
    return calls


# ── предел тела ───────────────────────────────────────────────────────────


def test_content_length_over_limit_is_413_before_model(client, scan_calls, monkeypatch) -> None:
    monkeypatch.setattr(app_module, "MAX_UPLOAD_BYTES", 1000)
    response = client.post("/scan", content=b"\xff" * 2000, headers={"content-type": "image/jpeg"})
    assert response.status_code == 413
    assert scan_calls == []


def test_chunked_body_over_limit_is_413(client, scan_calls, monkeypatch) -> None:
    """Без Content-Length тело читается потоком и обрывается на пределе."""
    monkeypatch.setattr(app_module, "MAX_UPLOAD_BYTES", 1000)

    def chunks():
        for _ in range(100):
            yield b"\xff" * 500

    response = client.post("/scan", content=chunks(), headers={"content-type": "image/jpeg"})
    assert response.status_code == 413
    assert scan_calls == []


def test_multipart_over_limit_is_413(client, scan_calls, monkeypatch) -> None:
    monkeypatch.setattr(app_module, "MAX_UPLOAD_BYTES", 1000)
    response = client.post("/scan", files={"file": ("a.jpg", b"\xff" * 5000, "image/jpeg")})
    assert response.status_code == 413
    assert scan_calls == []


@pytest.mark.parametrize("content_type", ["image/jpeg", "multipart/form-data; boundary=x"])
def test_endless_body_is_cut_at_limit(monkeypatch, content_type: str) -> None:
    """Бесконечное тело без Content-Length: читаем чуть больше предела и 413,
    а не копим в памяти (или во временном файле формы), пока клиент шлёт."""
    from fastapi import HTTPException
    from starlette.requests import Request

    monkeypatch.setattr(app_module, "MAX_UPLOAD_BYTES", 1000)
    pulled: list[int] = []

    head = (b'--x\r\nContent-Disposition: form-data; name="file"; filename="a.jpg"\r\n'
            b"Content-Type: image/jpeg\r\n\r\n")

    async def receive() -> dict:
        pulled.append(1)
        body = head if len(pulled) == 1 and content_type.startswith("multipart") else b""
        return {"type": "http.request", "body": body + b"\xff" * 500, "more_body": True}

    scope = {"type": "http", "method": "POST", "path": "/scan", "query_string": b"",
             "headers": [(b"content-type", content_type.encode())]}
    with pytest.raises(HTTPException) as exc:
        asyncio.run(app_module._read_payload(Request(scope, receive)))
    assert exc.value.status_code == 413
    assert len(pulled) <= 3  # предел 1000 байт — это 2–3 куска по 500


def test_chunked_body_under_limit_is_scanned(client) -> None:
    payload = jpeg()
    parts = [payload[i:i + 1000] for i in range(0, len(payload), 1000)]
    assert_slug(client.post("/scan", content=iter(parts), headers={"content-type": "image/jpeg"}))


@pytest.mark.parametrize("path", ["/scan", "/v1/eval/predict", "/predict"])
def test_contract_kept_on_all_scan_paths(client, path: str) -> None:
    assert_slug(client.post(path, files={"file": ("a.jpg", jpeg(), "image/jpeg")}))
    assert_slug(client.post(path, content=jpeg(), headers={"content-type": "text/plain"}))


# ── request-id и лог ──────────────────────────────────────────────────────


def test_request_id_generated_when_absent(client) -> None:
    first = client.get("/health").headers["x-request-id"]
    second = client.get("/health").headers["x-request-id"]
    assert len(first) == 32 and first != second


def test_request_id_echoed_when_sane(client) -> None:
    response = client.post("/scan", content=jpeg(), headers={"x-request-id": "eval-42"})
    assert_slug(response)
    assert response.headers["x-request-id"] == "eval-42"


@pytest.mark.parametrize("bad", ["a" * 200, "two words", "x;rm -rf"])
def test_request_id_replaced_when_suspicious(client, bad: str) -> None:
    """Чужой id идёт в лог: пробелы и мусор могли бы подделать поля строки."""
    returned = client.get("/health", headers={"x-request-id": bad}).headers["x-request-id"]
    assert returned != bad and len(returned) == 32


def test_scan_logs_one_structured_line(client, caplog) -> None:
    with caplog.at_level(logging.INFO, logger=app_module.log.name):
        body = assert_slug(client.post(
            "/scan", files={"image": ("a.jpg", jpeg(), "image/jpeg")},
            headers={"x-request-id": "req-1"},
        ))
    lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith("scan ")]
    assert len(lines) == 1
    line = lines[0]
    for field in ("request_id=req-1", "status=200", f"slug={body['slug']}",
                  "confident=", "elapsed_ms=", "bytes=", "error=false"):
        assert field in line, line
    # Текст этикетки — данные пользователя, в лог он не идёт.
    assert "ocr" not in line


def test_scan_logs_rejected_request(client, caplog) -> None:
    with caplog.at_level(logging.INFO, logger=app_module.log.name):
        assert client.post("/scan").status_code == 400
    line = next(r.getMessage() for r in caplog.records if r.getMessage().startswith("scan "))
    assert "status=400" in line and "error=true" in line


def test_fallback_is_logged_as_error(client, caplog) -> None:
    with caplog.at_level(logging.INFO, logger=app_module.log.name):
        body = assert_slug(client.post("/scan", content=b"not an image"))
    assert body["error"]
    line = next(r.getMessage() for r in caplog.records if r.getMessage().startswith("scan "))
    assert "status=200" in line and "error=true" in line


# ── заголовки ─────────────────────────────────────────────────────────────


def test_scan_headers(client) -> None:
    response = client.post("/scan", content=jpeg(), headers={"accept-encoding": "gzip"})
    assert_slug(response)
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["cache-control"] == "no-store"
    # /scan не сжимается: клиент, заявивший gzip и не распаковывающий ответ,
    # не должен потерять slug.
    assert "content-encoding" not in response.headers


def test_wines_revalidates_and_is_gzipped(client) -> None:
    response = client.get("/wines", params={"limit": 6}, headers={"accept-encoding": "gzip"})
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-cache"
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers.get("content-encoding") == "gzip"
    assert response.json()["items"]  # httpx распаковал — тело целое


def test_small_json_not_gzipped(client) -> None:
    response = client.get("/health", headers={"accept-encoding": "gzip"})
    assert "content-encoding" not in response.headers
    assert response.headers["cache-control"] == "no-store"


def test_frontend_revalidates_with_referrer_policy(client) -> None:
    response = client.get("/")
    if response.status_code == 404:
        pytest.skip("frontend/ не найден")
    assert response.headers["content-type"].startswith("text/html")
    assert response.headers["cache-control"] == "no-cache"
    assert response.headers["referrer-policy"] == "strict-origin-when-cross-origin"


def test_json_has_no_referrer_policy(client) -> None:
    assert "referrer-policy" not in client.get("/health").headers


def test_cors_exposes_request_id(client) -> None:
    response = client.get("/health", headers={"origin": "https://vino-svoe.ru"})
    assert "x-request-id" in response.headers.get("access-control-expose-headers", "").lower()


def test_catalog_images_cached_forever_but_not_404(tmp_path) -> None:
    (tmp_path / "wine.jpg").write_bytes(jpeg())
    images = Starlette(
        routes=[Mount(CATALOG_IMAGES_ROUTE, StaticFiles(directory=tmp_path))],
        middleware=[Middleware(app_module.ResponseHeadersMiddleware)],
    )
    with TestClient(images) as client:
        found = client.get(f"{CATALOG_IMAGES_ROUTE}/wine.jpg")
        assert found.status_code == 200
        assert found.headers["cache-control"] == app_module.CACHE_IMMUTABLE
        missing = client.get(f"{CATALOG_IMAGES_ROUTE}/none.jpg")
        assert missing.status_code == 404
        assert missing.headers["cache-control"] == "no-store"


# ── очередь к модели ──────────────────────────────────────────────────────


@pytest.mark.parametrize("slots", [1, 2])
def test_scan_concurrency_is_limited(client, monkeypatch, slots: int) -> None:
    pipeline = client.app.state.pipeline
    original = pipeline.scan
    lock = threading.Lock()
    state = {"now": 0, "peak": 0}

    def slow(payload: bytes) -> dict:
        with lock:
            state["now"] += 1
            state["peak"] = max(state["peak"], state["now"])
        time.sleep(0.15)
        with lock:
            state["now"] -= 1
        return original(payload)

    monkeypatch.setattr(pipeline, "scan", slow)

    async def burst() -> list[httpx.Response]:
        # Семафор создаётся внутри цикла — как в lifespan.
        monkeypatch.setattr(client.app.state, "scan_slots", asyncio.Semaphore(slots),
                            raising=False)
        transport = httpx.ASGITransport(app=client.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as http:
            return await asyncio.gather(*(
                http.post("/scan", content=jpeg(), headers={"content-type": "image/jpeg"})
                for _ in range(4)
            ))

    for response in asyncio.run(burst()):
        assert_slug(response)
    assert state["peak"] == slots
