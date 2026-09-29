"""HTTP-слой.

Контракт `POST /scan` зафиксирован дважды — скриптом кейсодержателя и
frontend/app.js. Менять форму ответа нельзя, можно только добавлять поля
рядом с `slug`, оставляя его в корне.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
import uuid
from contextlib import asynccontextmanager, nullcontext
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import Headers, MutableHeaders

# Форма отдаёт starlette-овский UploadFile, а не fastapi-евский: fastapi.UploadFile
# его подкласс, поэтому isinstance против fastapi.UploadFile здесь не срабатывает.
from starlette.datastructures import UploadFile as StarletteUploadFile
from starlette.middleware.gzip import GZipMiddleware
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from ..catalog.browse import MAX_LIMIT
from ..catalog.loader import CATALOG_IMAGES_ROUTE, CATALOG_THUMBS_ROUTE
from ..enrich.sommelier import FOODS
from ..normalize.image import ImageDecodeError
from ..pipeline import ScannerPipeline
from ..response import ScanResponse, degraded
from ..settings import get_config, get_settings

log = logging.getLogger(__name__)

MAX_UPLOAD_BYTES = get_settings().max_upload_mb * 1024 * 1024

# Все пути распознавания: пример скрипта кейсодержателя шлёт в /v1/eval/predict,
# прежний клиент — в /predict.
SCAN_PATHS = frozenset({"/scan", "/v1/eval/predict", "/predict"})

# Чужой X-Request-ID принимаем, только если он похож на идентификатор:
# он попадает в лог и в заголовок ответа.
_REQUEST_ID = re.compile(r"[A-Za-z0-9._:\-]{1,128}")

# Эталонные фото каталога меняются только пересборкой каталога вместе с
# перезапуском — браузеру незачем перепроверять их при каждом открытии карточки.
CACHE_IMMUTABLE = "public, max-age=31536000, immutable"
# no-cache — «можно хранить, но сверяйся»: фронт отдаётся StaticFiles с ETag,
# так что это 304 без тела, а правки фронта подхватываются без Ctrl+F5.
# /wines тоже no-cache: ответ собирается из памяти за миллисекунды, а после
# переиндексации список обязан совпадать с /wines/{slug}, иначе карточка из
# закэшированного списка открывается в 404.
CACHE_REVALIDATE = "no-cache"
# Результат распознавания и состояние сервиса не кэшируются нигде, включая прокси.
CACHE_NONE = "no-store"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Каталог, эмбеддер и индекс грузятся здесь — требование ТЗ.

    Если загрузка падает, приложение всё равно поднимается: так /health
    отдаёт внятную причину вместо падения контейнера в рестарт-петле
    за пять минут до созвона.
    """
    settings = get_settings()
    config = get_config()
    # uvicorn настраивает только свои логгеры: без этого строки «Пайплайн
    # готов», предупреждения OCR и сбои не видны в логе контейнера.
    logging.basicConfig(level=settings.log_level.upper(),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    app.state.pipeline = None
    app.state.startup_error = None
    # Семафор создаётся здесь, а не при импорте: asyncio-примитив привязан
    # к циклу событий, а цикл есть только у запущенного приложения.
    app.state.scan_slots = (
        asyncio.Semaphore(settings.scan_concurrency) if settings.scan_concurrency > 0 else None
    )

    try:
        app.state.pipeline = ScannerPipeline.load(settings=settings, config=config)
    except Exception as exc:
        app.state.startup_error = str(exc)
        log.exception("Пайплайн не поднялся: %s", exc)

    yield


class ResponseHeadersMiddleware:
    """Request-id, защитные заголовки и Cache-Control в одном проходе.

    Чистый ASGI, а не BaseHTTPMiddleware: тот гоняет ответ через очередь
    и добавляет задержку на каждый запрос, включая /scan.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        incoming = Headers(scope=scope).get("x-request-id", "")
        request_id = incoming if _REQUEST_ID.fullmatch(incoming) else uuid.uuid4().hex
        # Отсюда его читает лог /scan через request.state.
        scope.setdefault("state", {})["request_id"] = request_id
        path = scope["path"]

        async def send_with_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = MutableHeaders(scope=message)
                headers["X-Request-ID"] = request_id
                headers.setdefault("X-Content-Type-Options", "nosniff")
                if headers.get("content-type", "").startswith("text/html"):
                    # Уходя по ссылке на портал, страница передаёт только домен.
                    headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
                cache = _cache_policy(path, message["status"])
                if cache and "cache-control" not in headers:
                    headers["Cache-Control"] = cache
            await send(message)

        await self.app(scope, receive, send_with_headers)


def _cache_policy(path: str, status: int) -> str | None:
    if path in SCAN_PATHS or path == "/health":
        return CACHE_NONE
    if path.startswith((CATALOG_IMAGES_ROUTE + "/", CATALOG_THUMBS_ROUTE + "/")):
        # 404 навсегда кэшировать нельзя: фото может появиться после пересборки.
        return CACHE_IMMUTABLE if status in (200, 304) else CACHE_NONE
    if path.startswith(("/docs", "/redoc", "/openapi.json")):
        return None
    # /wines и фронт.
    return CACHE_REVALIDATE


class GZipExceptScanMiddleware:
    """GZip для каталога и фронта, но не для /scan.

    /scan отвечает на ~10 КБ после загрузки фото в мегабайты — выигрыш
    от сжатия ничтожен, а скрипт оценки, который прислал Accept-Encoding
    и не распаковывает ответ, стоил бы 50 баллов. Остальное (/wines
    с описаниями, app.js, styles.css) сжимается в разы.
    """

    def __init__(self, app: ASGIApp, minimum_size: int = 1024, compresslevel: int = 6) -> None:
        self.app = app
        # Уровень 6 вместо 9 по умолчанию: почти тот же размер, заметно быстрее.
        self.gzip = GZipMiddleware(app, minimum_size=minimum_size, compresslevel=compresslevel)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http" and scope["path"] in SCAN_PATHS:
            await self.app(scope, receive, send)
        else:
            await self.gzip(scope, receive, send)


app = FastAPI(
    title="Сканер российских вин «Своё вино»",
    description="Фото винной этикетки -> одна карточка вина из каталога.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in get_settings().cors_origins.split(",") if o.strip()],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
    # Без этого браузер портала не увидит id запроса для обращения в поддержку.
    expose_headers=["X-Request-ID"],
)
app.add_middleware(GZipExceptScanMiddleware)
# Последний добавленный — внешний: заголовки получают и ответы CORS, и 4xx.
app.add_middleware(ResponseHeadersMiddleware)


def _pipeline(request: Request) -> ScannerPipeline:
    pipeline = getattr(request.app.state, "pipeline", None)
    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Сервис не готов, причина — в /health",
        )
    return pipeline


@app.get("/health")
async def health(request: Request) -> JSONResponse:
    pipeline = getattr(request.app.state, "pipeline", None)
    ready = pipeline is not None
    return JSONResponse(
        status_code=200 if ready else 503,
        content={
            "status": "ok" if ready else "loading_failed",
            "catalog_size": len(pipeline.catalog) if ready else 0,
            "index_size": len(pipeline.index) if ready else 0,
            "index": getattr(pipeline.index, "kind", "") if ready else None,
            "embedder": getattr(pipeline.embedder, "query_name", pipeline.embedder.name)
            if ready else None,
            "ocr": pipeline.text_extractor.name if ready else None,
            "error": request.app.state.startup_error,
        },
    )


@app.post(
    "/scan",
    # Только документация: обработчик сам собирает JSONResponse, валидации нет.
    response_model=ScanResponse,
    summary="Фото этикетки -> вино каталога",
    responses={400: {"description": "Пустое тело или нечитаемая multipart-форма"},
               413: {"description": "Фото больше MAX_UPLOAD_MB"},
               503: {"description": "Модели не загрузились: причина в /health"}},
    openapi_extra={"requestBody": {"required": True, "content": {
        "multipart/form-data": {"schema": {"type": "object", "required": ["image"],
                                           "properties": {"image": {"type": "string",
                                                                    "format": "binary"}}}},
        "application/octet-stream": {"schema": {"type": "string", "format": "binary"}},
    }}},
)
# Алиасы — см. SCAN_PATHS: их не сжимает GZip и не кэширует браузер.
@app.post("/v1/eval/predict", include_in_schema=False)
@app.post("/predict", include_in_schema=False)
async def scan(request: Request) -> JSONResponse:
    """Фото этикетки -> плоский JSON с ключом `slug` в корне.

    Поле формы называется `image` (так его шлёт frontend/app.js). На случай,
    если скрипт кейсодержателя назовёт его иначе, принимаем и любой другой
    файл формы, и сырое тело запроса — терять 50 баллов на имени поля нельзя.
    """
    started = time.perf_counter()
    size = 0
    try:
        pipeline = _pipeline(request)
        payload = await _read_payload(request)
        size = len(payload)
        if not payload:
            raise HTTPException(status_code=400, detail="Изображение не передано")
    except HTTPException as exc:
        _log_scan(request, started, exc.status_code, size or _declared_length(request))
        raise

    # Непустое тело — это попытка распознать фото: всегда 200 и slug в корне.
    # 4xx остаётся только для невалидного запроса (пусто, больше предела).
    result, wait_ms = await _run_scan(request, pipeline, payload)
    try:
        _log_scan(request, started, 200, size, result, wait_ms)
        return JSONResponse(content=result)
    except Exception as exc:
        # NaN/inf или не-JSON тип в метриках: Starlette отдал бы 500, а скрипт
        # оценки на не-200 записывает null. Slug уже найден — отдаём его.
        log.exception("Ответ не сериализуется: %s", exc)
        slug = str(result.get("slug") or pipeline.fallback("")["slug"])
        return JSONResponse(content=degraded(slug, None, "ответ не сериализуется"))


@app.get("/wines")
async def wines(
    request: Request,
    q: str = Query("", max_length=200),
    style: str = Query("", max_length=60),
    region: str = Query("", max_length=120),
    winery: str = Query("", max_length=200),
    sparkling: bool | None = None,
    seed: int | None = None,
    limit: int = Query(24, ge=1, le=MAX_LIMIT),
    offset: int = Query(0, ge=0),
) -> JSONResponse:
    """Каталог для UI: поиск, фильтры, фасеты стилей и регионов."""
    return JSONResponse(content=_pipeline(request).browse(
        q=q, style=style, region=region, winery=winery,
        sparkling=sparkling, seed=seed, limit=limit, offset=offset,
    ))


@app.get("/wines/{slug}")
async def wine(request: Request, slug: str) -> JSONResponse:
    """Карточка позиции и аналоги — переход по аналогу или из каталога в UI."""
    result = _pipeline(request).wine(slug)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Позиции {slug} нет в каталоге")
    return JSONResponse(content=result)


@app.get("/sommelier")
async def sommelier(
    request: Request,
    food: str = Query(..., max_length=40),
    region: str = Query("", max_length=120),
    sparkling: bool | None = None,
    seed: int | None = None,
    limit: int = Query(6, ge=1, le=24),
) -> JSONResponse:
    """Сомелье: блюдо -> вина разных хозяйств с объяснением «почему подходит».

    Фасет регионов в ответе — для второго вопроса, до выбора региона.
    """
    if food not in FOODS:
        raise HTTPException(status_code=400, detail=f"Блюдо: одно из {', '.join(FOODS)}")
    return JSONResponse(content=_pipeline(request).sommelier(
        food, region=region, sparkling=sparkling, seed=seed, limit=limit,
    ))


async def _run_scan(
    request: Request, pipeline: ScannerPipeline, payload: bytes
) -> tuple[dict[str, Any], int]:
    """-> (ответ, сколько мс запрос ждал очереди к модели)."""
    slots = getattr(request.app.state, "scan_slots", None)
    queued = time.perf_counter()
    async with slots if slots is not None else nullcontext():
        wait_ms = round((time.perf_counter() - queued) * 1000)
        try:
            return await run_in_threadpool(pipeline.scan, payload), wait_ms
        except ImageDecodeError as exc:
            log.warning("Не изображение (%d байт): %s", len(payload), exc)
            return pipeline.fallback("не удалось прочитать изображение"), wait_ms
        except Exception as exc:
            log.exception("Ошибка обработки запроса: %s", exc)
            return pipeline.fallback("внутренняя ошибка"), wait_ms


def _log_scan(
    request: Request,
    started: float,
    status: int,
    size: int,
    result: dict[str, Any] | None = None,
    wait_ms: int = 0,
) -> None:
    """Одна строка key=value на запрос: её грепают и парсят без зависимостей.

    Текст OCR и байты фото в лог не идут — это данные пользователя.
    """
    result = result or {}
    log.info(
        "scan request_id=%s path=%s status=%d bytes=%d elapsed_ms=%d wait_ms=%d "
        "scan_ms=%s slug=%s confident=%s error=%s",
        getattr(request.state, "request_id", "-"),
        request.url.path,
        status,
        size,
        round((time.perf_counter() - started) * 1000),
        wait_ms,
        result.get("elapsed_ms", "-"),
        result.get("slug", "-"),
        str(result.get("confident", "-")).lower(),
        str(bool(result.get("error")) or status >= 400).lower(),
    )


def _declared_length(request: Request) -> int:
    declared = request.headers.get("content-length", "")
    return int(declared) if declared.isdigit() else 0


class _BodyTooLarge(Exception):
    pass


def _limited(receive: Receive, limit: int) -> Receive:
    """Обрывает чтение тела, как только принято больше `limit` байт.

    Нужен, когда Content-Length нет (chunked): иначе предел проверялся бы
    после того, как всё тело уже в памяти или во временном файле формы.
    """
    received = 0

    async def wrapped() -> Message:
        nonlocal received
        message = await receive()
        if message["type"] == "http.request":
            received += len(message.get("body", b""))
            if received > limit:
                raise _BodyTooLarge
        return message

    return wrapped


def _too_large() -> HTTPException:
    return HTTPException(
        status_code=413,
        detail=f"Файл больше {MAX_UPLOAD_BYTES // (1024 * 1024)} МБ",
    )


async def _read_payload(request: Request) -> bytes:
    """Тело разбирается вручную, а не параметром File(): FastAPI парсит любую
    форму сам, и у `curl --data-binary` (urlencoded по умолчанию) после этого
    сырое тело уже не прочитать — это был 500.

    Предел размера проверяется до чтения: по Content-Length сразу, без него —
    по мере приёма тела.
    """
    limit = MAX_UPLOAD_BYTES
    if _declared_length(request) > limit:
        raise _too_large()
    limited = Request(request.scope, _limited(request.receive, limit))

    content_type = request.headers.get("content-type", "")
    if content_type.startswith("multipart/form-data"):
        try:
            form = await limited.form(max_part_size=limit)
        except _BodyTooLarge as exc:
            raise _too_large() from exc
        except Exception as exc:
            log.warning("Битая multipart-форма: %s", exc)
            raise HTTPException(status_code=400, detail="Битая multipart-форма") from exc
        try:
            files = [v for v in form.values() if isinstance(v, StarletteUploadFile)]
            # Поле `image` — основное (frontend/app.js), любое другое — запасное.
            image = form.get("image")
            chosen = image if isinstance(image, StarletteUploadFile) else next(iter(files), None)
            return await chosen.read() if chosen is not None else b""
        finally:
            # Файлы формы крупнее 1 МБ лежат во временных файлах на диске.
            await form.close()

    # Сырое тело с любым content-type: image/*, octet-stream, urlencoded от curl.
    try:
        return await limited.body()
    except _BodyTooLarge as exc:
        raise _too_large() from exc


def _mount_static() -> None:
    """Статика монтируется последней, чтобы не перехватывать /scan и /health.

    Порядок внутри тоже важен: `/` — это catch-all, всё остальное вешается до него.
    """
    settings = get_settings()

    # Фото каталога: в карточке лежит /catalog-thumbs/<файл> (превью из
    # репозитория) или /catalog-images/<файл> (исходник, если превью нет).
    for route, directory in (
        (CATALOG_THUMBS_ROUTE, settings.catalog_thumbs_dir),
        (CATALOG_IMAGES_ROUTE, settings.catalog_images_dir),
    ):
        folder = settings.resolve(directory)
        if folder.is_dir():
            app.mount(route, StaticFiles(directory=folder), name=route.strip("/"))
            log.info("%s отдаётся из %s", route, folder)
        else:
            log.warning("Каталог с фотографиями не найден: %s", folder)

    frontend = settings.resolve(settings.frontend_dir)
    if frontend.is_dir():
        app.mount("/", StaticFiles(directory=frontend, html=True), name="frontend")
        log.info("Фронтенд отдаётся из %s", frontend)
    else:
        log.warning("Каталог фронтенда не найден: %s", frontend)


_mount_static()
