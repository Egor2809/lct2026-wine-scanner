"""Оркестрация: байты фото -> ответ сервиса.

Здесь нет ни одной строчки предметной логики — только порядок вызовов.
Любое «а давай тут быстро поправим скор» относится к search/ или features/.

Порядок на запрос:
  decode + кроп этикетки (normalize/)
    ├─ визуальный вектор -> top-k по индексу        (features/, search/)
    └─ OCR этикетки -> текстовый скор по каталогу   (features/text/, search/rerank/)
  -> пул кандидатов -> слияние сигналов -> confidence -> карточка + аналоги
     (уверенно — от найденного вина, иначе — от профиля топ-кандидатов и OCR)
Визуальная и текстовая ветки идут параллельно: латентность = max, а не сумма.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout
from typing import Any

from . import response
from .catalog.browse import browse
from .catalog.loader import Catalog, load_catalog
from .catalog.models import Wine
from .enrich.analogs import PROFILE_TOP, Analog, match_analogs, out_of_catalog
from .enrich.card import card_for
from .enrich.sommelier import pick as sommelier_pick
from .features.embedder import Embedder, build_embedder, encode_query
from .features.text import NullExtractor, TextExtractor, build_text_extractor
from .normalize.query import QueryNormalizer
from .search.confidence import decide, estimate
from .search.index import VectorIndex, load_index
from .search.rerank.fusion import rank_candidates
from .search.rerank.geometric import GeometricVerifier, build_verifier
from .search.rerank.text_match import TextMatcher
from .settings import ROOT, Config, Settings, get_config, get_settings

log = logging.getLogger(__name__)


def assert_index_matches(index: VectorIndex, embedder: Embedder) -> None:
    """Индекс и эмбеддер обязаны быть одного происхождения.

    Смена эмбеддера без пересборки индекса не вызывает ни одной ошибки —
    поиск продолжает работать и молча отдаёт мусор. Падать надо на старте.
    """
    if len(index) == 0:
        raise RuntimeError("Индекс пуст. Соберите его: python scripts/build_index.py")

    built_by = (getattr(index, "meta", {}) or {}).get("embedder")
    if built_by and built_by != embedder.name:
        raise RuntimeError(
            f"Индекс собран эмбеддером '{built_by}', а сервис поднят с "
            f"'{embedder.name}'. Пересоберите индекс: python scripts/build_index.py"
        )


class ScannerPipeline:
    """Собранный сервис. Создаётся один раз на старте приложения."""

    def __init__(
        self,
        catalog: Catalog,
        embedder: Embedder,
        index: VectorIndex,
        config: Config,
        normalizer: QueryNormalizer | None = None,
        text_extractor: TextExtractor | None = None,
        verifier: GeometricVerifier | None = None,
    ) -> None:
        self.catalog = catalog
        self.embedder = embedder
        self.index = index
        self.config = config
        self.normalizer = normalizer or QueryNormalizer(
            {**(config.get("normalize") or {}), "crop": {"enabled": False}}, ROOT
        )
        self.text_extractor = text_extractor or NullExtractor()
        self.matcher = TextMatcher(catalog)
        self.verifier = verifier
        # Запас потоков: OCR, брошенный по бюджету, не должен занять очередь.
        self._pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="scan")

    # ── сборка ────────────────────────────────────────────────────────────

    @classmethod
    def load(
        cls,
        settings: Settings | None = None,
        config: Config | None = None,
    ) -> ScannerPipeline:
        """Загружает каталог, модели и индекс.

        Вызывается из lifespan приложения: ТЗ требует, чтобы модель и индекс
        грузились при старте, а не по первому запросу — иначе первый запрос
        скрипта оценки оплачивает прогрев.
        """
        settings = settings or get_settings()
        config = config or get_config()

        started = time.perf_counter()
        catalog = load_catalog(settings=settings, config=config)
        embedder = build_embedder(config.get("embedder") or {})
        index = load_index(
            kind=settings.index_kind or config.get_path("index.kind", "numpy"),
            path=settings.index_path,
        )
        assert_index_matches(index, embedder)

        normalizer = QueryNormalizer(config.get("normalize") or {}, ROOT)
        text_extractor = build_text_extractor(
            config.get("text") or {},
            api_key=settings.yandex_api_key,
            folder_id=settings.yandex_folder_id,
        )

        verifier = build_verifier(catalog, config.get_path("rerank.geometric", {}) or {},
                                  settings.resolve(settings.artifacts_dir))
        pipeline = cls(catalog, embedder, index, config, normalizer, text_extractor, verifier)
        pipeline.warm_up()
        log.info(
            "Пайплайн готов за %.2f c: каталог %d, индекс %d, эмбеддер %s, OCR %s",
            time.perf_counter() - started,
            len(catalog),
            len(index),
            getattr(embedder, "query_name", embedder.name),
            text_extractor.name,
        )
        return pipeline

    def warm_up(self) -> None:
        """Первый прогон через все модели — чтобы не платить за него на запросе."""
        from PIL import Image

        blank = Image.new("RGB", (384, 384), (200, 180, 160))
        self.embedder.encode([blank], role="query")
        try:
            self.text_extractor.extract(blank)
        except Exception as exc:  # облачный OCR на старте может быть недоступен
            log.warning("Прогрев OCR не удался: %s", exc)

    # ── инференс ──────────────────────────────────────────────────────────

    def scan(self, payload: bytes) -> dict[str, Any]:
        """Главный вход. Всегда возвращает slug в корне ответа."""
        started = time.perf_counter()
        views, candidates, lines, timings = self.rank(payload)
        try:
            return self._respond(started, views, candidates, lines, timings)
        except Exception as exc:
            # Вино уже найдено: сбой карточки или аналогов не должен менять slug.
            log.exception("Сбой сборки ответа: %s", exc)
            return self.degraded(candidates[0].slug, "ошибка сборки ответа")

    def _respond(self, started: float, views, candidates: list, lines: list,
                 timings: dict) -> dict[str, Any]:
        metrics = estimate(
            [c.score for c in candidates],
            float(self.config.get_path("confidence.temperature", 0.05)),
            groups=[self._identity(c.slug) for c in candidates],
        )
        decision = self.confident(metrics["confidence"], candidates[0], lines)
        result = response.build(
            candidates, metrics, decision,
            card=lambda slug: self._card(self.catalog.get(slug)),
            return_k=int(self.config.get_path("index.return_k", 5)),
            label_detected=views.box is not None,
            ocr_text=" | ".join(line.text for line in lines),
            timings=timings,
        )
        if result["confident"]:
            result["analogs"] = self.analogs(result["slug"])
        else:
            # Лучший кандидат здесь заведомо неверен: аналоги — от профиля.
            result["analogs_profile"], result["analogs"] = self.profile_analogs(
                candidates, lines)
        result["elapsed_ms"] = round((time.perf_counter() - started) * 1000)
        return result

    def confident(self, confidence: float, best, lines: list) -> tuple[bool, float, float]:
        """-> (показать карточку сразу, match_score, name_recall). Только для UI."""
        wine = self.catalog.get(best.slug)
        name_recall, name_idf = self.matcher.name_recall(lines, wine) if wine else (0.0, 0.0)
        confident, match_score = decide(self.config.get("confidence") or {}, confidence,
                                        best.visual, best.textual, name_recall, name_idf)
        return confident, match_score, name_recall

    def fallback(self, reason: str) -> dict[str, Any]:
        """Фото не удалось обработать (битый файл, сбой модели): 200 и позиция каталога."""
        wine = next(iter(self.catalog))
        return response.fallback(wine.slug, self._card(wine), reason)

    def degraded(self, slug: str, reason: str) -> dict[str, Any]:
        """Вино найдено, но карточку или аналоги собрать не удалось."""
        try:
            card = self._card(self.catalog.get(slug))
        except Exception:
            card = None
        return response.degraded(slug, card, reason)

    def rank(self, payload: bytes, lines: list | None = None):
        """Фото -> (виды запроса, кандидаты по убыванию, строки OCR, тайминги).

        Отдельно от scan(), чтобы калибровка и стенд оценки гоняли ровно
        тот же код ранжирования, что и сервис. `lines` — готовые строки OCR
        (стенд берёт их из кэша, чтобы не платить облаку за повторные прогоны).
        """
        started = time.perf_counter()
        timings: dict[str, float] = {}

        views = self.normalizer(payload)
        timings["normalize"] = time.perf_counter() - started

        text_future = (
            self._pool.submit(self._read_text, views.ocr) if lines is None else None
        )

        mark = time.perf_counter()
        vector = encode_query(self.embedder, views,
                              float(self.config.get_path("embedder.packshot_mix", 0.5)))
        timings["visual"] = time.perf_counter() - mark

        if text_future is not None:
            # SLA 3 с: текст, не успевший к бюджету, не ждём — отвечаем по картинке.
            budget = float(self.config.get_path("text.budget", 2.4))
            try:
                lines, timings["ocr"] = text_future.result(
                    timeout=max(0.0, budget - (time.perf_counter() - started))
                )
            except FuturesTimeout:
                log.warning("OCR не уложился в %.1f c: ответ только по картинке", budget)
                lines, timings["ocr"] = [], budget
        candidates = rank_candidates(
            self.index,
            vector,
            self.matcher.score(lines) if lines else {},
            self.config.get_path("fusion.weights", {}),
            top_k=int(self.config.get_path("index.top_k", 50)),
            text_pool=int(self.config.get_path("text.pool", 20)),
        )
        if self.verifier is not None:
            mark = time.perf_counter()
            candidates = self.verifier.rerank(views, candidates,
                                              self.config.get_path("fusion.weights", {}))
            timings["geometric"] = time.perf_counter() - mark
        return views, candidates, lines, timings

    # ── карточки ──────────────────────────────────────────────────────────

    def wine(self, slug: str) -> dict[str, Any] | None:
        """Карточка позиции + аналоги — для перехода по аналогу и каталога в UI."""
        wine = self.catalog.get(slug)
        if wine is None:
            return None
        return {"slug": wine.slug, "card": self._card(wine), "analogs": self.analogs(wine.slug)}

    def browse(self, **filters: Any) -> dict[str, Any]:
        page = browse(self.catalog, **filters)
        return {
            "total": page.total,
            "items": [{"slug": w.slug, "card": self._card(w)} for w in page.items],
            "styles": [{"name": n, "count": c} for n, c in page.styles],
            "regions": [{"name": n, "count": c} for n, c in page.regions],
        }

    def sommelier(self, food: str, **filters: Any) -> dict[str, Any]:
        found = sommelier_pick(self.catalog, food, **filters)
        return {
            "total": found.total,
            "relaxed": found.relaxed,
            "used": found.used,
            "items": [{"slug": p.wine.slug, "card": self._card(p.wine), "why": p.why}
                      for p in found.items],
            "regions": [{"name": n, "count": c} for n, c in found.regions],
        }

    def analogs(self, slug: str) -> list[dict[str, Any]]:
        wine = self.catalog.get(slug)
        if wine is None:
            return []
        count = int(self.config.get_path("enrich.analogs_count", 4))
        return [self._analog(a) for a in match_analogs(self.catalog, wine, count)]

    def profile_analogs(self, candidates: list, lines: list) -> tuple[dict[str, Any], list]:
        """Вина нет в каталоге: аналоги от профиля топ-кандидатов и текста этикетки."""
        profile, found = out_of_catalog(
            self.catalog,
            [(self.catalog.get(c.slug), c.score) for c in candidates[:PROFILE_TOP]],
            " ".join(line.text for line in lines),
            int(self.config.get_path("enrich.analogs_count", 4)),
        )
        return profile.as_dict(), [self._analog(a) for a in found]

    def _analog(self, analog: Analog) -> dict[str, Any]:
        return {"slug": analog.wine.slug, "card": self._card(analog.wine),
                "reasons": analog.reasons}

    def _identity(self, slug: str) -> str:
        wine = self.catalog.get(slug)
        return wine.identity if wine is not None else slug

    def _read_text(self, image) -> tuple[list, float]:
        mark = time.perf_counter()
        try:
            lines = self.text_extractor.extract(image)
        except Exception as exc:
            # Текст — усилитель, а не условие ответа: без него работает визуальный канал.
            log.warning("OCR не сработал: %s", exc)
            lines = []
        return lines, time.perf_counter() - mark

    @staticmethod
    def _card(wine: Wine | None) -> dict[str, Any] | None:
        # None, а не {}: так в схеме ScanResponse (card: Card | None).
        return card_for(wine) if wine is not None else None
