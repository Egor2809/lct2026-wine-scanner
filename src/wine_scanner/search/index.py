"""Векторный индекс каталога.

Протокол `VectorIndex` с тремя реализациями: NumpyIndex (по умолчанию),
FaissIndex (ANN) и PgVectorIndex (search/pgvector_index.py — стек заказчика из
ТЗ). На хакатоне выигрывает локальный индекс без внешних зависимостей, а
переносимость показывается сменой одной строки в YAML.

У одной позиции может быть несколько строк-эталонов: фото каталога и
реальные фото с полки из публичного набора кейса. Поиск агрегирует их
максимумом — вино находится по ближайшему из своих эталонов. У каждой
строки есть `source` («catalog» или имя полевого фото): так стенд оценки
честно исключает из индекса само проверяемое фото.

NumpyIndex — точный косинусный поиск. На 2000×768 это ~6 МБ и единицы
миллисекунд на запрос; ANN здесь не даёт ничего, кроме риска приближения.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from ..features.embedder import l2_normalize

log = logging.getLogger(__name__)


@runtime_checkable
class VectorIndex(Protocol):
    meta: dict

    def add(self, vectors: np.ndarray, slugs: list[str], sources: list[str] | None = None
            ) -> None: ...

    def search(self, query: np.ndarray, k: int, exclude: Iterable[str] = ()
               ) -> list[tuple[str, float]]: ...

    def score(self, query: np.ndarray, slugs: Iterable[str], exclude: Iterable[str] = ()
              ) -> dict[str, float]: ...

    def save(self, path: Path) -> None: ...

    def __len__(self) -> int: ...


def _best_per_slug(slugs: list[str], scores: np.ndarray, order: np.ndarray, k: int
                   ) -> list[tuple[str, float]]:
    out: list[tuple[str, float]] = []
    seen: set[str] = set()
    for row in order:
        slug = slugs[row]
        if slug in seen:
            continue
        seen.add(slug)
        out.append((slug, float(scores[row])))
        if len(out) == k:
            break
    return out


class NumpyIndex:
    """Точный косинусный поиск (brute force) на L2-нормированных векторах."""

    kind = "numpy"

    def __init__(self, dim: int, meta: dict | None = None) -> None:
        self.dim = dim
        self.meta = meta or {}
        self._vectors = np.zeros((0, dim), dtype=np.float32)
        self._slugs: list[str] = []
        self._sources: list[str] = []

    def __len__(self) -> int:
        """Число позиций каталога (не строк: у позиции бывает несколько эталонов)."""
        return len(set(self._slugs))

    @property
    def rows(self) -> int:
        return len(self._slugs)

    @property
    def slugs(self) -> list[str]:
        return list(dict.fromkeys(self._slugs))

    def add(self, vectors: np.ndarray, slugs: list[str], sources: list[str] | None = None
            ) -> None:
        vectors = l2_normalize(vectors)
        sources = sources or ["catalog"] * len(slugs)
        if not (vectors.shape[0] == len(slugs) == len(sources)):
            raise ValueError(
                f"Рассинхрон векторов и slug: {vectors.shape[0]} против {len(slugs)}"
            )
        if vectors.shape[1] != self.dim:
            raise ValueError(f"Ожидалась размерность {self.dim}, получена {vectors.shape[1]}")
        self._vectors = np.vstack([self._vectors, vectors])
        self._slugs.extend(slugs)
        self._sources.extend(sources)

    def _scores(self, query: np.ndarray, exclude: Iterable[str]) -> np.ndarray:
        scores = self._vectors @ l2_normalize(query)[0]
        excluded = set(exclude)
        if excluded:
            mask = np.fromiter((s in excluded for s in self._sources), bool, len(self._sources))
            scores = np.where(mask, -np.inf, scores)
        return scores

    def search(self, query: np.ndarray, k: int, exclude: Iterable[str] = ()
               ) -> list[tuple[str, float]]:
        if self.rows == 0:
            return []
        scores = self._scores(query, exclude)
        # С запасом: у позиции бывает несколько строк, дубли схлопнутся.
        fetch = min(self.rows, k * 4)
        top = np.argpartition(-scores, fetch - 1)[:fetch]
        top = top[np.argsort(-scores[top])]
        return _best_per_slug(self._slugs, scores, top, k)

    def score(self, query: np.ndarray, slugs: Iterable[str], exclude: Iterable[str] = ()
              ) -> dict[str, float]:
        """Точный визуальный скор для заданных позиций (кандидаты из текстового канала)."""
        wanted = set(slugs)
        scores = self._scores(query, exclude)
        out: dict[str, float] = {}
        for row, slug in enumerate(self._slugs):
            if slug in wanted and scores[row] > out.get(slug, -np.inf):
                out[slug] = float(scores[row])
        return out

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            vectors=self._vectors,
            slugs=np.array(self._slugs, dtype=object),
            sources=np.array(self._sources, dtype=object),
            meta=np.array([self.meta], dtype=object),
        )
        log.info("Индекс сохранён: %s (%d позиций, %d строк, dim=%d)", path, len(self),
                 self.rows, self.dim)

    @classmethod
    def load(cls, path: Path) -> NumpyIndex:
        if not path.exists():
            raise FileNotFoundError(
                f"Индекс не найден: {path}. Соберите его: python scripts/build_index.py"
            )
        data = np.load(path, allow_pickle=True)
        vectors = data["vectors"].astype(np.float32)
        meta = dict(data["meta"][0]) if "meta" in data else {}
        index = cls(dim=int(vectors.shape[1]), meta=meta)
        index._vectors = vectors
        index._slugs = [str(s) for s in data["slugs"].tolist()]
        index._sources = (
            [str(s) for s in data["sources"].tolist()]
            if "sources" in data
            else ["catalog"] * len(index._slugs)
        )
        log.info("Индекс загружен: %s (%d позиций, %d строк)", path, len(index), index.rows)
        return index


class FaissIndex(NumpyIndex):
    """faiss.IndexFlatIP поверх того же формата. Включается index.kind: faiss.

    Демонстрирует переносимость на ANN-движок; на 2000 позициях точнее и не
    медленнее NumpyIndex. На macOS faiss-cpu и torch тянут две копии OpenMP —
    в одном процессе с SigLIP это падение, поэтому по умолчанию numpy.
    """

    kind = "faiss"

    def _faiss(self):
        try:
            import faiss
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("index.kind: faiss требует faiss-cpu") from exc
        if getattr(self, "_ann", None) is None or self._ann.ntotal != self.rows:
            self._ann = faiss.IndexFlatIP(self.dim)
            self._ann.add(self._vectors)
        return self._ann

    def search(self, query: np.ndarray, k: int, exclude: Iterable[str] = ()
               ) -> list[tuple[str, float]]:
        if exclude:
            return super().search(query, k, exclude)
        fetch = min(self.rows, k * 4)
        scores, ids = self._faiss().search(l2_normalize(query), fetch)
        found = [(self._slugs[i], s) for i, s in zip(ids[0], scores[0], strict=True) if i >= 0]
        slugs = [slug for slug, _ in found]
        values = np.array([score for _, score in found], dtype=np.float32)
        return _best_per_slug(slugs, values, np.arange(len(found)), k)


_REGISTRY = {"numpy": NumpyIndex, "faiss": FaissIndex}


def _index_class(kind: str):
    kind = kind.lower()
    if kind == "pgvector":
        # Опциональная зависимость (psycopg): импорт только по запросу.
        from .pgvector_index import PgVectorIndex

        return PgVectorIndex
    try:
        return _REGISTRY[kind]
    except KeyError:
        raise ValueError(
            f"Неизвестный index.kind: {kind}. Доступны: {sorted([*_REGISTRY, 'pgvector'])}"
        ) from None


def build_index(kind: str, dim: int, meta: dict | None = None) -> VectorIndex:
    # pgvector — зеркало npz: сборка всегда пишет npz, база сверяется с ним на старте.
    if kind.lower() == "pgvector":
        return NumpyIndex(dim=dim, meta=meta)
    return _index_class(kind)(dim=dim, meta=meta)


def load_index(kind: str, path: Path) -> VectorIndex:
    return _index_class(kind).load(path)
