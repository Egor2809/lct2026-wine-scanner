"""VectorIndex поверх PostgreSQL + pgvector — стек заказчика из ТЗ.

Включается `index.kind: pgvector` и `PGVECTOR_DSN` (таблица — `PGVECTOR_TABLE`,
по умолчанию wine_vectors). Источник истины — artifacts/index.npz: на старте
таблица сверяется с ним по хэшу и при расхождении перезаливается, так что
перенос индекса на сторону заказчика не требует пересчёта эмбеддингов.

Поиск идёт через HNSW по косинусному расстоянию (`<=>`); у позиции бывает
несколько строк-эталонов, поэтому берётся запас и схлопывается до лучшей
строки на slug — так же, как в NumpyIndex.

Зависимости опциональны: pip install "psycopg[binary]" pgvector.
"""

from __future__ import annotations

import hashlib
import logging
import os
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from ..features.embedder import l2_normalize
from ..settings import get_settings
from .index import NumpyIndex

log = logging.getLogger(__name__)

def _connect(dsn: str):
    try:
        import psycopg
        from pgvector.psycopg import register_vector
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            'index.kind: pgvector требует pip install "psycopg[binary]" pgvector') from exc
    conn = psycopg.connect(dsn, autocommit=True)
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    # HNSW отдаёт не больше ef_search строк (по умолчанию 40), а поиску нужен
    # запас k*4 на дубли эталонов; 1000 — предел pgvector.
    conn.execute("SET hnsw.ef_search = 1000")
    register_vector(conn)
    return conn


def _statements(table: str, dim: int) -> dict:
    """SQL с именем таблицы через Identifier: имя из настроек, не строкой в запрос."""
    from psycopg import sql

    names = {"t": sql.Identifier(table), "m": sql.Identifier(f"{table}_meta"),
             "h": sql.Identifier(f"{table}_hnsw"), "dim": sql.Literal(dim)}
    queries = {
        "create": "CREATE TABLE IF NOT EXISTS {t} (id bigserial PRIMARY KEY,"
                  " slug text NOT NULL, source text NOT NULL, embedding vector({dim}) NOT NULL)",
        "hnsw": "CREATE INDEX IF NOT EXISTS {h} ON {t} USING hnsw (embedding vector_cosine_ops)",
        "create_meta": "CREATE TABLE IF NOT EXISTS {m} (key text PRIMARY KEY, value text)",
        "drop": "DROP TABLE IF EXISTS {t}",
        "clear_meta": "DELETE FROM {m}",
        "read_meta": "SELECT key, value FROM {m}",
        "write_meta": "INSERT INTO {m} (key, value) VALUES (%s, %s)"
                      " ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value",
        "positions": "SELECT count(DISTINCT slug) FROM {t}",
        "rows": "SELECT count(*) FROM {t}",
        "insert": "INSERT INTO {t} (slug, source, embedding) VALUES (%s, %s, %s)",
        "search": "SELECT slug, 1 - (embedding <=> %s) FROM {t}"
                  " WHERE NOT (source = ANY(%s)) ORDER BY embedding <=> %s LIMIT %s",
        "score": "SELECT slug, max(1 - (embedding <=> %s)) FROM {t}"
                 " WHERE slug = ANY(%s) AND NOT (source = ANY(%s)) GROUP BY slug",
    }
    return {key: sql.SQL(text).format(**names) for key, text in queries.items()}


class PgVectorIndex:
    kind = "pgvector"

    def __init__(self, dim: int, meta: dict | None = None, dsn: str | None = None,
                 table: str | None = None) -> None:
        settings = get_settings()
        self._dsn = dsn or os.environ.get("PGVECTOR_DSN") or settings.pgvector_dsn
        table = table or os.environ.get("PGVECTOR_TABLE") or settings.pgvector_table
        if not self._dsn:
            raise RuntimeError("index.kind: pgvector требует PGVECTOR_DSN (см. .env.example)")
        self.dim = dim
        self._conn = _connect(self._dsn)
        self._q = _statements(table, dim)
        for key in ("create", "hnsw", "create_meta"):
            self._conn.execute(self._q[key])
        stored = dict(self._exec("read_meta").fetchall())
        self.meta = stored if meta is None else dict(meta)

    def _exec(self, key: str, params: tuple | None = None):
        """Запрос с одним переподключением: перезапуск Postgres не должен
        превращать каждый /scan в fallback до рестарта сервиса."""
        import psycopg

        try:
            return self._conn.execute(self._q[key], params)
        except psycopg.OperationalError:
            log.warning("pgvector: соединение потеряно, переподключаюсь")
            self._conn = _connect(self._dsn)
            return self._conn.execute(self._q[key], params)

    def __len__(self) -> int:
        return self._exec("positions").fetchone()[0]

    @property
    def rows(self) -> int:
        return self._exec("rows").fetchone()[0]

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
        with self._conn.cursor() as cur:
            cur.executemany(self._q["insert"], zip(slugs, sources, vectors, strict=True))

    def search(self, query: np.ndarray, k: int, exclude: Iterable[str] = ()
               ) -> list[tuple[str, float]]:
        q = l2_normalize(query)[0]
        rows = self._exec("search", (q, list(exclude), q, k * 4)).fetchall()
        out: list[tuple[str, float]] = []
        seen: set[str] = set()
        for slug, score in rows:
            if slug not in seen:
                seen.add(slug)
                out.append((slug, float(score)))
        return out[:k]

    def score(self, query: np.ndarray, slugs: Iterable[str], exclude: Iterable[str] = ()
              ) -> dict[str, float]:
        rows = self._exec("score", (l2_normalize(query)[0], list(slugs), list(exclude)))
        return {slug: float(score) for slug, score in rows.fetchall()}

    def save(self, path: Path | None = None) -> None:
        """Векторы уже в базе; сохраняется только происхождение индекса."""
        self._conn.execute(self._q["clear_meta"])
        with self._conn.cursor() as cur:
            cur.executemany(self._q["write_meta"], [(k, str(v)) for k, v in self.meta.items()])

    def mirror(self, seed: NumpyIndex, digest: str) -> None:
        """Таблица := содержимое npz. Одной транзакцией: поиск не видит полуиндекс."""
        with self._conn.transaction():
            self._conn.execute(self._q["drop"])
            for key in ("create", "hnsw"):
                self._conn.execute(self._q[key])
            self.add(seed._vectors, seed._slugs, seed._sources)
            self.meta = {**seed.meta, "digest": digest}
            self.save()

    @classmethod
    def load(cls, path: Path) -> PgVectorIndex:
        """npz — источник истины (его пишет scripts/build_index.py), база — зеркало.

        Хэш содержимого npz хранится в meta: пересборка индекса или смена
        модели меняют хэш, и таблица перезаливается. Иначе поиск молча шёл бы
        по векторам старой модели, а проверка эмбеддера на старте — по новой meta.
        """
        seed = NumpyIndex.load(path)
        digest = _digest(seed)
        index = cls(dim=seed.dim)
        if index.meta.get("digest") != digest:
            index.mirror(seed, digest)
            log.info("pgvector: таблица синхронизирована с %s (%d строк)", path, seed.rows)
        log.info("Индекс pgvector: %d позиций, %d строк", len(index), index.rows)
        return index


def _digest(index: NumpyIndex) -> str:
    h = hashlib.sha1(index._vectors.tobytes())
    h.update("\n".join(index._slugs).encode())
    h.update("\n".join(index._sources).encode())
    h.update(str(index.meta.get("embedder", "")).encode())
    return h.hexdigest()
