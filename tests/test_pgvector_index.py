"""PgVectorIndex против NumpyIndex на одних данных: тот же протокол, те же ответы.

Нужна база: PGVECTOR_DSN=postgresql://... (в CI — сервис pgvector/pgvector).
Без неё тесты пропускаются.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path

import numpy as np
import pytest

from wine_scanner.search.index import NumpyIndex, VectorIndex, load_index

DSN = os.environ.get("PGVECTOR_DSN", "")
pytestmark = pytest.mark.skipif(not DSN, reason="PGVECTOR_DSN не задан")


@pytest.fixture()
def data():
    rng = np.random.default_rng(0)
    vectors = rng.normal(size=(60, 16)).astype(np.float32)
    slugs = [f"wine-{i % 40}" for i in range(60)]
    sources = ["catalog" if i < 40 else f"field-{i}.jpg" for i in range(60)]
    return vectors, slugs, sources


@pytest.fixture()
def pg(data):
    from wine_scanner.search.pgvector_index import PgVectorIndex

    table = f"test_{uuid.uuid4().hex[:8]}"
    index = PgVectorIndex(dim=16, meta={"embedder": "test"}, dsn=DSN, table=table)
    index.add(*data)
    yield index
    for name in (table, f"{table}_meta"):
        index._conn.execute(f"DROP TABLE {name}")


def test_matches_numpy_index(pg, data) -> None:
    reference = NumpyIndex(dim=16)
    reference.add(*data)
    assert isinstance(pg, VectorIndex)
    assert len(pg) == len(reference) == 40 and pg.rows == 60

    query = data[0][7:8] + 0.05
    excluded = ["field-47.jpg"]
    for exclude in ((), excluded):
        found, expected = pg.search(query, 10, exclude), reference.search(query, 10, exclude)
        assert [s for s, _ in found] == [s for s, _ in expected]
        assert np.allclose([v for _, v in found], [v for _, v in expected], atol=1e-4)
    assert pg.score(query, ["wine-3", "wine-7"], excluded) == pytest.approx(
        reference.score(query, ["wine-3", "wine-7"], excluded), abs=1e-4)


def test_load_mirrors_npz_and_follows_rebuilds(tmp_path: Path, data, monkeypatch) -> None:
    table = f"test_{uuid.uuid4().hex[:8]}"
    monkeypatch.setenv("PGVECTOR_TABLE", table)
    npz = tmp_path / "index.npz"

    def build(vectors, embedder: str) -> None:
        seed = NumpyIndex(dim=vectors.shape[1], meta={"embedder": embedder})
        seed.add(vectors, data[1], data[2])
        seed.save(npz)

    build(data[0], "model-a")
    first = load_index("pgvector", npz)
    try:
        assert load_index("pgvector", npz).rows == 60, "повторный старт не дублирует строки"

        # Пересборка другой моделью той же размерности: таблица обязана перезалиться,
        # иначе поиск идёт по старым векторам, а проверка на старте видит новую meta.
        build(data[0][::-1].copy(), "model-b")
        rebuilt = load_index("pgvector", npz)
        assert rebuilt.rows == 60 and rebuilt.meta["embedder"] == "model-b"
        assert rebuilt.search(data[0][-1:], 1)[0][0] == data[1][0]

        # Смена размерности тоже перезаливает таблицу.
        build(np.hstack([data[0], data[0]]), "model-c")
        wide = load_index("pgvector", npz)
        assert wide.dim == 32 and wide.rows == 60

        # Перезапуск Postgres: соединение восстанавливается на следующем запросе.
        wide._conn.close()
        assert wide.search(np.hstack([data[0], data[0]])[:1], 1)
    finally:
        for name in (table, f"{table}_meta"):
            first._conn.execute(f"DROP TABLE IF EXISTS {name}")
