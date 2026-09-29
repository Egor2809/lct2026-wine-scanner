"""Границы слоёв из ARCHITECTURE.md — проверкой, а не обещанием.

Разбор импортов по AST: модели не грузятся, тест идёт за миллисекунды. Новый
импорт через границу слоя роняет CI — его придётся либо убрать, либо
осознанно дописать сюда и в ARCHITECTURE.md.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parents[1] / "src" / "wine_scanner"

# Слой -> от каких слоёв пакета он вправе зависеть.
ALLOWED = {
    "settings": set(),
    "response": set(),
    "catalog": {"settings"},
    "features": {"settings"},
    "normalize": {"features", "settings"},
    "search": {"catalog", "features", "settings"},
    "enrich": {"catalog", "settings"},
    "pipeline": {"catalog", "enrich", "features", "normalize", "response", "search",
                 "settings"},
    # Модели и индекс — только через пайплайн: API не знает, чем ищется вино.
    "api": {"catalog", "enrich", "normalize", "pipeline", "response", "settings"},
}

# Тяжёлые ML-библиотеки живут за протоколами Embedder, TextExtractor, VectorIndex.
MODEL_LIBS = {"torch", "transformers", "ultralytics", "cv2", "rapidocr", "ocrmac",
              "onnxruntime", "faiss", "psycopg", "pgvector"}
MODEL_FREE = {"pipeline", "api", "response", "enrich", "catalog", "settings"}


def _layer(path: Path) -> str:
    rel = path.relative_to(PACKAGE)
    return rel.parts[0] if len(rel.parts) > 1 else rel.stem


def _imports(path: Path) -> tuple[set[str], set[str]]:
    """-> (слои пакета, внешние библиотеки), которые импортирует файл."""
    package = list(path.relative_to(PACKAGE).parts[:-1])
    internal, external = set(), set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.ImportFrom) and node.level:
            base = package[: len(package) - node.level + 1]
            parts = base + (node.module.split(".") if node.module else [])
            internal |= {parts[0]} if parts else {alias.name for alias in node.names}
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [node.module] if isinstance(node, ast.ImportFrom) else [
                alias.name for alias in node.names]
            for name in names:
                parts = name.split(".")
                # Абсолютный импорт своего пакета — та же граница слоя.
                if parts[0] == PACKAGE.name and len(parts) > 1:
                    internal.add(parts[1])
                else:
                    external.add(parts[0])
    return internal, external


FILES = sorted(p for p in PACKAGE.rglob("*.py") if p.name != "__init__.py" or p.parent != PACKAGE)


def test_every_layer_is_declared() -> None:
    assert {_layer(p) for p in FILES} <= set(ALLOWED)


@pytest.mark.parametrize("path", FILES, ids=lambda p: str(p.relative_to(PACKAGE)))
def test_layer_boundaries(path: Path) -> None:
    layer = _layer(path)
    internal, external = _imports(path)
    crossed = internal - ALLOWED[layer] - {layer}
    assert not crossed, f"{layer} не должен импортировать {sorted(crossed)}"
    if layer in MODEL_FREE:
        heavy = external & MODEL_LIBS
        assert not heavy, f"{layer} импортирует модели напрямую: {sorted(heavy)}"
