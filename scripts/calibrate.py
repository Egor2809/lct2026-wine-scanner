#!/usr/bin/env python3
"""Калибровка confidence: температура softmax и порог «уверенно».

    python scripts/calibrate.py

Гоняет полевые фото через тот же ScannerPipeline.rank(), что и сервис:
- data/field/labels.csv          — вина из каталога (правильный ответ известен);
- data/field/out_of_catalog.csv  — вин нет в каталоге (правильного ответа нет).

Температура подбирается по log-loss на винах из каталога, порог — так,
чтобы «уверенно» почти никогда не срабатывало на винах вне каталога и
срабатывало на правильных ответах. Результат — две строки для config.yaml.
Порог влияет только на UI: на оценочном прогоне slug отдаётся всегда.
"""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wine_scanner.pipeline import ScannerPipeline  # noqa: E402
from wine_scanner.search.confidence import estimate  # noqa: E402


def read(path: Path, column: str) -> list[tuple[str, str]]:
    with path.open(encoding="utf-8") as fh:
        return [(row["image"], row[column]) for row in csv.DictReader(fh)]


def main() -> int:
    pipeline = ScannerPipeline.load()
    field = ROOT / "data" / "field"
    known = read(field / "labels.csv", "slug")
    unknown = read(field / "out_of_catalog.csv", "wine")

    runs = []  # (scores, groups, truth_identity | None)
    for position, (name, truth) in enumerate(known + unknown, 1):
        _, candidates, _, _ = pipeline.rank((field / name).read_bytes())
        groups = [pipeline._identity(c.slug) for c in candidates]
        is_known = position <= len(known)
        runs.append(([c.score for c in candidates], groups,
                     pipeline._identity(truth) if is_known else None))
        print(f"  {position}/{len(known) + len(unknown)}", end="\r", flush=True)
    print()

    best_t, best_loss = None, math.inf
    for t in [0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.07, 0.1]:
        loss = 0.0
        for scores, groups, truth in runs:
            if truth is None:
                continue
            probs = estimate(scores, t, groups)  # вероятность top-1 группы
            p = probs["confidence"] if groups[0] == truth else 1e-3
            loss -= math.log(max(p, 1e-6))
        if loss < best_loss:
            best_t, best_loss = t, loss

    rows = []
    for scores, groups, truth in runs:
        conf = estimate(scores, best_t, groups)["confidence"]
        rows.append((conf, truth is not None and groups[0] == truth, truth is None))

    print(f"temperature = {best_t} (log-loss {best_loss:.2f})")
    header = ("порог", "верных уверенно", "ошибок уверенно", "вне каталога уверенно")
    print(f"{header[0]:>6s} {header[1]:>16s} {header[2]:>16s} {header[3]:>22s}")
    for threshold in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        right = sum(1 for c, ok, _ in rows if ok and c >= threshold)
        wrong = sum(1 for c, ok, ooc in rows if not ok and not ooc and c >= threshold)
        ooc = sum(1 for c, _, is_ooc in rows if is_ooc and c >= threshold)
        n_right = sum(1 for _, ok, _ in rows if ok)
        n_ooc = sum(1 for *_, is_ooc in rows if is_ooc)
        print(f"{threshold:6.2f} {right:>9d}/{n_right:<6d} {wrong:>16d} {ooc:>15d}/{n_ooc:<6d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
