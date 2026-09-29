#!/usr/bin/env python3
"""Стенд VLM-судьи: точность пайплайна до и после судьи.

    python scripts/eval_judge.py                     # судья на каждом фото
    python scripts/eval_judge.py --thinking low      # с рассуждениями модели

Ранжирование — тот же ScannerPipeline.rank(), что в сервисе; строки OCR
берутся из кэша стенда (artifacts/eval_cache/ocr_yandex.json), ответы судьи
тоже кэшируются — повторный прогон бесплатный. Печатает точность отдельно по
фото, которых не видела LoRA, и что судья говорит о винах вне каталога.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wine_scanner.catalog.models import same_wine  # noqa: E402
from wine_scanner.features.text import TextLine  # noqa: E402
from wine_scanner.pipeline import ScannerPipeline  # noqa: E402
from wine_scanner.search.confidence import estimate  # noqa: E402
from wine_scanner.search.rerank.vlm_judge import JudgeCandidate, VLMJudge  # noqa: E402
from wine_scanner.settings import get_settings  # noqa: E402

CACHE = ROOT / "artifacts" / "eval_cache"
FIELD = ROOT / "data" / "field"


def finalists(pipeline: ScannerPipeline, candidates, k: int) -> list[str]:
    """Топ-k различных вин: близнецы схлопываются в лучшего представителя."""
    out, seen = [], set()
    for c in candidates:
        key = pipeline._identity(c.slug)
        if key not in seen:
            seen.add(key)
            out.append(c.slug)
        if len(out) == k:
            break
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--thinking", default="disabled", help="disabled | low | medium | high")
    parser.add_argument("--k", type=int, default=5, help="сколько финалистов видит судья")
    parser.add_argument("--ooc", action="store_true", help="прогнать и 37 фото вне каталога")
    args = parser.parse_args()

    settings = get_settings()
    judge = VLMJudge(settings.vlm_base_url, settings.vlm_api_key.strip(), settings.vlm_model,
                     thinking=args.thinking, timeout=90)
    pipeline = ScannerPipeline.load()
    ocr = json.loads((CACHE / "ocr_yandex.json").read_text(encoding="utf-8"))
    lora_train = set((ROOT / "data/raw/lora_train_photos.txt").read_text().split())
    cache_path = CACHE / f"judge_{judge.name.replace(':', '_')}_k{args.k}.json"
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}

    def rows(name: str, column: str):
        with (FIELD / name).open(encoding="utf-8") as fh:
            return [(r["image"], r[column]) for r in csv.DictReader(fh)]

    items = [(n, s, True) for n, s in rows("labels.csv", "slug")]
    if args.ooc:
        items += [(n, None, False) for n, _ in rows("out_of_catalog.csv", "wine")]

    def hit(pred: str, truth: str) -> bool:
        return same_wine(pred, truth) or pipeline._identity(pred) == pipeline._identity(truth)

    stats = {k: 0 for k in ["n", "base", "judge", "base_strict", "judge_strict", "fixed",
                            "broken", "clean_n", "clean_base", "clean_judge", "none",
                            "gated", "clean_gated", "uncertain"]}
    ooc = {"n": 0, "none": 0}
    latencies = []
    for position, (name, truth, known) in enumerate(items, 1):
        cached_lines = ocr.get(name)
        lines = [TextLine(t, 1.0, h) for t, h in cached_lines] if cached_lines is not None else None
        views, candidates, lines, _ = pipeline.rank((FIELD / name).read_bytes(), lines=lines)
        if cached_lines is None:  # фото вне каталога: OCR оплачен один раз, дальше из кэша
            ocr[name] = [[line.text, line.height] for line in lines]
            (CACHE / "ocr_yandex.json").write_text(json.dumps(ocr, ensure_ascii=False),
                                                  encoding="utf-8")
        final = finalists(pipeline, candidates, args.k)
        key = name + ":" + hashlib.sha256("|".join(final).encode()).hexdigest()[:10]
        if key not in cache:
            wines = [pipeline.catalog.get(s) for s in final]
            verdict = judge.judge(
                views.ocr,
                [JudgeCandidate(w.slug, w.title, w.winery, w.style, w.grape, w.image)
                 for w in wines],
                " | ".join(line.text for line in lines),
            )
            cache[key] = {"index": verdict.index, "confidence": verdict.confidence,
                          "elapsed": verdict.elapsed, "raw": verdict.raw[:300]}
            cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=1),
                                  encoding="utf-8")
        verdict = cache[key]
        latencies.append(verdict["elapsed"])
        base_pred = candidates[0].slug
        cfg = pipeline.config
        conf = estimate([x.score for x in candidates],
                        float(cfg.get_path("confidence.temperature", 0.01)),
                        [pipeline._identity(x.slug) for x in candidates])["confidence"]
        confident = pipeline.confident(conf, candidates[0], lines)[0]
        gated_pred = base_pred if confident else (
            final[verdict["index"]] if verdict["index"] is not None else base_pred)
        judge_pred = final[verdict["index"]] if verdict["index"] is not None else base_pred
        print(f"  {position}/{len(items)}", end="\r", flush=True)

        if not known:
            ooc["n"] += 1
            ooc["none"] += verdict["index"] is None
            continue
        b, j = hit(base_pred, truth), hit(judge_pred, truth)
        g = hit(gated_pred, truth)
        stats["gated"] += g
        stats["uncertain"] += not confident
        if name not in lora_train:
            stats["clean_gated"] += g
        stats["n"] += 1
        stats["base"] += b
        stats["judge"] += j
        stats["base_strict"] += same_wine(base_pred, truth)
        stats["judge_strict"] += same_wine(judge_pred, truth)
        stats["fixed"] += j and not b
        stats["broken"] += b and not j
        stats["none"] += verdict["index"] is None
        if name not in lora_train:
            stats["clean_n"] += 1
            stats["clean_base"] += b
            stats["clean_judge"] += j
        if b != j:
            what = "ИСПРАВИЛ" if j else "СЛОМАЛ "
            print(f"  {what} {name[:24]}: ждали {truth}, пайплайн {base_pred}, судья {judge_pred}")

    n, c = stats["n"], stats["clean_n"]
    latencies.sort()
    print(f"\nСудья: {judge.name}, финалистов: {args.k}")
    print("                          без судьи   с судьёй")
    print(f"top-1 строго             {stats['base_strict']/n:9.1%} {stats['judge_strict']/n:10.1%}")
    print(f"top-1 с близнецами       {stats['base']/n:9.1%} {stats['judge']/n:10.1%}")
    print(f"  фото вне обучения LoRA {stats['clean_base']/c:9.1%} {stats['clean_judge']/c:10.1%}"
          f"   (n={c})")
    print(f"судья только на неуверенных ({stats['uncertain']}/{n} фото): "
          f"{stats['gated']/n:.1%}, вне обучения LoRA {stats['clean_gated']/c:.1%}")
    print(f"исправил {stats['fixed']}, сломал {stats['broken']}, "
          f"«ни один не подходит» на винах каталога: {stats['none']}/{n}")
    if ooc["n"]:
        print(f"вина вне каталога: судья ответил «ни один» в {ooc['none']}/{ooc['n']}")
    print(f"задержка судьи p50 {latencies[len(latencies)//2]:.1f} c, "
          f"p95 {latencies[int(len(latencies)*0.95)-1]:.1f} c")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
