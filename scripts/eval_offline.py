#!/usr/bin/env python3
"""Офлайн-стенд: честная точность и подбор весов без HTTP.

    python scripts/eval_offline.py                # сводная таблица конфигураций
    python scripts/eval_offline.py --errors       # + разбор промахов лучшей

Признаки каждого полевого фото (вектор без LoRA и с LoRA, текст OCR)
считаются один раз и кэшируются в artifacts/eval_cache/, дальше перебор
конфигураций занимает секунды.

Протоколы — от строгого к реалистичному:
- unseen : полевые фото этого вина в индексе не участвуют вовсе
           (вино, которого команда никогда не видела на полке);
- loo    : в индексе есть другие полевые фото этого вина, само фото
           исключено (leave-one-out: «вино видели, это фото — нет»).

LoRA обучалась на 48 из этих фото, поэтому её цифры на них завышены;
стенд печатает их отдельно по «чистым» фото, которых LoRA не видела.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wine_scanner.catalog.loader import load_catalog  # noqa: E402
from wine_scanner.catalog.models import same_wine  # noqa: E402
from wine_scanner.features.embedder import SigLIPEmbedder  # noqa: E402
from wine_scanner.features.text import TextLine, build_text_extractor  # noqa: E402
from wine_scanner.normalize.query import QueryNormalizer  # noqa: E402
from wine_scanner.search.index import NumpyIndex  # noqa: E402
from wine_scanner.search.rerank.fusion import rank_candidates  # noqa: E402
from wine_scanner.search.rerank.text_match import TextMatcher  # noqa: E402
from wine_scanner.settings import get_config  # noqa: E402

CACHE = ROOT / "artifacts" / "eval_cache"
LORA = ROOT / "artifacts" / "models" / "siglip_lora_v2.pt"
# Фото, на которых обучалась LoRA (сплит finetune_lora.py: seed 42, первые 12 — holdout)
LORA_TRAIN = ROOT / "data" / "raw" / "lora_train_photos.txt"


def macro_f1(pairs: list[tuple[str, str]]) -> float:
    """Macro F1 по винам: классы — slug из разметки и предсказаний (как sklearn average="macro")."""
    tp, fp, fn = defaultdict(int), defaultdict(int), defaultdict(int)
    for truth, pred in pairs:
        if pred == truth:
            tp[truth] += 1
        else:
            fp[pred] += 1
            fn[truth] += 1
    classes = set(tp) | set(fp) | set(fn)
    return sum(2 * tp[c] / (2 * tp[c] + fp[c] + fn[c]) for c in classes) / max(len(classes), 1)


def twins_key(wine) -> str:
    """Карточки-близнецы: одинаковые название и винодельня (…-shardone и …-shardone-1)."""
    return f"{wine.title.strip().lower()}|{wine.winery.strip().lower()}"


HEAD = ROOT / "artifacts" / "models" / "query_head.npz"


def load_features(catalog, labels: list[tuple[str, str]], field_dir: Path):
    CACHE.mkdir(parents=True, exist_ok=True)
    cat_path = CACHE / "catalog_base.npz"
    q_path = CACHE / "queries.npz"
    ocr_path = CACHE / "ocr.json"

    wines = [w for w in catalog if w.image and Path(w.image).exists()]
    embedder = None

    if cat_path.exists():
        data = np.load(cat_path, allow_pickle=True)
        cat_vecs, cat_slugs = data["vectors"], [str(s) for s in data["slugs"]]
    else:
        embedder = SigLIPEmbedder(lora_path=LORA)
        images = [ImageOps.exif_transpose(Image.open(w.image)).convert("RGB") for w in wines]
        cat_vecs = embedder.encode(images, role="catalog")
        cat_slugs = [w.slug for w in wines]
        np.savez(cat_path, vectors=cat_vecs, slugs=np.array(cat_slugs, dtype=object))

    names = [name for name, _ in labels]
    if q_path.exists() and ocr_path.exists():
        data = np.load(q_path, allow_pickle=True)
        if [str(s) for s in data["names"]] == names:
            ocr = json.loads(ocr_path.read_text(encoding="utf-8"))
            return cat_vecs, cat_slugs, data["base"], data["lora"], ocr

    embedder = embedder or SigLIPEmbedder(lora_path=LORA)
    normalizer = QueryNormalizer(get_config().get("normalize") or {}, ROOT)
    extractor = build_text_extractor({"kind": "apple"})
    base, lora, ocr = [], [], {}
    for position, name in enumerate(names, 1):
        views = normalizer((field_dir / name).read_bytes())
        base.append(embedder.encode([views.embed], role="catalog")[0])  # без адаптера
        lora.append(embedder.encode([views.embed], role="query")[0])
        ocr[name] = [[line.text, line.height] for line in extractor.extract(views.ocr)]
        print(f"  признаки {position}/{len(names)}", end="\r", flush=True)
    print()
    np.savez(q_path, base=np.stack(base), lora=np.stack(lora),
             names=np.array(names, dtype=object))
    ocr_path.write_text(json.dumps(ocr, ensure_ascii=False), encoding="utf-8")
    return cat_vecs, cat_slugs, np.stack(base), np.stack(lora), ocr


def load_cloud_ocr(names: list[str], field_dir: Path) -> dict[str, list]:
    """Yandex Vision OCR по тем же кропам, что видит сервис. Кэш — платим один раз."""
    from wine_scanner.features.text.yandex_ocr import YandexOCRExtractor
    from wine_scanner.settings import get_settings

    path = CACHE / "ocr_yandex.json"
    cache = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    missing = [n for n in names if n not in cache]
    if missing:
        settings = get_settings()
        if not settings.yandex_api_key:
            raise SystemExit("--ocr yandex: YANDEX_API_KEY в .env пуст")
        extractor = YandexOCRExtractor(settings.yandex_api_key.strip(),
                                       settings.yandex_folder_id.strip())
        normalizer = QueryNormalizer(get_config().get("normalize") or {}, ROOT)
        for position, name in enumerate(missing, 1):
            views = normalizer((field_dir / name).read_bytes())
            cache[name] = [[line.text, line.height] for line in extractor.extract(views.ocr)]
            path.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
            time.sleep(1.1)  # квота нового облака ~1 запрос/с
            print(f"  yandex ocr {position}/{len(missing)}", end="\r", flush=True)
        print()
    return cache


def load_rapid_ocr(names: list[str], field_dir: Path) -> dict[str, list]:
    """RapidOCR (локальный, кроссплатформенный) по тем же кропам, что видит сервис."""
    from wine_scanner.features.text.rapid_ocr import RapidOCRExtractor

    path = CACHE / "ocr_rapid.json"
    cache = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    missing = [n for n in names if n not in cache]
    if missing:
        extractor = RapidOCRExtractor()
        normalizer = QueryNormalizer(get_config().get("normalize") or {}, ROOT)
        for position, name in enumerate(missing, 1):
            views = normalizer((field_dir / name).read_bytes())
            cache[name] = [[line.text, line.height] for line in extractor.extract(views.ocr)]
            print(f"  rapid ocr {position}/{len(missing)}", end="\r", flush=True)
        path.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
        print()
    return cache


def rank(index, query, text_scores, w_text):
    """Тот же rank_candidates, что в сервисе; меняется только вес текста."""
    cfg = get_config()
    weights = {**cfg.get_path("fusion.weights", {}), "textual": w_text}
    candidates = rank_candidates(index, query[None, :], text_scores if w_text else {}, weights,
                                 top_k=int(cfg.get_path("index.top_k", 50)),
                                 text_pool=int(cfg.get_path("text.pool", 20)))
    return [c.slug for c in candidates]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=ROOT / "data/field/labels.csv")
    parser.add_argument("--field-dir", type=Path, default=ROOT / "data/field")
    parser.add_argument("--errors", action="store_true")
    parser.add_argument("--ocr", choices=["apple", "yandex", "rapid"], default="apple",
                        help="чей OCR подставлять в текстовый канал")
    args = parser.parse_args()

    catalog = load_catalog()
    by_slug = {w.slug: w for w in catalog}
    twin = {w.slug: twins_key(w) for w in catalog}
    with args.labels.open(encoding="utf-8") as fh:
        labels = [(r["image"], r["slug"]) for r in csv.DictReader(fh)]
    lora_train = set(LORA_TRAIN.read_text().split()) if LORA_TRAIN.exists() else set()

    cat_vecs, cat_slugs, base, lora, ocr = load_features(catalog, labels, args.field_dir)
    if args.ocr == "yandex":
        ocr = load_cloud_ocr([name for name, _ in labels], args.field_dir)
    elif args.ocr == "rapid":
        ocr = load_rapid_ocr([name for name, _ in labels], args.field_dir)
    print(f"OCR: {args.ocr}")
    matcher = TextMatcher(catalog)
    text = {
        name: matcher.score([TextLine(t, 1.0, a) for t, a in ocr.get(name, [])])
        for name, _ in labels
    }

    def hit(pred: str, truth: str, lenient: bool) -> bool:
        if same_wine(pred, truth):
            return True
        return lenient and twin.get(pred) == twin.get(truth)

    ref_slugs = [slug for _, slug in labels]
    index = NumpyIndex(cat_vecs.shape[1])
    index.add(cat_vecs, cat_slugs)

    def evaluate(space: str, protocol: str, w_text: float, subset=None):
        # protocol оставлен в ключе таблицы: полевые фото в индекс не идут (замер 23.09)
        queries = {"base": base, "lora": lora, "head": head}[space]
        stats = defaultdict(int)
        misses = []
        pairs: list[tuple[str, str]] = []
        for i, (name, truth) in enumerate(labels):
            if subset is not None and name not in subset:
                continue
            order = rank(index, queries[i], text[name], w_text)
            # попадание по same_wine (другой винтаж) считается верным классом
            pairs.append((truth, truth if hit(order[0], truth, False) else order[0]))
            stats["n"] += 1
            stats["top1"] += hit(order[0], truth, False)
            stats["top1_lenient"] += hit(order[0], truth, True)
            stats["top5"] += any(hit(p, truth, True) for p in order[:5])
            if not hit(order[0], truth, True):
                misses.append((name, truth, order[:3]))
        stats["macro_f1"] = macro_f1(pairs)
        return stats, misses

    clean = {n for n, _ in labels if n not in lora_train} if lora_train else None
    print(f"Полевых фото с меткой: {len(labels)} (вин: {len(set(ref_slugs))}); "
          f"не видела LoRA: {len(clean) if clean else '?'}")
    print(f"{'конфигурация':44s} {'top1':>7s} {'top1~':>7s} {'top5~':>7s} {'mF1':>7s}")
    results = {}
    head = None
    if HEAD.exists():  # голова запросов поверх базового вектора (train_query_head.py)
        with np.load(HEAD) as data:
            out = base + base @ data["weight"] + data["bias"]
        head = out / np.linalg.norm(out, axis=1, keepdims=True)
    for space in ["base", "lora"] + (["head"] if head is not None else []):
        for protocol in ["catalog"]:  # unseen/loo: полевые фото в индексе вредят (замер 23.09)
            for w_text in [0.0, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8]:
                stats, misses = evaluate(space, protocol, w_text)
                n = stats["n"]
                key = f"{space:4s} refs={protocol:7s} w_text={w_text:.1f}"
                results[key] = (stats, misses)
                print(f"{key:44s} {stats['top1']/n:7.1%} {stats['top1_lenient']/n:7.1%} "
                      f"{stats['top5']/n:7.1%} {stats['macro_f1']:7.3f}")
            if clean:
                for w_text in [0.0, 0.3, 0.4]:
                    stats, _ = evaluate(space, protocol, w_text, clean)
                    print(f"  └ только фото вне обучения LoRA (n={stats['n']}), "
                          f"w_text={w_text}: top1 {stats['top1']/stats['n']:.1%}  "
                          f"top1~ {stats['top1_lenient']/stats['n']:.1%}  mF1 {stats['macro_f1']:.3f}")
    print("top1~ / top5~ — с учётом карточек-близнецов (одинаковые название и винодельня)")

    if args.errors:
        best = max(results, key=lambda k: results[k][0]["top1_lenient"])
        print(f"\nПромахи лучшей конфигурации: {best}")
        for name, truth, top in results[best][1]:
            print(f"  {name}: ждали {truth} [{by_slug[truth].title}], "
                  f"получили {[by_slug[s].title for s in top]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
