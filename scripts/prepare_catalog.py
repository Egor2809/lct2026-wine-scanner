#!/usr/bin/env python3
"""Сырой дамп «Своё вино» -> чистый data/catalog.csv.

    python scripts/prepare_catalog.py \
        --dump data/raw/catalog_dump.csv --photos data/catalog_photos

Что делает:
1. схлопывает дубли строк (в выданном дампе почти каждая позиция дважды);
2. находит файл фото на диске: в дампе исходное имя, на диске — имя после
   Strapi (см. catalog/photo_match.py);
3. разбирает неоднозначности: клиенты грузили разные фото под одним именем
   («Шардоне.webp»). Одинаковые по содержимому файлы схлопываются, а между
   разными выбирается тот, на этикетке которого OCR читает название и
   винодельню именно этого вина;
4. пишет отчёт artifacts/catalog_report.txt: что не сопоставилось и как
   разрешены спорные случаи — это решения, которые должен видеть человек.

Используются только фото из датасета кейса (требование организаторов).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from pathlib import Path

from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wine_scanner.catalog.models import Wine  # noqa: E402
from wine_scanner.catalog.photo_match import match_photos  # noqa: E402
from wine_scanner.features.text import build_text_extractor  # noqa: E402
from wine_scanner.search.rerank.text_match import TextMatcher  # noqa: E402

# Колонки выданного дампа -> колонки data/catalog.csv (их понимает loader.py)
DUMP_COLUMNS = {
    "Slug": "slug",
    "Название вина": "title",
    "Винодельня": "winery",
    "Регион": "region",
    "Сорт винограда": "grape",
    "Категория": "style",
    "Цвет": "color",
    "Описание": "description",
}
PHOTO_COLUMN = "Название фото"


def content_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# В папке с фото каталога лежат и картинки блога: баннеры ярмарок, застолья,
# групповые снимки. Отличать эталон от них — zero-shot текстовой башней SigLIP.
BOTTLE_PROMPTS = [
    "a product photo of a single wine bottle",
    "a bottle of wine with a label on a plain background",
]
OTHER_PROMPTS = [
    "a poster or banner with large text",
    "people at an event",
    "food on a table",
    "a landscape or a building",
    "a row of many different wine bottles",
    "a glass of wine",
]


class BottleScorer:
    def __init__(self) -> None:
        from wine_scanner.features.embedder import SigLIPEmbedder

        self.model = SigLIPEmbedder(lora_path=None)
        self.texts = self.model.text_vectors(BOTTLE_PROMPTS + OTHER_PROMPTS)

    def __call__(self, images: list[Image.Image]) -> list[float]:
        import numpy as np

        logits = self.model.image_text_vectors(images) @ self.texts.T * 100.0
        logits -= logits.max(axis=1, keepdims=True)
        probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
        return probs[:, : len(BOTTLE_PROMPTS)].sum(axis=1).tolist()


def open_image(path: Path) -> Image.Image:
    return ImageOps.exif_transpose(Image.open(path)).convert("RGB")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump", type=Path, default=ROOT / "data/raw/catalog_dump.csv")
    parser.add_argument("--photos", type=Path, default=ROOT / "data/catalog_photos")
    parser.add_argument("--out", type=Path, default=ROOT / "data/catalog.csv")
    parser.add_argument("--report", type=Path, default=ROOT / "artifacts/catalog_report.txt")
    args = parser.parse_args()

    rows: dict[str, dict[str, str]] = {}
    with args.dump.open(encoding="utf-8-sig", newline="") as fh:
        for raw in csv.DictReader(fh):
            slug = (raw.get("Slug") or "").strip()
            if slug and slug not in rows:
                rows[slug] = {k: (v or "").strip() for k, v in raw.items()}
    print(f"Позиций в дампе: {len(rows)}")

    disk = [p.name for p in args.photos.iterdir() if p.is_file()]
    matches = match_photos({s: r.get(PHOTO_COLUMN, "") for s, r in rows.items()}, disk)

    wines = {
        slug: Wine(slug=slug, title=r.get("Название вина", ""), winery=r.get("Винодельня", ""),
                   grape=r.get("Сорт винограда", ""))
        for slug, r in rows.items()
    }
    matcher = TextMatcher(wines.values())
    ocr = build_text_extractor({"kind": "apple"})
    bottle = BottleScorer()
    report: list[str] = []

    chosen: dict[str, str] = {}
    for slug, match in matches.items():
        # Дубли загрузки: одинаковое содержимое под разными хэшами Strapi.
        unique: dict[str, str] = {}
        for name in match.files:
            unique.setdefault(content_hash(args.photos / name), name)
        files = list(unique.values())
        if len(files) == 1:
            chosen[slug] = files[0]
            continue

        images = [open_image(args.photos / name) for name in files]
        bottles = bottle(images)
        scored = []
        for name, image, is_bottle in zip(files, images, bottles, strict=True):
            text = matcher.score(ocr.extract(image), [slug]).get(slug, 0.0)
            # Фото блога отсекает «бутылочность», между бутылками решает текст.
            scored.append((text + is_bottle, text, is_bottle, name))
        scored.sort(reverse=True)
        chosen[slug] = scored[0][3]
        report.append(
            f"НЕОДНОЗНАЧНО {slug} ({rows[slug].get(PHOTO_COLUMN)}): выбран {scored[0][3]} "
            f"[ocr={scored[0][1]:.2f} bottle={scored[0][2]:.2f}], отклонены "
            + ", ".join(f"{n} [ocr={t:.2f} bottle={b:.2f}]" for _, t, b, n in scored[1:])
        )

    # Аудит всех эталонов: фото, не похожее на бутылку, почти наверняка ошибка
    # дампа, и такое вино не найдётся никогда. Решение — за человеком.
    slugs = list(chosen)
    for start in range(0, len(slugs), 32):
        batch = slugs[start : start + 32]
        scores = bottle([open_image(args.photos / chosen[s]) for s in batch])
        for slug, is_bottle in zip(batch, scores, strict=True):
            if is_bottle < 0.5:
                report.append(f"НЕ БУТЫЛКА {slug}: {chosen[slug]} [bottle={is_bottle:.2f}]")

    missing = [s for s in rows if s not in chosen]
    for slug in missing:
        report.append(f"НЕТ ФОТО {slug} ({rows[slug].get(PHOTO_COLUMN)})")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=[*DUMP_COLUMNS.values(), "image"])
        writer.writeheader()
        for slug, r in rows.items():
            out = {dst: r.get(src, "") for src, dst in DUMP_COLUMNS.items()}
            out["image"] = chosen.get(slug, "")
            writer.writerow(out)

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(report) + "\n", encoding="utf-8")
    ambiguous = sum(1 for line in report if line.startswith("НЕОДНОЗНАЧНО"))
    print(f"С фото: {len(chosen)}/{len(rows)}; спорных разрешено по OCR: {ambiguous}; "
          f"без фото: {len(missing)}")
    print(f"Каталог: {args.out}\nОтчёт: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
