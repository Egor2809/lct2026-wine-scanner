#!/usr/bin/env python3
"""Синтетический набор для контроля точности в CI: фото каталога, снятые «телефоном».

    python scripts/make_synthetic.py --n 40 --out /tmp/synth
    python scripts/evaluate.py --labels /tmp/synth/labels.csv --min-top1 0.9

Искажения: перспектива и поворот, кроп, свет и цвет, размытие, JPEG, фон вокруг
бутылки. Это не замена полевым фото (они в data/field/, в репозиторий не входят):
набор ловит регрессии пайплайна на каждом пуше, а не меряет точность на полке.
Seed фиксирован — набор одинаков от прогона к прогону.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter

ROOT = Path(__file__).resolve().parents[1]


def distort(image: Image.Image, rng: random.Random) -> Image.Image:
    w, h = image.size
    # Бутылка на фоне, как на полке или столе. Прозрачный фон пэкшота — по маске.
    pad = int(max(w, h) * rng.uniform(0.1, 0.35))
    background = tuple(rng.randint(40, 220) for _ in range(3))
    canvas = Image.new("RGB", (w + 2 * pad, h + 2 * pad), background)
    mask = image.getchannel("A") if "A" in image.getbands() else None
    canvas.paste(image.convert("RGB"), (pad, pad), mask)
    image = canvas
    w, h = image.size

    # Ракурс: сдвиг углов до 8 % стороны.
    d = [rng.uniform(-0.08, 0.08) * s for s in (w, h, w, h, w, h, w, h)]
    quad = (d[0], d[1], d[2], h + d[3], w + d[4], h + d[5], w + d[6], d[7])
    image = image.transform((w, h), Image.Transform.QUAD, quad, Image.Resampling.BICUBIC,
                            fillcolor=background)
    image = image.rotate(rng.uniform(-8, 8), Image.Resampling.BICUBIC, expand=False,
                         fillcolor=background)

    # Кадрирование: кадр не по центру.
    cx, cy = rng.uniform(0, 0.08) * w, rng.uniform(0, 0.08) * h
    image = image.crop((cx, cy, w - rng.uniform(0, 0.08) * w, h - rng.uniform(0, 0.08) * h))

    # Свет магазина: яркость, контраст, цветовая температура.
    image = ImageEnhance.Brightness(image).enhance(rng.uniform(0.6, 1.3))
    image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.7, 1.2))
    image = ImageEnhance.Color(image).enhance(rng.uniform(0.7, 1.2))
    r, g, b = image.split()
    warm = rng.uniform(0.9, 1.1)
    image = Image.merge("RGB", (r.point(lambda v: min(255, int(v * warm))), g,
                                b.point(lambda v: min(255, int(v / warm)))))
    if rng.random() < 0.6:
        image = image.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 1.5)))
    return image


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--catalog", type=Path, default=ROOT / "data/catalog.csv")
    parser.add_argument("--photos", type=Path, default=ROOT / "data/catalog_thumbs")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    with args.catalog.open(encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r.get("image") and (args.photos / r["image"]).exists()]
    rng = random.Random(args.seed)
    picked = rng.sample(rows, min(args.n, len(rows)))

    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "labels.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "slug"])
        for i, row in enumerate(picked):
            with Image.open(args.photos / row["image"]) as photo:
                shot = distort(photo, rng)
            name = f"{i:03d}.jpg"
            shot.save(args.out / name, "JPEG", quality=rng.randint(55, 85))
            writer.writerow([name, row["slug"]])
    print(f"синтетика: {len(picked)} фото -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
