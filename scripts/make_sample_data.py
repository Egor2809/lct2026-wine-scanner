#!/usr/bin/env python3
"""Синтетический каталог для разработки, пока не пришёл дамп заказчика.

Генерирует `data/catalog_sample.csv` и эталонные «этикетки» в
`data/catalog_images/`, плюс «полевые» фото (поворот, шум, затемнение,
перспектива) и `data/labels_sample.csv` для evaluate.py.

Нужен ровно для одного: чтобы пайплайн, индекс, API и стенд оценки были
проверены end-to-end до появления настоящих данных. На реальных цифрах
точности эти картинки ничего не говорят.
"""

from __future__ import annotations

import csv
import random
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

WINERIES = [
    ("Ведерниковъ", "vedernikov", "Ростовская область", (122, 32, 44)),
    ("Галицкий и Галицкий", "galitsky-galitsky", "Крымск", (30, 64, 88)),
    ("UPPA Winery", "uppa-winery", "Севастополь", (46, 84, 52)),
    ("Сикоры", "sikory", "Семигорье", (176, 142, 58)),
    ("Гунько Вайнери", "gunko-winery", "Саук-Дере", (92, 44, 96)),
]

WINES = [
    ("Красностоп Золотовский", "krasnostop-zolotovsky", "Красное сухое", "red", "Красностоп"),
    ("Сибирьковый", "sibirkovy", "Белое сухое", "white", "Сибирьковый"),
    ("Рислинг", "riesling", "Белое сухое", "white", "Рислинг"),
    ("Каберне Фран", "cabernet-franc", "Красное сухое", "red", "Каберне фран"),
    ("Пино Нуар", "pinot-noir", "Красное сухое", "red", "Пино нуар"),
    ("Шардоне", "chardonnay", "Белое сухое", "white", "Шардоне"),
    ("Брют Натюр", "brut-nature", "Игристое брют", "sparkling", "Шардоне, пино нуар"),
    ("Мерло", "merlot", "Красное сухое", "red", "Мерло"),
]

VINTAGES = ["2020", "2021", "2022", "2023"]


def label_image(seed: int, base: tuple[int, int, int], title: str, winery: str) -> Image.Image:
    """Студийная «этикетка»: узнаваемый цвет + геометрия, уникальная по seed."""
    rng = random.Random(seed)
    image = Image.new("RGB", (400, 600), (245, 242, 236))
    draw = ImageDraw.Draw(image)

    draw.rectangle([40, 60, 360, 540], fill=base)
    for _ in range(4):
        x0, y0 = rng.randint(60, 240), rng.randint(90, 420)
        shade = tuple(min(255, c + rng.randint(30, 90)) for c in base)
        draw.ellipse([x0, y0, x0 + rng.randint(60, 120), y0 + rng.randint(60, 120)], fill=shade)

    draw.rectangle([70, 240, 330, 300], fill=(245, 242, 236))
    draw.text((84, 258), title[:22], fill=base)
    draw.text((84, 330), winery[:24], fill=(245, 242, 236))
    return image


def field_photo(label: Image.Image, seed: int) -> Image.Image:
    """Имитация съёмки в магазине: поворот, перспектива, тусклый свет, шум."""
    rng = random.Random(seed * 7919)
    photo = Image.new("RGB", (700, 900), (40 + rng.randint(0, 30),) * 3)
    angle = rng.uniform(-12, 12)
    scaled = label.resize((320, 480)).rotate(angle, expand=True, fillcolor=(40, 40, 40))
    photo.paste(scaled, (rng.randint(120, 220), rng.randint(180, 280)))
    photo = photo.point(lambda v: int(v * rng.uniform(0.65, 0.9)))
    return photo.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 1.1)))


def main() -> int:
    data = ROOT / "data"
    images_dir = data / "catalog_images"
    field_dir = data / "field_photos"
    images_dir.mkdir(parents=True, exist_ok=True)
    field_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    labels: list[tuple[str, str]] = []
    seed = 0

    for winery, winery_slug, region, color in WINERIES:
        for title, wine_slug, style, kind, grape in WINES:
            # Два винтажа на часть позиций — те самые near-duplicates,
            # на которых визуальный эмбеддинг обязан ошибаться.
            for vintage in (VINTAGES if seed % 5 == 0 else VINTAGES[:1]):
                seed += 1
                slug = f"{winery_slug}-{wine_slug}-{vintage}"
                filename = f"{slug}.png"
                label = label_image(seed, color, title, winery)
                label.save(images_dir / filename)

                rows.append(
                    {
                        "slug": slug,
                        "name": f"{title} {vintage}",
                        "producer": winery,
                        "region": region,
                        "grape": grape,
                        "category": style,
                        "color": kind,
                        "year": vintage,
                        "description": f"{title} от «{winery}», {region}.",
                        "food_pairing": "Мясо; Сыры",
                        "image_url": filename,
                        "url": f"https://vino-svoe.ru/wines/{slug}",
                    }
                )

                if seed % 3 == 0:  # каждое третье вино попадает в hold-out
                    field_name = f"{slug}.jpg"
                    field_photo(label, seed).save(field_dir / field_name, quality=82)
                    labels.append((f"field_photos/{field_name}", slug))

    catalog_csv = data / "catalog_sample.csv"
    with catalog_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    labels_csv = data / "labels_sample.csv"
    with labels_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["image", "slug"])
        writer.writerows(labels)

    print(f"Каталог : {catalog_csv} ({len(rows)} позиций)")
    print(f"Hold-out: {labels_csv} ({len(labels)} фото)")
    print("\nДальше:\n  CATALOG_PATH=data/catalog_sample.csv python scripts/build_index.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
