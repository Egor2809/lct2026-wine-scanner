#!/usr/bin/env python3
"""Фото каталога -> превью для карточек UI (data/catalog_thumbs, ~23 МБ).

    python scripts/make_thumbnails.py

Исходные фото (2,1 ГБ) в репозиторий не входят, а карточка без фото — не
карточка. Превью лежат в репозитории и попадают в Docker-образ; распознаванию
они не нужны, эмбеддинги и SIFT считаются по исходникам.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--catalog", type=Path, default=ROOT / "data/catalog.csv")
    parser.add_argument("--photos", type=Path, default=ROOT / "data/catalog_photos")
    parser.add_argument("--out", type=Path, default=ROOT / "data/catalog_thumbs")
    parser.add_argument("--side", type=int, default=600)
    parser.add_argument("--quality", type=int, default=72)
    args = parser.parse_args()

    with args.catalog.open(encoding="utf-8") as f:
        names = sorted({row["image"] for row in csv.DictReader(f) if row.get("image")})
    args.out.mkdir(parents=True, exist_ok=True)

    made = skipped = missing = 0
    for name in names:
        src, dst = args.photos / name, args.out / name
        if not src.exists():
            missing += 1
            continue
        if dst.exists() and dst.stat().st_mtime >= src.stat().st_mtime:
            skipped += 1
            continue
        with Image.open(src) as im:
            im = ImageOps.exif_transpose(im)
            im.thumbnail((args.side, args.side))
            if im.mode not in ("RGB", "RGBA"):
                im = im.convert("RGBA" if "A" in im.getbands() else "RGB")
            im.save(dst, "WEBP", quality=args.quality, method=4)
        made += 1

    stale = {p.name for p in args.out.iterdir()} - set(names)
    for name in stale:
        (args.out / name).unlink()

    total = sum(p.stat().st_size for p in args.out.iterdir()) / 1e6
    print(f"превью: {made} новых, {skipped} без изменений, {len(stale)} удалено, "
          f"нет исходника {missing}; {total:.1f} МБ в {args.out}")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
