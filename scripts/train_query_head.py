"""Дообучение адаптера запросов: лёгкая голова поверх замороженного SigLIP 2.

Своих фото вин нет, поэтому обучаемся на синтетике из фото каталога:
бутылку вырезаем с белого фона, кладём на кусок реальной полки (полевые фото
вин ВНЕ каталога) и искажаем под съёмку в магазине (перспектива, блик, темнота,
смаз, шум, JPEG). Дальше — тот же QueryNormalizer (YOLO-кроп этикетки), что и в
сервисе, и базовый SigLIP. Голова учится сводить вектор такого кадра к вектору
эталона каталога (InfoNCE против замороженного индекса).

Полевые фото вин каталога в обучение не идут вовсе: все 63 — честный тест
на реальных снимках. Вина скрытого тестового набора (JSON со slug, путь в
переменной HIDDEN_LABELS) и отложенная часть каталога (val) тоже исключены.

  python scripts/train_query_head.py embed --per-wine 3   # тяжёлое, один раз, с докачкой
  python scripts/train_query_head.py train                 # секунды, CPU
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import zlib
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wine_scanner.catalog.loader import load_catalog  # noqa: E402
from wine_scanner.catalog.models import same_wine  # noqa: E402
from wine_scanner.normalize.query import QueryNormalizer  # noqa: E402
from wine_scanner.settings import get_config  # noqa: E402

CACHE = ROOT / "artifacts" / "train_cache"
EVAL_CACHE = ROOT / "artifacts" / "eval_cache"
HIDDEN_LABELS = Path(os.environ.get("HIDDEN_LABELS", ROOT / "data" / "hidden" / "labels.json"))
OUT = ROOT / "artifacts" / "models" / "query_head.npz"
SEED = 28
VAL_SHARE = 0.1


def split_wines(wines):
    """-> (train, val). Вина скрытого тестового набора не попадают никуда."""
    hidden = set()
    if HIDDEN_LABELS.exists():
        hidden = {r["slug"] for r in json.loads(HIDDEN_LABELS.read_text())}
    pool = sorted((w for w in wines if w.slug not in hidden), key=lambda w: w.slug)
    rng = np.random.default_rng(SEED)
    is_val = rng.random(len(pool)) < VAL_SHARE
    return ([w for w, v in zip(pool, is_val, strict=True) if not v],
            [w for w, v in zip(pool, is_val, strict=True) if v])


def shelf_backgrounds() -> list[np.ndarray]:
    names = [r["image"] for r in csv.DictReader(
        (ROOT / "data/field/out_of_catalog.csv").open(encoding="utf-8"))]
    out = []
    for name in names:
        path = ROOT / "data/field" / name
        if path.exists():
            img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
            img.thumbnail((1280, 1280))
            out.append(np.array(img))
    return out


def bottle_on_shelf(rgb: np.ndarray, shelf: np.ndarray, rng):
    """Бутылка каталога поверх размытого куска реальной полки.

    Фон эталона (белый, чёрный, серый) оценивается по краям кадра и вырезается.
    Бутылка — главный объект кадра, как на снимке покупателя; полка размыта,
    чтобы детектор не цеплялся за этикетки соседей. -> (кадр, рамка бутылки x0,y0,x1,y1).
    """
    h, w = rgb.shape[:2]
    border = np.concatenate([rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]]).astype(np.float32)
    bg_color = np.median(border, axis=0)
    mask = (np.abs(rgb.astype(np.float32) - bg_color).max(axis=2) > 28).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    ys, xs = np.nonzero(mask)
    if len(ys) > 0.05 * h * w:  # эталон с полями: оставляем саму бутылку
        m = 4
        y0, y1 = max(ys.min() - m, 0), min(ys.max() + m + 1, h)
        x0, x1 = max(xs.min() - m, 0), min(xs.max() + m + 1, w)
        rgb, mask = rgb[y0:y1, x0:x1], mask[y0:y1, x0:x1]
        h, w = rgb.shape[:2]
    mask = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), 1.2)[..., None]
    H = int(h * rng.uniform(1.05, 1.35))
    W = int(max(w * rng.uniform(1.2, 2.2), H * rng.uniform(0.55, 0.8)))
    sh, sw = shelf.shape[:2]
    ch, cw = int(sh * rng.uniform(0.2, 0.5)), int(sw * rng.uniform(0.2, 0.5))
    y0, x0 = int(rng.integers(0, sh - ch + 1)), int(rng.integers(0, sw - cw + 1))
    bg = cv2.resize(shelf[y0:y0 + ch, x0:x0 + cw], (W, H), interpolation=cv2.INTER_LINEAR)
    bg = cv2.GaussianBlur(bg, (0, 0), rng.uniform(4.0, 10.0))
    ox, oy = int(rng.integers(0, W - w + 1)), int(rng.integers(0, H - h + 1))
    patch = bg[oy:oy + h, ox:ox + w].astype(np.float32)
    bg[oy:oy + h, ox:ox + w] = (mask * rgb + (1 - mask) * patch).astype(np.uint8)
    return bg, (ox, oy, ox + w, oy + h)


def distort(img: np.ndarray, rng) -> np.ndarray:
    h, w = img.shape[:2]
    d = rng.uniform(0.0, 0.10, (4, 2)) * [w, h]
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32(src + d * [[1, 1], [-1, 1], [-1, -1], [1, -1]])
    img = cv2.warpPerspective(img, cv2.getPerspectiveTransform(src, dst), (w, h),
                              borderMode=cv2.BORDER_REPLICATE)
    out = img.astype(np.float32)
    if rng.random() < 0.4:  # блик
        yy, xx = np.mgrid[0:h, 0:w]
        gx, gy = rng.uniform(0.2, 0.8) * w, rng.uniform(0.2, 0.8) * h
        r = rng.uniform(0.03, 0.15) * max(h, w)
        out += 255 * rng.uniform(0.3, 0.8) * np.exp(
            -((xx - gx) ** 2 + (yy - gy) ** 2) / (2 * r * r))[..., None]
    if rng.random() < 0.6:  # магазинный свет: темнее, теплее/холоднее
        out *= rng.uniform(0.45, 1.05)
        out *= np.array(rng.uniform(0.85, 1.12, 3))
    out = np.clip(out, 0, 255).astype(np.uint8)
    if rng.random() < 0.5:
        if rng.random() < 0.5:
            out = cv2.GaussianBlur(out, (0, 0), rng.uniform(0.6, 2.0))
        else:
            k = int(rng.integers(5, 11))
            kern = np.zeros((k, k), np.float32)
            kern[k // 2, :] = 1.0 / k
            m = cv2.getRotationMatrix2D((k / 2, k / 2), rng.uniform(0, 180), 1)
            out = cv2.filter2D(out, -1, cv2.warpAffine(kern, m, (k, k)))
    out = np.clip(out + rng.normal(0, rng.uniform(0, 6), out.shape), 0, 255).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", out[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY,
                                                   int(rng.integers(45, 90))])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)[..., ::-1]


def cmd_embed(args) -> int:
    from wine_scanner.features.embedder import SigLIPEmbedder

    CACHE.mkdir(parents=True, exist_ok=True)
    wines = [w for w in load_catalog() if w.image and Path(w.image).exists()]
    train, val = split_wines(wines)
    jobs = [(w.slug, "train", k) for w in train for k in range(args.per_wine)]
    jobs += [(w.slug, "val", 0) for w in val]
    by_slug = {w.slug: w for w in wines}
    shelves = shelf_backgrounds()
    normalizer = QueryNormalizer(get_config().get("normalize") or {}, ROOT)
    embedder = SigLIPEmbedder(batch_size=8)
    chunk = 400
    for start in range(0, len(jobs), chunk):
        part = CACHE / f"synth_{args.per_wine}_{start:06d}.npz"
        if part.exists():
            continue
        vecs, slugs, splits, fallback = [], [], [], 0
        for slug, split, k in jobs[start:start + chunk]:
            rng = np.random.default_rng(zlib.crc32(f"{slug}|{k}|{SEED}".encode()))
            rgb = np.array(ImageOps.exif_transpose(Image.open(by_slug[slug].image)).convert("RGB"))
            if max(rgb.shape[:2]) > 1000:
                f = 1000 / max(rgb.shape[:2])
                rgb = cv2.resize(rgb, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
            img, (bx0, by0, bx1, by1) = bottle_on_shelf(
                rgb, shelves[int(rng.integers(len(shelves)))], rng)
            img = distort(img, rng)
            f = min(1.0, 1280 / max(img.shape[:2]))  # ниже порога ресайза нормализатора
            if f < 1.0:
                img = cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
                bx0, by0, bx1, by1 = (int(v * f) for v in (bx0, by0, bx1, by1))
            views = normalizer.from_image(Image.fromarray(img))
            box = getattr(views, "box", None)
            if box is None or not (bx0 <= (box.x1 + box.x2) / 2 <= bx1
                                   and by0 <= (box.y1 + box.y2) / 2 <= by1):
                # детектор взял не нашу бутылку: кроп по самой бутылке
                views = normalizer.from_image(Image.fromarray(img[by0:by1, bx0:bx1]))
                fallback += 1
            vecs.append(embedder.encode([views.embed], role="catalog")[0].astype(np.float16))
            slugs.append(slug)
            splits.append(split)
        np.savez(part, vectors=np.stack(vecs), slugs=np.array(slugs), splits=np.array(splits))
        print(f"  эмбеддинги {min(start + chunk, len(jobs))}/{len(jobs)} "
              f"(кроп по бутылке: {fallback})", flush=True)
    return 0


def load_synth(per_wine: int):
    vecs, slugs, splits = [], [], []
    for part in sorted(CACHE.glob(f"synth_{per_wine}_*.npz")):
        with np.load(part) as data:  # один раз на файл, без чтения в цикле
            vecs.append(data["vectors"].astype(np.float32))
            slugs += data["slugs"].tolist()
            splits += data["splits"].tolist()
    return np.concatenate(vecs), slugs, np.array(splits)


def apply_head(q: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    out = q + q @ weight + bias
    return out / np.linalg.norm(out, axis=1, keepdims=True)


def top1(queries, truths, cat, cat_slugs) -> float:
    pred = np.argmax(queries @ cat.T, axis=1)
    return float(np.mean([same_wine(cat_slugs[p], t) for p, t in zip(pred, truths, strict=True)]))


def cmd_train(args) -> int:
    import torch

    torch.manual_seed(SEED)
    with np.load(EVAL_CACHE / "catalog_base.npz", allow_pickle=True) as data:
        cat, cat_slugs = data["vectors"].astype(np.float32), [str(s) for s in data["slugs"]]
    pos = {s: i for i, s in enumerate(cat_slugs)}
    x, slugs, splits = load_synth(args.per_wine)
    keep = np.array([s in pos for s in slugs])
    x, slugs, splits = x[keep], [s for s, k in zip(slugs, keep, strict=True) if k], splits[keep]
    y = np.array([pos[s] for s in slugs])
    tr, va = splits == "train", splits == "val"

    with (ROOT / "data/field/labels.csv").open(encoding="utf-8") as fh:
        field_truth = [r["slug"] for r in csv.DictReader(fh)]
    with np.load(EVAL_CACHE / "queries.npz", allow_pickle=True) as data:
        field_base, field_lora = data["base"].astype(np.float32), data["lora"].astype(np.float32)

    print(f"синтетика: train {tr.sum()}, val {va.sum()} (вина val в обучение не входят)")
    print(f"база:  val {top1(x[va], np.array(slugs)[va], cat, cat_slugs):.1%}  "
          f"полевые {top1(field_base, field_truth, cat, cat_slugs):.1%}  "
          f"(старая LoRA на полевых {top1(field_lora, field_truth, cat, cat_slugs):.1%}, "
          f"видела 48 из них)")

    d = x.shape[1]
    weight = torch.zeros(d, d, requires_grad=True)
    bias = torch.zeros(d, requires_grad=True)
    C = torch.from_numpy(cat)
    X, Y = torch.from_numpy(x[tr]), torch.from_numpy(y[tr])
    opt = torch.optim.AdamW([weight, bias], lr=args.lr, weight_decay=args.wd)
    best = (-1.0, None)
    for epoch in range(args.epochs):
        perm = torch.randperm(len(X))
        for start in range(0, len(X), 256):
            idx = perm[start:start + 256]
            q = X[idx] + X[idx] @ weight + bias
            q = q / q.norm(dim=1, keepdim=True)
            loss = torch.nn.functional.cross_entropy(q @ C.T / args.tau, Y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        w, b = weight.detach().numpy(), bias.detach().numpy()
        v = top1(apply_head(x[va], w, b), np.array(slugs)[va], cat, cat_slugs)
        f = top1(apply_head(field_base, w, b), field_truth, cat, cat_slugs)
        print(f"  эпоха {epoch + 1}: loss {loss.item():.3f}  val {v:.1%}  полевые {f:.1%}")
        if v > best[0]:  # выбор по синтетическому val, не по полевым
            best = (v, (w.copy(), b.copy(), epoch + 1))
    w, b, epoch = best[1]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT, weight=w.astype(np.float32), bias=b.astype(np.float32))
    print(f"сохранено {OUT.relative_to(ROOT)} (эпоха {epoch}, val {best[0]:.1%}, полевые "
          f"{top1(apply_head(field_base, w, b), field_truth, cat, cat_slugs):.1%})")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["embed", "train"])
    parser.add_argument("--per-wine", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=1e-2)
    parser.add_argument("--tau", type=float, default=0.05)
    args = parser.parse_args()
    return cmd_embed(args) if args.stage == "embed" else cmd_train(args)


if __name__ == "__main__":
    raise SystemExit(main())
