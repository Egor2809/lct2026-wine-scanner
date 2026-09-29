#!/usr/bin/env python3
"""Дообучение LoRA-адаптера SigLIP 2 для фото запросов -> artifacts/models/siglip_lora_v2.pt.

    pip install -r requirements-train.txt
    python scripts/train_lora.py --out artifacts/models/siglip_lora_v2.pt
    python scripts/train_lora.py --eval-only artifacts/models/siglip_lora_v2.pt

Идея: в каталоге одно эталонное фото на вино, а запросы — снимки с полки.
Адаптер LoRA (r=8, alpha=16) на attention-проекциях vision-башни учится
приближать вектор полевого фото к ЗАМОРОЖЕННОМУ вектору его эталона:

    loss = CrossEntropy(q @ V.T / tau, индекс эталона),  tau = 0.07

V — векторы всего каталога из artifacts/index.npz (InfoNCE по ~2100 «классам»).
Индекс не пересобирается: адаптер включается только на запросах (features/lora.py).

Данные — полевые фото кейса (data/field/, из архива кейса, в репозиторий не
входят) и разметка data/field/labels.csv. Сплит закреплён в репозитории:
training/lora_train_photos.txt — 48 фото обучения; остальные размеченные фото
(15) — валидация, по ним же scripts/eval_offline.py считает честную строку
«только фото вне обучения LoRA». Запрос нормализуется тем же QueryNormalizer,
что и в сервисе (кроп этикетки YOLOv8n, баланс белого).

Происхождение весов: siglip_lora_v2.pt в репозитории получен этим рецептом
(гиперпараметры, аугментации и сплит те же) в исходном прототипе команды до
переноса в слои MOS; побайтного совпадения при повторе не будет — меняется
матрица каталога (после чистки каталога) и недетерминированность GPU/MPS.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

RANK, ALPHA = 8, 16.0
LR, WD = 1e-4, 1e-4
BATCH, STEPS, EVAL_EVERY, PATIENCE = 12, 400, 50, 3
TAU = 0.07
SEED = 42


def load_pairs(labels: Path, photos: Path, slug_row: dict[str, int]) -> list[dict]:
    pairs = []
    with labels.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            slug = row["slug"].strip()
            if slug in slug_row and (photos / row["image"]).exists():
                pairs.append({"name": row["image"], "path": photos / row["image"],
                              "target": slug_row[slug]})
    return pairs


def augmenter():
    """Телефонные искажения: ракурс, кроп, свет, шум, размытие, JPEG."""
    import albumentations as A

    return A.Compose([
        A.RandomResizedCrop(size=(480, 480), scale=(0.6, 1.0), ratio=(0.8, 1.25), p=0.9),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.8),
        A.HueSaturationValue(hue_shift_limit=12, sat_shift_limit=30, val_shift_limit=25, p=0.4),
        A.GaussNoise(std_range=(0.02, 0.10), mean_range=(0.0, 0.0), per_channel=True, p=0.4),
        A.GaussianBlur(blur_limit=(3, 5), p=0.25),
        A.Affine(scale=(0.85, 1.15), translate_percent=(-0.06, 0.06), rotate=(-15, 15), p=0.7),
        A.Perspective(scale=(0.03, 0.10), fit_output=False, p=0.4),
        A.ImageCompression(compression_type="jpeg", quality_range=(55, 95), p=0.3),
    ])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--labels", type=Path, default=ROOT / "data/field/labels.csv")
    parser.add_argument("--photos", type=Path, default=ROOT / "data/field")
    parser.add_argument("--split", type=Path, default=ROOT / "training/lora_train_photos.txt")
    parser.add_argument("--index", type=Path, default=ROOT / "artifacts/index.npz")
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--out", type=Path,
                        help="куда сохранить адаптер (боевой: artifacts/models/siglip_lora_v2.pt)")
    target.add_argument("--eval-only", type=Path,
                        help="не обучать: top-1/top-5 готового адаптера на той же валидации")
    parser.add_argument("--steps", type=int, default=STEPS)
    args = parser.parse_args()

    import torch
    import torch.nn.functional as F
    from PIL import Image
    from transformers import AutoModel, AutoProcessor

    from wine_scanner.features.lora import LoraSwitch, wrap_vision
    from wine_scanner.normalize.query import QueryNormalizer
    from wine_scanner.search.index import NumpyIndex
    from wine_scanner.settings import get_config

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    config = get_config()
    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")

    # Каталог — заморожен: те же векторы, по которым ищет сервис.
    index = NumpyIndex.load(args.index)
    slug_row = {slug: i for i, slug in reversed(list(enumerate(index._slugs)))}
    catalog = torch.from_numpy(index._vectors).to(device)

    pairs = load_pairs(args.labels, args.photos, slug_row)
    train_names = set(args.split.read_text(encoding="utf-8").split())
    train = [p for p in pairs if p["name"] in train_names]
    val = [p for p in pairs if p["name"] not in train_names]
    print(f"[data] каталог {tuple(catalog.shape)}, пар {len(pairs)}: "
          f"обучение {len(train)}, валидация {len(val)}")
    if not train:
        print("Нет фото обучения: положите полевые фото кейса в data/field/")
        return 1

    # Запрос — ровно тот вид, что уходит в эмбеддер сервиса.
    normalizer = QueryNormalizer(config.get("normalize") or {}, ROOT)
    for pair in pairs:
        pair["view"] = normalizer(pair["path"].read_bytes()).embed.convert("RGB")

    # fp32: в fp16 градиенты LoRA нестабильны (NaN).
    model_id = config.get_path("embedder.siglip.model_id")
    model = AutoModel.from_pretrained(model_id, dtype=torch.float32).to(device).eval()
    processor = AutoProcessor.from_pretrained(model_id)
    for parameter in model.parameters():
        parameter.requires_grad = False
    layers = wrap_vision(model, LoraSwitch(), r=RANK, alpha=ALPHA)
    trainable = [p for _, _, layer in layers for p in (layer.lora_A, layer.lora_B)]
    vision = model.vision_model
    vision.train()

    if args.eval_only:
        state = torch.load(args.eval_only, map_location="cpu", weights_only=True)
        with torch.no_grad():
            for i, name, layer in layers:
                layer.lora_A.copy_(state[f"{i}.{name}.lora_A"])
                layer.lora_B.copy_(state[f"{i}.{name}.lora_B"])

    aug = augmenter()

    def embed(images, grad: bool):
        pixels = processor(images=images, return_tensors="pt")["pixel_values"].to(device)
        with torch.set_grad_enabled(grad):
            return F.normalize(vision(pixel_values=pixels).pooler_output, dim=1)

    def validate() -> tuple[float, float]:
        if not val:
            return 0.0, 0.0
        vision.eval()
        with torch.no_grad():
            sims = embed([p["view"] for p in val], grad=False) @ catalog.T
        vision.train()
        top = torch.argsort(-sims, dim=1)[:, :5].tolist()
        hit1 = sum(p["target"] == t[0] for p, t in zip(val, top, strict=True))
        hit5 = sum(p["target"] in t for p, t in zip(val, top, strict=True))
        return hit1 / len(val), hit5 / len(val)

    def save() -> None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        torch.save({f"{i}.{name}.{part}": getattr(layer, part).detach().cpu()
                    for i, name, layer in layers for part in ("lora_A", "lora_B")}, args.out)

    if args.eval_only:
        top1, top5 = validate()
        print(f"{args.eval_only.name}: валидация top-1={top1:.1%} top-5={top5:.1%} (n={len(val)})")
        return 0

    optimizer = torch.optim.AdamW(trainable, lr=LR, weight_decay=WD)
    best, bad, started = -1.0, 0, time.perf_counter()
    for step in range(1, args.steps + 1):
        batch = random.sample(train, min(BATCH, len(train)))
        images = [Image.fromarray(aug(image=np.asarray(p["view"]))["image"]) for p in batch]
        logits = embed(images, grad=True) @ catalog.T / TAU
        targets = torch.tensor([p["target"] for p in batch], device=device)
        loss = F.cross_entropy(logits, targets)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, max_norm=5.0)
        optimizer.step()

        if step % 10 == 0:
            accuracy = (logits.argmax(1) == targets).float().mean().item()
            print(f"  шаг {step}/{args.steps} loss={loss.item():.3f} "
                  f"batch_acc={accuracy:.2f} ({time.perf_counter() - started:.0f} c)", flush=True)
        if step % EVAL_EVERY == 0 or step == args.steps:
            top1, top5 = validate()
            print(f"  >>> валидация top-1={top1:.1%} top-5={top5:.1%} (шаг {step})")
            if top1 > best:
                best, bad = top1, 0
                save()
                print(f"  лучший чекпойнт -> {args.out}")
            else:
                bad += 1
                if bad >= PATIENCE:
                    print(f"[train] ранняя остановка на шаге {step}")
                    break

    print(f"[train] {time.perf_counter() - started:.0f} c, лучший top-1 на валидации {best:.1%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
