#!/usr/bin/env python3
"""Дообучение YOLOv8n — детектор этикетки для кропа запроса -> artifacts/models/label_yolov8n.pt.

    pip install -r requirements-train.txt
    python scripts/train_label_detector.py --out artifacts/models/label_yolov8n.pt

Один класс «label» — лицевая этикетка центральной бутылки. Разметка рамок
сделана командой по полевым фото кейса и лежит в репозитории:
training/label_boxes/{train,val}/<фото>.txt (формат YOLO, 80 + 20 фото).
Сами фото — data/field/ из архива кейса. Скрипт собирает из них датасет
ultralytics во временной папке и обучает с параметрами исходного прогона
(yolov8n.pt, 50 эпох, 640 px, batch 8, seed 0, deterministic).

Происхождение весов: label_yolov8n.pt в репозитории — best.pt этого прогона
в исходном прототипе команды (обучение на GPU); на CPU/MPS результат
повторяется с точностью до недетерминированности железа.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]

TRAIN_ARGS = {"epochs": 50, "imgsz": 640, "batch": 8, "seed": 0, "deterministic": True,
              "workers": 2, "patience": 100, "pretrained": True}


def build_dataset(boxes: Path, photos: Path, work: Path) -> Path:
    """training/label_boxes + data/field -> датасет ultralytics (images/labels, data.yaml)."""
    missing = []
    for split in ("train", "val"):
        (work / "images" / split).mkdir(parents=True)
        (work / "labels" / split).mkdir(parents=True)
        for label in sorted((boxes / split).glob("*.txt")):
            found = sorted(photos.glob(label.stem + ".*"))
            found = [p for p in found if p.suffix.lower() != ".txt"]
            if not found:
                missing.append(label.stem)
                continue
            # Разметка сделана по кадру с учётом EXIF-поворота.
            with Image.open(found[0]) as image:
                ImageOps.exif_transpose(image).convert("RGB").save(
                    work / "images" / split / f"{label.stem}.jpg", quality=95)
            shutil.copy(label, work / "labels" / split / label.name)
    if missing:
        raise SystemExit(f"Нет фото для {len(missing)} рамок (например {missing[0]}): "
                         f"положите полевые фото кейса в {photos}")
    data = work / "data.yaml"
    data.write_text(f"path: {work}\ntrain: images/train\nval: images/val\n"
                    f"nc: 1\nnames: ['label']\n", encoding="utf-8")
    return data


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--boxes", type=Path, default=ROOT / "training/label_boxes")
    parser.add_argument("--photos", type=Path, default=ROOT / "data/field")
    parser.add_argument("--out", type=Path, required=True,
                        help="куда сохранить веса (боевые: artifacts/models/label_yolov8n.pt)")
    parser.add_argument("--epochs", type=int, default=TRAIN_ARGS["epochs"])
    parser.add_argument("--device", default=None, help="cuda:0 | mps | cpu; по умолчанию сам")
    args = parser.parse_args()

    from ultralytics import YOLO

    with tempfile.TemporaryDirectory(prefix="label-yolo-") as tmp:
        work = Path(tmp)
        data = build_dataset(args.boxes, args.photos, work / "dataset")
        # Предобученный yolov8n.pt ultralytics скачивает в текущую папку — пусть во временную.
        args.out = args.out.resolve()
        os.chdir(work)
        model = YOLO("yolov8n.pt")
        train_args = {**TRAIN_ARGS, "epochs": args.epochs}
        if args.device:
            train_args["device"] = args.device
        model.train(data=str(data), project=str(work / "runs"), name="label", **train_args)
        best = work / "runs" / "label" / "weights" / "best.pt"
        metrics = YOLO(str(best)).val(data=str(data), split="val", verbose=False)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(best, args.out)
    print(f"детектор этикетки: mAP50 {metrics.box.map50:.3f}, mAP50-95 {metrics.box.map:.3f} "
          f"на валидации -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
