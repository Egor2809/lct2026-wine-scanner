"""Данные дообучения в репозитории: разметка рамок, сплит LoRA, сборка датасета YOLO.

Сами полевые фото в репозиторий не входят, поэтому здесь проверяется то, что
лежит в git и что скрипты из него собирают, — без обучения и без моделей.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
BOXES = ROOT / "training" / "label_boxes"


def _script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_label_boxes_are_valid_yolo() -> None:
    files = {split: sorted((BOXES / split).glob("*.txt")) for split in ("train", "val")}
    assert (len(files["train"]), len(files["val"])) == (80, 20)
    assert not {f.stem for f in files["train"]} & {f.stem for f in files["val"]}
    for path in files["train"] + files["val"]:
        rows = [line.split() for line in path.read_text().splitlines() if line.strip()]
        assert rows, f"{path.name}: нет рамки"
        for cls, *box in rows:
            assert cls == "0" and len(box) == 4
            assert all(0.0 <= float(v) <= 1.0 for v in box), path.name


def test_lora_split_is_fixed() -> None:
    names = (ROOT / "training" / "lora_train_photos.txt").read_text().split()
    assert len(names) == len(set(names)) == 48


def test_detector_dataset_pairs_boxes_with_photos(tmp_path: Path) -> None:
    build_dataset = _script("train_label_detector").build_dataset
    boxes, photos = tmp_path / "boxes", tmp_path / "photos"
    for split, stem in (("train", "a"), ("val", "b")):
        (boxes / split).mkdir(parents=True)
        (boxes / split / f"{stem}.txt").write_text("0 0.5 0.5 0.2 0.3\n")
    photos.mkdir()
    Image.new("RGB", (40, 60)).save(photos / "a.webp")
    Image.new("RGB", (40, 60)).save(photos / "b.webp")

    data = build_dataset(boxes, photos, tmp_path / "dataset")
    assert "nc: 1" in data.read_text()
    assert (tmp_path / "dataset/images/train/a.jpg").exists()
    assert (tmp_path / "dataset/labels/val/b.txt").read_text() == "0 0.5 0.5 0.2 0.3\n"

    (photos / "b.webp").unlink()
    with pytest.raises(SystemExit, match="Нет фото"):
        build_dataset(boxes, photos, tmp_path / "again")
