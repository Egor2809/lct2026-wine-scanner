"""Геометрическая проверка кандидатов: SIFT + RANSAC-гомография.

Работает только для пэкшотов (normalize.query: бутылка целиком на однотонном
фоне). Такой кадр — почти копия эталона каталога, и число геометрически
согласованных совпадений локальных признаков отделяет верную позицию от
соседей по серии (одна этикетка, другой сорт) надёжнее глобального вектора:
размытие, шум, тёмный свет и блик портят вектор целиком, а совпадения — лишь
частично. На фото с полки сигнал выключен: этикетки одной винодельни делят
логотипы и шрифты, ложных совпадений там столько же, сколько верных
(замер на полевых фото: 61/63 -> 59/63 уже при малом весе).

Признаки каталога считаются один раз при старте (потоками, OpenCV отпускает
GIL) и кэшируются на диск рядом с индексом.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from .fusion import Candidate, fuse

log = logging.getLogger(__name__)

_Features = tuple[np.ndarray, np.ndarray]  # (xy float32 N×2, дескрипторы uint8 N×128)


class GeometricVerifier:
    def __init__(
        self,
        images: dict[str, str],
        side: int = 640,
        nfeatures: int = 500,
        top_k: int = 10,
        weight: float = 1.0,
        text_weight: float | None = None,
        cap: int = 100,
        ratio: float = 0.75,
        reproj: float = 8.0,
        cache_path: Path | None = None,
        workers: int = 8,
    ) -> None:
        import cv2

        self._cv2 = cv2
        self.images = images
        self.side = side
        self.nfeatures = nfeatures
        self.top_k = top_k
        # Свой вес, а не fusion.weights.geometric: тот — для фото с полки, где
        # геометрия выключена. weight / cap — прибавка к скору за одно совпадение.
        self.weight = weight
        # Вес текста на пэкшоте (None — как в fusion.weights).
        self.text_weight = text_weight
        self.cap = cap
        self.ratio = ratio
        self.reproj = reproj
        self.cache_path = cache_path
        self.workers = workers
        self.features: dict[str, _Features] = {}

    # ── признаки ─────────────────────────────────────────────────────────

    def _extract(self, image: Image.Image) -> _Features:
        cv2 = self._cv2
        image = image.convert("L")
        if max(image.size) > self.side:
            image = image.copy()
            image.thumbnail((self.side, self.side))
        detector = cv2.SIFT_create(nfeatures=self.nfeatures)  # не потокобезопасен: свой на вызов
        keypoints, desc = detector.detectAndCompute(np.asarray(image), None)
        if desc is None or not keypoints:
            return np.zeros((0, 2), np.float32), np.zeros((0, 128), np.uint8)
        xy = np.float32([kp.pt for kp in keypoints])
        return xy, np.clip(desc, 0, 255).astype(np.uint8)

    def _extract_path(self, slug: str) -> tuple[str, _Features]:
        try:
            with Image.open(self.images[slug]) as image:
                return slug, self._extract(ImageOps.exif_transpose(image).convert("RGB"))
        except Exception as exc:  # битое эталонное фото не должно ронять старт
            log.warning("SIFT эталона %s не посчитан: %s", slug, exc)
            return slug, (np.zeros((0, 2), np.float32), np.zeros((0, 128), np.uint8))

    def prepare(self) -> None:
        """Признаки всех эталонов: из дискового кэша или расчётом (один раз)."""
        key = f"sift:{self.side}:{self.nfeatures}"
        if self.cache_path is not None and self.cache_path.exists():
            data = np.load(self.cache_path, allow_pickle=True)
            if str(data["key"]) == key:
                # Каждое data[...] заново распаковывает массив из npz целиком:
                # читать один раз, иначе 2096 срезов держат 2096 полных копий.
                slugs = [str(s) for s in data["slugs"]]
                bounds, xy, desc = data["offsets"], data["xy"], data["desc"]
                for i, slug in enumerate(slugs):
                    lo, hi = int(bounds[i]), int(bounds[i + 1])
                    self.features[slug] = (xy[lo:hi], desc[lo:hi])
        missing = [slug for slug in self.images if slug not in self.features]
        if missing:
            with ThreadPoolExecutor(self.workers) as pool:
                self.features.update(pool.map(self._extract_path, missing))
            if self.cache_path is not None:
                self._save(key)
        log.info("SIFT эталонов: %d (посчитано сейчас %d)", len(self.features), len(missing))

    def _save(self, key: str) -> None:
        slugs = list(self.features)
        sizes = [len(self.features[s][0]) for s in slugs]
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(  # 87 -> 63 МБ: кэш едет в образ и репозиторий
            self.cache_path,
            key=np.array(key),
            slugs=np.array(slugs, dtype=object),
            offsets=np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64),
            xy=np.concatenate([self.features[s][0] for s in slugs]),
            desc=np.concatenate([self.features[s][1] for s in slugs]),
        )

    def _catalog(self, slug: str) -> _Features | None:
        if slug not in self.features and slug in self.images:
            self.features[slug] = self._extract_path(slug)[1]
        return self.features.get(slug)

    # ── сопоставление ────────────────────────────────────────────────────

    def inliers(self, query: _Features, slug: str) -> int:
        cv2 = self._cv2
        ref = self._catalog(slug)
        if ref is None or len(query[0]) < 5 or len(ref[0]) < 5:
            return 0
        matcher = cv2.BFMatcher(cv2.NORM_L2)
        pairs = matcher.knnMatch(query[1].astype(np.float32), ref[1].astype(np.float32), k=2)
        good = [m for m, n in (p for p in pairs if len(p) == 2)
                if m.distance < self.ratio * n.distance]
        if len(good) < 6:
            return len(good) // 2
        src = query[0][[m.queryIdx for m in good]]
        dst = ref[0][[m.trainIdx for m in good]]
        _, mask = cv2.findHomography(src, dst, cv2.RANSAC, self.reproj)
        return int(mask.sum()) if mask is not None else 0

    def scores(self, image: Image.Image, slugs: Iterable[str]) -> dict[str, float]:
        """-> {slug: доля совпадений 0..1 (inliers / cap)}."""
        query = self._extract(image)
        return {slug: min(self.inliers(query, slug), self.cap) / self.cap for slug in slugs}

    def rerank(self, views, candidates: list[Candidate], weights: dict[str, float]
               ) -> list[Candidate]:
        """Пэкшот: геометрия для top_k кандидатов и повторное слияние. Иначе — как есть."""
        if not getattr(views, "packshot", False) or self.weight <= 0 or not candidates:
            return candidates
        weights = {**weights, "geometric": self.weight}
        if self.text_weight is not None:
            weights["textual"] = self.text_weight
        top = candidates[: self.top_k]
        found = self.scores(views.embed, [c.slug for c in top])
        for candidate in top:
            candidate.geometric = found.get(candidate.slug, 0.0)
        return fuse(candidates, weights)


def build_verifier(catalog, config: dict, artifacts: Path) -> GeometricVerifier | None:
    """Фабрика по секции rerank.geometric (по умолчанию включён, веса — замер на dev)."""
    if not config.get("enabled", True):
        return None
    images = {w.slug: w.image for w in catalog if w.image and Path(w.image).exists()}
    text_weight = config.get("text_weight", 0.1)
    verifier = GeometricVerifier(
        images,
        side=int(config.get("side", 640)),
        nfeatures=int(config.get("nfeatures", 500)),
        top_k=int(config.get("top_k", 10)),
        weight=float(config.get("weight", 1.0)),
        text_weight=None if text_weight is None else float(text_weight),
        cache_path=artifacts / "geometric_sift.npz",
    )
    verifier.prepare()
    return verifier
