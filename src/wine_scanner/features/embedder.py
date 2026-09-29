"""Визуальный эмбеддер.

Единственный контракт — протокол `Embedder`: изображения на входе, матрица
(N, dim) L2-нормированных векторов на выходе. Пайплайн знает только его.

`role` различает два потока: эталоны каталога (студийные фото) и фото
запроса (телефон у полки). LoRA-адаптер обучен сдвигать именно фото запроса
к эталонам, поэтому включается только для role="query"; каталог кодируется
базовой моделью.

- `siglip`    — SigLIP 2 (+ опциональный LoRA на запросах), рабочий режим;
- `histogram` — цветовая гистограмма, нижний baseline без нейросети.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal, Protocol, runtime_checkable

import numpy as np
from PIL import Image

log = logging.getLogger(__name__)

Role = Literal["query", "catalog"]


@runtime_checkable
class Embedder(Protocol):
    name: str
    dim: int

    def encode(self, images: list[Image.Image], role: Role = "query") -> np.ndarray:
        """-> (len(images), dim), float32, каждая строка L2-нормирована."""
        ...


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """L2-нормировка по строкам.

    Отдельная функция, потому что это самое частое место тихой деградации:
    косинусный поиск по ненормированным векторам превращается в скалярное
    произведение и молча ранжирует по яркости картинки.
    """
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim == 1:
        vectors = vectors[None, :]
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)


class HistogramEmbedder:
    """Цветовая гистограмма по сетке grid x grid.

    Сетка нужна, чтобы вектор нёс хоть какую-то геометрию: у этикеток верх и
    низ различаются, а глобальная гистограмма это теряет.
    """

    name = "histogram"

    def __init__(self, bins: int = 8, grid: int = 2) -> None:
        self.bins = bins
        self.grid = grid
        self.dim = bins**3 * grid**2

    def encode(self, images: list[Image.Image], role: Role = "query") -> np.ndarray:
        if not images:
            return np.zeros((0, self.dim), dtype=np.float32)
        return l2_normalize(np.stack([self._one(image) for image in images]))

    def _one(self, image: Image.Image) -> np.ndarray:
        # Фиксированный размер: гистограмма не должна зависеть от разрешения.
        side = 64 * self.grid
        array = np.asarray(image.convert("RGB").resize((side, side), Image.BILINEAR))
        quantized = (array.astype(np.int32) * self.bins) // 256
        codes = (
            quantized[:, :, 0] * self.bins**2
            + quantized[:, :, 1] * self.bins
            + quantized[:, :, 2]
        )

        cell = side // self.grid
        parts = []
        for row in range(self.grid):
            for col in range(self.grid):
                patch = codes[row * cell : (row + 1) * cell, col * cell : (col + 1) * cell]
                counts = np.bincount(patch.ravel(), minlength=self.bins**3)
                parts.append(counts.astype(np.float32) / max(patch.size, 1))
        return np.concatenate(parts)


def pick_device(requested: str = "auto") -> str:
    import torch

    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class SigLIPEmbedder:
    """SigLIP 2 vision tower; LoRA-адаптер на attention-проекциях — только для запросов.

    fp16 — только на CUDA: на CPU половинная точность в PyTorch медленнее fp32.
    """

    def __init__(
        self,
        model_id: str = "google/siglip2-base-patch16-384",
        device: str = "auto",
        batch_size: int = 16,
        lora_path: str | Path | None = None,
        max_side: int = 800,
        threads: int = 0,
        query_head_path: str | Path | None = None,
    ) -> None:
        import torch
        from transformers import AutoModel, AutoProcessor

        from .lora import apply_lora

        if threads:
            torch.set_num_threads(threads)
        self._torch = torch
        self.device = pick_device(device)
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = AutoModel.from_pretrained(model_id, dtype=dtype).to(self.device).eval()
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.batch_size = batch_size
        self.max_side = max_side
        self.dim = int(self.model.config.vision_config.hidden_size)
        # Имя фиксирует пространство КАТАЛОГА: индекс не зависит от адаптера.
        self.name = f"siglip:{model_id}"

        self.lora = None
        if lora_path and Path(lora_path).exists():
            self.lora = apply_lora(self.model, Path(lora_path))
            log.info("LoRA для запросов: %s", lora_path)
        elif lora_path:
            log.warning("LoRA не найдена (%s): запросы кодируются базовой моделью", lora_path)
        self.query_name = self.name + (f"+lora:{Path(lora_path).name}" if self.lora else "")

        # Голова запросов (scripts/train_query_head.py): q + qW + b поверх
        # вектора запроса, обучена на синтетике «бутылка на полке» из каталога.
        self.query_head = None
        if query_head_path and Path(query_head_path).exists():
            with np.load(query_head_path) as data:
                self.query_head = (data["weight"].astype(np.float32),
                                   data["bias"].astype(np.float32))
            self.query_name += f"+head:{Path(query_head_path).name}"
            log.info("Голова запросов: %s", query_head_path)
        elif query_head_path:
            log.warning("Голова запросов не найдена (%s): без неё", query_head_path)
        log.info("SigLIP %s на %s, dim=%d", model_id, self.device, self.dim)

    def encode(self, images: list[Image.Image], role: Role = "query") -> np.ndarray:
        if not images:
            return np.zeros((0, self.dim), dtype=np.float32)
        if self.lora is not None:
            self.lora.enabled = role == "query"

        out = []
        with self._torch.inference_mode():
            for start in range(0, len(images), self.batch_size):
                chunk = []
                for image in images[start : start + self.batch_size]:
                    image = image.convert("RGB")
                    if max(image.size) > self.max_side:
                        # Дешёвый предресайз: процессор всё равно сожмёт до 384,
                        # а бикубика по 12-мегапиксельному кадру на CPU дорогая.
                        image = image.copy()
                        image.thumbnail((self.max_side, self.max_side))
                    chunk.append(image)
                pixels = self.processor(images=chunk, return_tensors="pt")["pixel_values"]
                pixels = pixels.to(self.device, dtype=self.model.dtype)
                pooled = self.model.vision_model(pixel_values=pixels).pooler_output
                out.append(pooled.float().cpu().numpy())
        vectors = l2_normalize(np.concatenate(out))
        if role == "query" and self.query_head is not None:
            weight, bias = self.query_head
            vectors = l2_normalize(vectors + vectors @ weight + bias)
        return vectors

    def text_vectors(self, texts: list[str]) -> np.ndarray:
        """Текстовая башня SigLIP — для zero-shot проверок каталога (бутылка ли на фото)."""
        inputs = self.processor(text=texts, padding="max_length", max_length=64,
                                return_tensors="pt").to(self.device)
        with self._torch.inference_mode():
            vectors = _as_tensor(self.model.get_text_features(**inputs))
        return l2_normalize(vectors.float().cpu().numpy())

    def image_text_vectors(self, images: list[Image.Image]) -> np.ndarray:
        """Эмбеддинг изображения в общем с текстом пространстве (с проекционной головой)."""
        if self.lora is not None:
            self.lora.enabled = False
        pixels = self.processor(images=[i.convert("RGB") for i in images],
                                return_tensors="pt")["pixel_values"]
        pixels = pixels.to(self.device, dtype=self.model.dtype)
        with self._torch.inference_mode():
            vectors = _as_tensor(self.model.get_image_features(pixel_values=pixels))
        return l2_normalize(vectors.float().cpu().numpy())


def _as_tensor(output):
    """transformers 5 возвращает из get_*_features объект вывода, 4.x — тензор."""
    return getattr(output, "pooler_output", output)


def build_embedder(config: dict) -> Embedder:
    """Фабрика по секции `embedder` из config.yaml."""
    kind = (config.get("kind") or "histogram").lower()

    if kind == "histogram":
        params = config.get("histogram") or {}
        embedder: Embedder = HistogramEmbedder(
            bins=int(params.get("bins", 8)),
            grid=int(params.get("grid", 2)),
        )
    elif kind == "siglip":
        params = config.get("siglip") or {}
        from ..settings import get_settings

        lora = params.get("lora_path") or None
        if lora and not Path(lora).is_absolute():
            lora = get_settings().resolve(Path(lora))
        head = params.get("query_head") or None
        if head and not Path(head).is_absolute():
            head = get_settings().resolve(Path(head))
        embedder = SigLIPEmbedder(
            model_id=params.get("model_id", "google/siglip2-base-patch16-384"),
            device=params.get("device", "auto"),
            batch_size=int(params.get("batch_size", 16)),
            lora_path=lora,
            max_side=int(params.get("max_side", 800)),
            threads=int(params.get("threads", 0)),
            query_head_path=head,
        )
    else:
        raise ValueError(f"Неизвестный embedder.kind: {kind}")

    log.info("Эмбеддер: %s, dim=%d", embedder.name, embedder.dim)
    return embedder


def encode_query(embedder: Embedder, views, packshot_mix: float = 0.5) -> np.ndarray:
    """Вектор фото запроса по его видам (normalize.query.QueryViews).

    Фото с полки — адаптер запросов (LoRA). Пэкшот на однотонном фоне похож
    на эталон каталога, и базовая модель узнаёт его не хуже адаптера: вектор —
    смесь обоих (packshot_mix — доля адаптера), в том же пространстве индекса.
    """
    vector = embedder.encode([views.embed], role="query")
    if getattr(views, "packshot", False) and packshot_mix < 1.0:
        base = embedder.encode([views.embed], role="catalog")
        vector = l2_normalize(packshot_mix * vector + (1.0 - packshot_mix) * base)
    return vector
