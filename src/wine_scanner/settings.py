"""Настройки процесса: пути и ключи — из ENV, веса и пороги — из YAML.

Разделение неслучайно: ENV меняется от машины к машине и содержит секреты,
YAML — предмет экспериментов и должен лежать в репозитории под ревью.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    catalog_path: Path = Path("data/catalog.csv")
    catalog_images_dir: Path = Path("data/catalog_photos")
    # Превью для карточек UI (scripts/make_thumbnails.py), лежат в репозитории.
    catalog_thumbs_dir: Path = Path("data/catalog_thumbs")
    artifacts_dir: Path = Path("artifacts")
    config_path: Path = Path("config/config.yaml")
    frontend_dir: Path = Path("frontend")
    # Перекрывает index.kind из config.yaml: переключение на стек заказчика
    # (docker-compose.pgvector.yml) без правки YAML.
    index_kind: str = ""
    # Только для index.kind: pgvector (search/pgvector_index.py).
    pgvector_dsn: str = ""
    pgvector_table: str = "wine_vectors"

    host: str = "0.0.0.0"
    port: int = 8080
    log_level: str = "info"
    # Через запятую. Сервис отдаёт фронт сам, CORS нужен только встраиванию в портал.
    cors_origins: str = "*"
    # Предел тела /scan. Фото с телефона на 50 МБ — валидный запрос; предел
    # только от явного мусора, который иначе целиком ляжет в память.
    max_upload_mb: int = 100
    # Сколько /scan одновременно идут в модель. На CPU torch и так занимает
    # все ядра: без предела параллельные запросы делят их и тормозят все сразу.
    # 2, а не 1 — пока один ждёт облачный OCR, второй успевает посчитать эмбеддинг.
    # 0 — без предела.
    scan_concurrency: int = 2

    # Облачный OCR (Yandex Vision). Пусто -> локальный OCR, сервис работает офлайн.
    yandex_api_key: str = ""
    yandex_folder_id: str = ""
    # VLM-судья (OpenAI-совместимый API, по умолчанию DeepSeek). Пусто -> выключен.
    vlm_api_key: str = ""
    vlm_base_url: str = ""
    vlm_model: str = ""

    def resolve(self, value: Path) -> Path:
        """Относительные пути считаются от корня репозитория, не от cwd."""
        return value if value.is_absolute() else ROOT / value

    @property
    def index_path(self) -> Path:
        return self.resolve(self.artifacts_dir) / "index.npz"


class Config(dict):
    """YAML-конфиг с доступом по точечному пути: cfg.get_path("fusion.weights.visual")."""

    def get_path(self, dotted: str, default: Any = None) -> Any:
        node: Any = self
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


@lru_cache(maxsize=1)
def get_config() -> Config:
    path = get_settings().resolve(get_settings().config_path)
    with path.open(encoding="utf-8") as fh:
        return Config(yaml.safe_load(fh) or {})
