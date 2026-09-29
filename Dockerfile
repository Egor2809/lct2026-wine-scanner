# Образ самодостаточен: код, каталог, превью фото, индекс и веса всех моделей
# (включая локальный RapidOCR) внутри. Сеть нужна только на сборке; Yandex OCR —
# необязательный ускоритель по ключу. Собирать: docker compose up --build (CPU).
FROM python:3.12-slim

# libGL/libglib — opencv (тянет ultralytics), curl — healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# torch с CPU-индекса: CUDA-сборка весит ~2.5 ГБ лишнего. Отдельным слоем,
# чтобы правка requirements.txt не перекачивала его заново. Версии — те же,
# что в constraints.txt: на них сняты замеры.
RUN pip install --no-cache-dir torch==2.14.0 torchvision==0.29.0 \
        --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt constraints.txt ./
RUN pip install --no-cache-dir -r requirements.txt -c constraints.txt

ENV PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/.cache/huggingface \
    YOLO_CONFIG_DIR=/tmp/ultralytics

# Веса SigLIP 2 скачиваются при сборке, а не на старте и не на первом запросе.
COPY config/ config/
RUN python -c "import yaml; from huggingface_hub import snapshot_download; \
cfg = yaml.safe_load(open('config/config.yaml')); \
snapshot_download(cfg['embedder']['siglip']['model_id'])"

COPY src/ src/
# Модели локального OCR (RapidOCR PP-OCRv5 eslav) — при сборке, в рантайме сети не нужно.
RUN python -c "from wine_scanner.features.text.rapid_ocr import RapidOCRExtractor; RapidOCRExtractor()"
COPY scripts/ scripts/
COPY frontend/ frontend/
# Рантайм-данные: каталог, индекс, YOLO-детектор этикетки, LoRA (~8,6 МБ) и SIFT (63 МБ).
COPY data/catalog.csv data/catalog.csv
# Превью фото для карточек (~23 МБ, scripts/make_thumbnails.py).
COPY data/catalog_thumbs/ data/catalog_thumbs/
COPY artifacts/index.npz artifacts/index.npz
# SIFT эталонов для геометрической проверки пэкшотов: без него признаки
# считаются из data/catalog_photos, а в чистом клоне фото нет — проверка
# молча выключается (скрытая синтетика 92,7 % -> 64,7 %).
COPY artifacts/geometric_sift.npz artifacts/geometric_sift.npz
COPY artifacts/models/ artifacts/models/

# После сборки — ни одного обращения в сеть за весами.
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    YOLO_OFFLINE=1 \
    CATALOG_PATH=data/catalog.csv \
    CATALOG_IMAGES_DIR=data/catalog_photos \
    CATALOG_THUMBS_DIR=data/catalog_thumbs \
    ARTIFACTS_DIR=artifacts

EXPOSE 8080

# Модель и индекс грузятся в lifespan (~20–40 с на CPU). Пока прогрев не
# закончен, /health не отвечает и контейнер в состоянии starting.
HEALTHCHECK --interval=10s --timeout=5s --start-period=120s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8080/health || exit 1

CMD ["uvicorn", "wine_scanner.api.app:app", "--host", "0.0.0.0", "--port", "8080"]
