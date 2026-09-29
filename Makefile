# Одна команда на шаг. Подробности и цифры — README.md, устройство — ARCHITECTURE.md.
PY ?= .venv/bin/python
URL ?= http://127.0.0.1:8080/scan
COMPOSE_PG = docker compose -f docker-compose.yml -f docker-compose.pgvector.yml

.PHONY: setup lint test catalog thumbs index run docker docker-pg eval synth train-yolo train-lora

setup:        ## окружение с точными версиями замеров
	python3.12 -m venv .venv
	$(PY) -m pip install -r requirements.txt -c constraints.txt

lint:
	$(PY) -m ruff check src tests scripts

test:         ## контракт ответа, границы слоёв, каталог, поиск (PGVECTOR_DSN — и pgvector)
	$(PY) -m pytest

catalog:      ## дамп заказчика -> data/catalog.csv (нужен архив кейса)
	$(PY) scripts/prepare_catalog.py

thumbs:       ## превью фото для карточек -> data/catalog_thumbs
	$(PY) scripts/make_thumbnails.py

index:        ## эмбеддинги каталога -> artifacts/index.npz
	$(PY) scripts/build_index.py

run:          ## сервис и интерфейс на :8080
	PYTHONPATH=src $(PY) -m uvicorn wine_scanner.api.app:app --host 0.0.0.0 --port 8080

docker:       ## то же в Docker, как у проверяющих
	docker compose up -d --build

docker-pg:    ## стек заказчика: индекс в PostgreSQL + pgvector
	$(COMPOSE_PG) up -d --build

eval:         ## полевые фото кейса (data/field, в репозиторий не входят)
	$(PY) scripts/evaluate.py --labels data/field/labels.csv --url $(URL)

synth:        ## синтетика из фото каталога, как в CI
	$(PY) scripts/make_synthetic.py --n 60 --out /tmp/wine-synth
	PYTHONPATH=src $(PY) scripts/evaluate.py --labels /tmp/wine-synth/labels.csv --url $(URL) --min-top1 0.75

train-yolo:   ## детектор этикетки по разметке training/label_boxes (нужны data/field)
	$(PY) -m pip install -q -r requirements-train.txt -c constraints.txt
	$(PY) scripts/train_label_detector.py --out artifacts/models/label_yolov8n.pt

train-lora:   ## LoRA запросов по сплиту training/lora_train_photos.txt (нужны data/field)
	$(PY) -m pip install -q -r requirements-train.txt -c constraints.txt
	$(PY) scripts/train_lora.py --out artifacts/models/siglip_lora_v2.pt
