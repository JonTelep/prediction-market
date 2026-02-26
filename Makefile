.PHONY: build run monitor scan backfill test lint fmt clean shell

IMAGE := prediction-market
DAYS  ?= 30

build:
	podman build -t $(IMAGE) -f Containerfile .

run:
	podman run --rm --env-file .env -v ./data:/app/data:Z $(IMAGE) $(CMD)

monitor:
	podman run --rm --env-file .env -v ./data:/app/data:Z $(IMAGE) monitor

scan:
	podman run --rm --env-file .env -v ./data:/app/data:Z $(IMAGE) scan

backfill:
	podman run --rm --env-file .env -v ./data:/app/data:Z $(IMAGE) backfill --days $(DAYS)

test:
	uv run pytest tests/ -v

lint:
	uv run ruff check src/ tests/

fmt:
	uv run ruff format src/ tests/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name '*.egg-info' -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true

shell:
	podman run --rm -it --env-file .env -v ./data:/app/data:Z $(IMAGE) /bin/bash
