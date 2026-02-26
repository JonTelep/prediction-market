# Stage 1: Build dependencies
FROM python:3.12-slim AS builder

RUN pip install --no-cache-dir uv

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src/ src/
COPY config/ config/
RUN uv sync --frozen --no-dev

# Stage 2: Runtime
FROM python:3.12-slim

WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/src /app/src
COPY --from=builder /app/config /app/config

ENV PATH="/app/.venv/bin:$PATH"
ENV DATABASE_PATH=/app/data/prediction_market.db
ENV REPORTING_OUTPUT_DIR=/app/data/reports

VOLUME ["/app/data"]

ENTRYPOINT ["prediction-market"]
CMD ["scan"]
