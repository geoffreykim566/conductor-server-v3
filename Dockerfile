FROM python:3.12-slim

WORKDIR /srv

# DEV=1 (set by docker-compose.yml, never on Railway) adds pytest.
ARG DEV=0

COPY pyproject.toml ./
COPY app ./app
COPY seed ./seed
RUN pip install --no-cache-dir . && \
    if [ "$DEV" = "1" ]; then pip install --no-cache-dir ".[dev]"; fi

EXPOSE 8000
CMD uvicorn app.api.main:app --host 0.0.0.0 --port ${PORT:-8000}
