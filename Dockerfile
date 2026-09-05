FROM python:3.12-slim

WORKDIR /srv

COPY pyproject.toml ./
COPY app ./app
COPY seed ./seed
COPY scenarios ./scenarios
RUN pip install --no-cache-dir .

EXPOSE 8000
CMD uvicorn app.api:app --host 0.0.0.0 --port ${PORT:-8000}
