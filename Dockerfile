FROM python:3.12-slim

WORKDIR /srv

COPY pyproject.toml ./
COPY app ./app
COPY seed ./seed
COPY scenarios ./scenarios
RUN pip install --no-cache-dir .

CMD ["python", "-m", "app.run_scenarios"]
