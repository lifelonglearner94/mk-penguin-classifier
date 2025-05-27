FROM python:3.10-slim

WORKDIR /app

RUN pip install poetry

COPY pyproject.toml poetry.lock ./
COPY README.md ./

RUN poetry config virtualenvs.create false && poetry install --no-interaction --no-ansi --no-root

COPY run_training.py ./

# For testing the append mode
COPY data/ ./data/

COPY training_logic/ ./training_logic/
COPY .env .env

ENV PYTHONPATH "${PYTHONPATH}:/app"
