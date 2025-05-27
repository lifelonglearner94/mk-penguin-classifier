FROM python:3.10-slim

RUN pip install poetry supervisor

WORKDIR /app

COPY pyproject.toml poetry.lock ./
COPY README.md ./

COPY ui_app/ ui_app/
COPY tests/ tests/
COPY .env .env

RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi --no-root

ENV PYTHONPATH=/app

# Expose both ports for UI and API
EXPOSE 8050 8080

# Create supervisor config
RUN mkdir -p /etc/supervisor/conf.d

# Copy supervisor configuration
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

CMD ["supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]
