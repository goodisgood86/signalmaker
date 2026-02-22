FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

COPY app /app/app
COPY static /app/static
COPY artifacts/models /app/artifacts/models
COPY scripts /app/scripts
RUN if ls /app/scripts/*.sh >/dev/null 2>&1; then chmod +x /app/scripts/*.sh; fi

RUN mkdir -p /app/backups/pass_check

EXPOSE 8000

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
