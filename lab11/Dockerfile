# Imagen única para el pipeline, MLflow y el tablero. No contiene datos institucionales:
# el Excel se monta en /datos (ver docker-compose.yml y .dockerignore).
FROM python:3.12.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/* \
    && git config --system --add safe.directory /app \
    && git config --system core.autocrlf true

WORKDIR /app
COPY lab11/requirements.txt lab11/requirements.txt
RUN pip install --no-cache-dir -r lab11/requirements.txt

COPY lab11/ lab11/
COPY Notebooks/resultados/ Notebooks/resultados/
ENV PYTHONUNBUFFERED=1
