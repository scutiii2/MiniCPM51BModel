# syntax=docker/dockerfile:1
FROM python:3.12-slim
ENV MINICPM_WEB_DIR=/app/web MINICPM_TOKENS_FILE=/data/tokens.json PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 MINICPM_HOST=0.0.0.0
WORKDIR /app

COPY pyproject.toml ./
COPY src ./src
COPY web ./web
RUN pip install .

EXPOSE 9130
CMD ["python", "src/app.py"]
