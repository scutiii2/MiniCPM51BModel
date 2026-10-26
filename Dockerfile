# syntax=docker/dockerfile:1
FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 MINICPM_HOST=0.0.0.0
WORKDIR /app

COPY pyproject.toml ./
COPY src ./src
RUN pip install .

EXPOSE 9130
CMD ["python", "-m", "minicpm_service.app"]
