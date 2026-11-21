FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    YOLO_CONFIG_DIR=/tmp/ultralytics_cfg

# OpenCV çalışma zamanı için gerekli minimum sistem kütüphaneleri
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# En az yetki ilkesi (Least Privilege): root olmayan özel kullanıcı
RUN groupadd --gid 10001 analyzer \
    && useradd --uid 10001 --gid 10001 --create-home --shell /usr/sbin/nologin analyzer

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/requirements.txt

COPY --chown=analyzer:analyzer api /app/api
COPY --chown=analyzer:analyzer config /app/config
COPY --chown=analyzer:analyzer modules /app/modules
COPY --chown=analyzer:analyzer utils /app/utils
COPY --chown=analyzer:analyzer main.py /app/main.py
COPY --chown=analyzer:analyzer models /app/models

RUN mkdir -p /app/data/jobs /app/data/exports && chown -R analyzer:analyzer /app/data

USER analyzer

EXPOSE 8000

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
