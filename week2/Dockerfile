FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime

ARG TRANSFORMERS_VERSION=4.53.2

RUN python -m pip install --no-cache-dir \
    "transformers==${TRANSFORMERS_VERSION}"

WORKDIR /app
COPY inference.py /app/inference.py
COPY service.py /app/service.py

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    TOKENIZERS_PARALLELISM=false \
    PYTHONUNBUFFERED=1

ENTRYPOINT ["python", "/app/service.py"]
