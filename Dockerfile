# Base image
FROM nvidia/cuda:11.8.0-base-ubuntu20.04

# Force Python à afficher les logs immédiatement
ENV PYTHONUNBUFFERED=1

# installation libs système
RUN apt-get update && apt-get install -y \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# installation dépendance
COPY docker-requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r docker-requirements.txt && \
    pip install --no-cache-dir torch==2.7.1+cu118 torchvision==0.22.1+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

# création des dossiers nécessaires avant de copier le code
RUN mkdir -p /app/datasets /app/data

# copie les éléments nécessaires
COPY ./src /app/src
COPY ./utils /app/utils
COPY ./docker-template/api.py /app
COPY ./models /app/models

# expose le port 5000 (pour mlflow) et 8000 (pour l'api)
EXPOSE 5000
EXPOSE 8000

# Run the FastAPI application
CMD ["/bin/bash", "-c", "python3 api.py 2>&1 | tee -a /app/api_log.log"]
