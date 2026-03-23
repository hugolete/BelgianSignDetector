# Base image
FROM nvidia/cuda:12.5.1-runtime-ubuntu24.04

# Force Python à afficher les logs immédiatement
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Brussels

# installation libs système
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Créer un lien pour que 'python' pointe vers 'python3'
RUN ln -s /usr/bin/python3 /usr/bin/python

# Set working directory
WORKDIR /app

# installation dépendance
COPY docker-requirements.txt .

# outil perso
COPY factory_mlops-0.0.1-py3-none-any.whl .
RUN pip install --no-cache-dir ./factory_mlops-0.0.1-py3-none-any.whl --break-system-packages

RUN pip install --no-cache-dir torch==2.6.0+cu124 torchvision==0.21.0+cu124 \
    --index-url https://download.pytorch.org/whl/cu124 --break-system-packages && \
    pip install --no-cache-dir -r docker-requirements.txt --break-system-packages

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
