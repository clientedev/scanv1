FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

COPY . .

RUN mkdir -p dataset embeddings

ENV PORT=5000
ENV PYTHONUNBUFFERED=1

EXPOSE 5000

ENV SKIP_DB_INIT=1
CMD python create_tables.py && uvicorn main:app --host 0.0.0.0 --port ${PORT}
