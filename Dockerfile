FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p dataset embeddings

EXPOSE 5000

CMD python -c "from database import init_db; init_db()" && uvicorn main:app --host 0.0.0.0 --port ${PORT:-5000}
