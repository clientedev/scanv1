FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p dataset embeddings

ENV PORT=5000
ENV PYTHONUNBUFFERED=1

EXPOSE 5000

CMD ["sh", "-c", "echo 'Initializing database...' && python -c 'from database import init_db; result = init_db(); print(\"Tables created successfully!\" if result else \"Warning: Database initialization had issues\")' && echo 'Starting server...' && uvicorn main:app --host 0.0.0.0 --port $PORT"]
