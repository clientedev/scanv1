#!/bin/bash
set -e

echo "=== MRX SCAN Starting ==="
echo "Database URL configured: $([ -n "$DATABASE_URL" ] && echo 'YES' || echo 'NO')"

echo "=== Creating database tables ==="
python << 'EOF'
import os
import time
from sqlalchemy import create_engine, text
from sqlalchemy.pool import QueuePool

def get_database_url():
    url = os.getenv("DATABASE_URL", "")
    if not url:
        print("ERROR: DATABASE_URL not set!")
        return None
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql://", 1)
    return url

db_url = get_database_url()
if db_url:
    print(f"Connecting to PostgreSQL...")
    for attempt in range(5):
        try:
            engine = create_engine(db_url, poolclass=QueuePool, pool_pre_ping=True)
            with engine.connect() as conn:
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS classifications (
                        id SERIAL PRIMARY KEY,
                        name VARCHAR(255) UNIQUE NOT NULL,
                        description TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """))
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS dataset_images (
                        id SERIAL PRIMARY KEY,
                        filename VARCHAR(255) NOT NULL,
                        classification VARCHAR(255) NOT NULL,
                        image_data BYTEA,
                        image_path VARCHAR(500),
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """))
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS embeddings (
                        id SERIAL PRIMARY KEY,
                        classification VARCHAR(255) NOT NULL,
                        embedding_data BYTEA NOT NULL,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """))
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS scan_history (
                        id SERIAL PRIMARY KEY,
                        predicted_class VARCHAR(255) NOT NULL,
                        confidence FLOAT NOT NULL,
                        scanned_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """))
                conn.commit()
            print("SUCCESS: All tables created!")
            break
        except Exception as e:
            print(f"Attempt {attempt+1}/5 failed: {e}")
            if attempt < 4:
                time.sleep(3)
            else:
                print("FAILED: Could not create tables after 5 attempts")
else:
    print("WARNING: No DATABASE_URL, using SQLite")
EOF

echo "=== Starting uvicorn server ==="
exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-5000}