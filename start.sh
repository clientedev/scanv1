#!/bin/bash
set -e

echo "=========================================="
echo "=== MRX SCAN Starting ==="
echo "=========================================="

if [ -z "$DATABASE_URL" ]; then
    echo "WARNING: DATABASE_URL not set, using SQLite"
else
    echo "DATABASE_URL is configured"
    echo "=== Running create_tables.py ==="
    python create_tables.py || echo "Table creation script failed, continuing..."
fi

echo "=== Starting uvicorn server ==="
exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-5000}