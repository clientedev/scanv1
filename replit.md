# MRX SCAN - Sistema de Classificacao de Sucata Eletronica

## Overview
MRX SCAN is an AI-powered image classification system for electronic scrap materials. It uses OpenCLIP (ViT-B-32) for image embeddings and cosine similarity to classify materials based on reference images in the dataset.

## Features
- **Camera Scanner**: Real-time camera capture and classification
- **Dataset Management**: Create folders, upload multiple images, view dataset
- **AI Classification**: Uses OpenCLIP for image similarity comparison
- **Visual Identity**: Green/white admin panel design (similar to Google Drive)

## Project Structure
```
/
├── main.py                 # FastAPI backend with all endpoints
├── embedding_engine.py     # OpenCLIP model and classification logic
├── templates/
│   └── index.html         # Frontend SPA with Scanner and Dataset views
├── dataset/               # Classification folders with images
├── embeddings/            # Cached embeddings (.npy files)
└── static/                # Static assets
```

## API Endpoints
- `GET /` - Main interface
- `POST /scan` - Classify an image
- `GET /dataset/classes` - List all classification folders
- `POST /dataset/create` - Create new folder
- `GET /dataset/images/{class_name}` - Get images from folder
- `POST /dataset/upload-multiple` - Upload multiple images
- `POST /dataset/capture-multiple` - Upload multiple captured camera images
- `POST /dataset/retrain` - Retrain embeddings
- `GET /health` - Health check

## Tech Stack
- **Backend**: FastAPI, Python 3.11
- **AI Model**: OpenCLIP ViT-B-32 (laion2b_s34b_b79k)
- **Frontend**: Vanilla HTML/CSS/JS with Font Awesome icons

## Running the App
The app runs on port 5000 with `python main.py`

## Recent Changes
- 2025-12-15: Added visual identity with MRX DO BRASIL logo in header
- 2025-12-15: Implemented 60% minimum similarity threshold (SIMILARITY_THRESHOLD = 0.60)
- 2025-12-15: Added "no_match" status when similarity < 60% with Portuguese warning message
- 2025-12-15: Added multiple capture mode (POST /dataset/capture-multiple) for batch camera uploads
- 2025-12-15: Renamed to MRX SCAN, new green/white visual identity
- 2025-12-15: Added dataset management interface (folder creation, multi-upload)
- 2025-12-15: Dynamic folder-based classifications (any folder in /dataset is a class)

## Configuration
- SIMILARITY_THRESHOLD: 0.60 (60%) - configurable in embedding_engine.py
