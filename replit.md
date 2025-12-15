# Electronic Scrap Classification Scanner

## Overview
A camera-based classification system for electronic scrap materials using OpenCLIP image similarity. The system captures images via webcam/camera and classifies materials according to official stock classifications.

## Project Architecture

### Backend (Python/FastAPI)
- `main.py` - FastAPI server with REST endpoints
- `embedding_engine.py` - OpenCLIP-based image embedding and classification engine

### Frontend
- `templates/index.html` - Camera scanner interface with classification display

### Data Storage
- `dataset/` - Contains subdirectories for each classification with training images
- `embeddings/` - Cached numpy embeddings for fast classification

## Official Classifications
1. HIGH GRADE
2. MIDION GRADE
3. MIDION GRADE 1 - MG 1
4. MIDION GRADE 2 - MG 2
5. LOW GRADE
6. LOW
7. DIVERSOS
8. GARIMPOS
9. MOAGEM
10. HD

## API Endpoints
- `POST /scan` - Classify an uploaded image
- `POST /dataset/upload` - Add new training image to a classification
- `GET /dataset/classes` - List all classifications with image counts
- `POST /dataset/retrain` - Regenerate all embeddings

## How to Use
1. Start the server
2. Upload training images to each classification category
3. Use the camera to scan materials
4. System returns top 3 classifications with confidence scores

## Technical Stack
- FastAPI for REST API
- OpenCLIP (ViT-B-32) for image embeddings
- Cosine similarity for classification
- Vanilla JavaScript with getUserMedia for camera capture
