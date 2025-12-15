"""
MRX SCAN - Sistema de Classificacao de Sucata Eletronica
Scanner por camera e gestao de dataset usando OpenCLIP
"""

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
from typing import Optional, List
from pathlib import Path
from contextlib import asynccontextmanager

from embedding_engine import get_engine, OFFICIAL_CLASSES
from database import init_db, get_db, ScanHistory, SessionLocal

app = FastAPI(
    title="MRX SCAN",
    description="Sistema de classificacao de sucata eletronica por similaridade de imagem",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATASET_DIR = "dataset"
os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs("embeddings", exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/images", StaticFiles(directory=DATASET_DIR), name="images")

@app.on_event("startup")
async def startup_event():
    """Initialize the database and embedding engine on startup"""
    print("Initializing database...")
    init_db()
    print("Database initialized!")
    print("Initializing MRX SCAN embedding engine...")
    get_engine()
    print("MRX SCAN ready!")

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main scanner interface"""
    with open("templates/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

@app.post("/scan")
async def scan_image(image: UploadFile = File(...)):
    """
    Scan an image and classify it
    Returns: classification, confidence score, and top 3 matches
    """
    try:
        contents = await image.read()
        
        if not contents:
            raise HTTPException(status_code=400, detail="Empty image file")
        
        engine = get_engine()
        result = engine.classify_image(contents)
        
        return JSONResponse(content=result)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/dataset/upload")
async def upload_to_dataset(
    image: UploadFile = File(...),
    classification: str = Form(...)
):
    """
    Upload a new image to a specific classification in the dataset
    This updates the embeddings automatically
    """
    try:
        contents = await image.read()
        
        if not contents:
            raise HTTPException(status_code=400, detail="Empty image file")
        
        engine = get_engine()
        saved_path = engine.add_image_to_class(
            classification,
            contents,
            image.filename or "image.jpg"
        )
        
        return JSONResponse(content={
            "success": True,
            "message": f"Image added to {classification}",
            "path": saved_path
        })
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/dataset/upload-multiple")
async def upload_multiple_images(
    images: List[UploadFile] = File(...),
    classification: str = Form(...)
):
    """
    Upload multiple images to a specific classification
    """
    try:
        engine = get_engine()
        saved_paths = []
        
        for image in images:
            contents = await image.read()
            if contents:
                saved_path = engine.add_image_to_class(
                    classification,
                    contents,
                    image.filename or "image.jpg"
                )
                saved_paths.append(saved_path)
        
        return JSONResponse(content={
            "success": True,
            "message": f"{len(saved_paths)} images added to {classification}",
            "paths": saved_paths
        })
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/dataset/classes")
async def get_classes():
    """
    Get all available classifications with image counts
    """
    try:
        engine = get_engine()
        classes = engine.get_class_info()
        
        return JSONResponse(content={
            "classes": classes,
            "total_classes": len(classes)
        })
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/dataset/create")
async def create_classification(name: str = Form(...)):
    """
    Create a new classification folder
    """
    try:
        folder_path = os.path.join(DATASET_DIR, name)
        
        if os.path.exists(folder_path):
            raise HTTPException(status_code=400, detail=f"Classification '{name}' already exists")
        
        os.makedirs(folder_path)
        
        return JSONResponse(content={
            "success": True,
            "message": f"Classification '{name}' created successfully",
            "path": folder_path
        })
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/dataset/images/{class_name:path}")
async def get_class_images(class_name: str):
    """
    Get all images from a specific classification folder
    """
    try:
        folder_path = os.path.join(DATASET_DIR, class_name)
        
        if not os.path.exists(folder_path):
            raise HTTPException(status_code=404, detail=f"Classification '{class_name}' not found")
        
        images = []
        valid_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
        
        for filename in os.listdir(folder_path):
            if Path(filename).suffix.lower() in valid_extensions:
                images.append({
                    "filename": filename,
                    "path": f"/images/{class_name}/{filename}"
                })
        
        return JSONResponse(content={
            "class_name": class_name,
            "images": images,
            "total": len(images)
        })
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/dataset/capture-multiple")
async def capture_multiple_images(
    images: List[UploadFile] = File(...),
    classification: str = Form(...)
):
    """
    Upload multiple captured images from camera to a specific classification
    """
    try:
        import uuid
        from datetime import datetime
        
        engine = get_engine()
        saved_paths = []
        
        for image in images:
            contents = await image.read()
            if contents:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                unique_id = uuid.uuid4().hex[:8]
                filename = f"capture_{timestamp}_{unique_id}.jpg"
                
                saved_path = engine.add_image_to_class(
                    classification,
                    contents,
                    filename
                )
                saved_paths.append(saved_path)
        
        return JSONResponse(content={
            "success": True,
            "message": f"{len(saved_paths)} imagens capturadas adicionadas a {classification}",
            "paths": saved_paths
        })
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/dataset/retrain")
async def retrain_dataset():
    """
    Retrain all embeddings from the dataset images
    Use this after manually adding/removing images from dataset folders
    """
    try:
        engine = get_engine()
        engine.update_all_embeddings()
        
        return JSONResponse(content={
            "success": True,
            "message": "All embeddings updated successfully"
        })
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model": "OpenCLIP ViT-B-32", "system": "MRX SCAN"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
