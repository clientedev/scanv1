"""
Electronic Scrap Classification Scanner - FastAPI Backend
A camera-based classification system using OpenCLIP image similarity
"""

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
from typing import Optional

from embedding_engine import get_engine, OFFICIAL_CLASSES

app = FastAPI(
    title="Electronic Scrap Scanner",
    description="Camera-based classification system for electronic scrap using image similarity",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/images", StaticFiles(directory="dataset"), name="images")

@app.on_event("startup")
async def startup_event():
    """Initialize the embedding engine on startup"""
    print("Initializing embedding engine...")
    get_engine()
    print("Server ready!")

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
        if classification not in OFFICIAL_CLASSES:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid classification. Must be one of: {OFFICIAL_CLASSES}"
            )
        
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
    return {"status": "healthy", "model": "OpenCLIP ViT-B-32"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
