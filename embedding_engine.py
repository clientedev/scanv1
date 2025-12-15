"""
Embedding Engine Module for Electronic Scrap Classification
Uses OpenCLIP for image embedding generation and cosine similarity comparison
"""

import os
import numpy as np
from PIL import Image
import open_clip
import torch
from typing import List, Dict, Tuple, Optional
import json
from pathlib import Path

DATASET_DIR = "dataset"
EMBEDDINGS_DIR = "embeddings"

OFFICIAL_CLASSES = [
    "HIGH GRADE",
    "MIDION GRADE",
    "MIDION GRADE 1 - MG 1",
    "MIDION GRADE 2 - MG 2",
    "LOW GRADE",
    "LOW",
    "DIVERSOS",
    "GARIMPOS",
    "MOAGEM",
    "HD"
]

class EmbeddingEngine:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.preprocess = None
        self.embeddings_cache: Dict[str, np.ndarray] = {}
        self.image_paths_cache: Dict[str, List[str]] = {}
        self._load_model()
        self._load_embeddings()
    
    def _load_model(self):
        """Load the OpenCLIP model for image embedding generation"""
        print("Loading OpenCLIP model...")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            'ViT-B-32',
            pretrained='laion2b_s34b_b79k'
        )
        self.model = self.model.to(self.device)
        self.model.eval()
        print(f"Model loaded on {self.device}")
    
    def _get_embedding_path(self, class_name: str) -> str:
        """Get the path for storing embeddings of a class"""
        safe_name = class_name.replace(" ", "_").replace("-", "_")
        return os.path.join(EMBEDDINGS_DIR, f"{safe_name}.npy")
    
    def _get_metadata_path(self, class_name: str) -> str:
        """Get the path for storing metadata of a class"""
        safe_name = class_name.replace(" ", "_").replace("-", "_")
        return os.path.join(EMBEDDINGS_DIR, f"{safe_name}_meta.json")
    
    def generate_embedding(self, image: Image.Image) -> np.ndarray:
        """Generate embedding vector for a single image"""
        with torch.no_grad():
            image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
            embedding = self.model.encode_image(image_tensor)
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)
            return embedding.cpu().numpy().flatten()
    
    def generate_embedding_from_bytes(self, image_bytes: bytes) -> np.ndarray:
        """Generate embedding from image bytes"""
        from io import BytesIO
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        return self.generate_embedding(image)
    
    def _load_embeddings(self):
        """Load all cached embeddings from disk"""
        print("Loading cached embeddings...")
        for class_name in OFFICIAL_CLASSES:
            embedding_path = self._get_embedding_path(class_name)
            metadata_path = self._get_metadata_path(class_name)
            
            if os.path.exists(embedding_path) and os.path.exists(metadata_path):
                self.embeddings_cache[class_name] = np.load(embedding_path)
                with open(metadata_path, 'r') as f:
                    self.image_paths_cache[class_name] = json.load(f)
                print(f"  Loaded {len(self.image_paths_cache[class_name])} embeddings for {class_name}")
            else:
                self.embeddings_cache[class_name] = np.array([])
                self.image_paths_cache[class_name] = []
        print("Embeddings loaded")
    
    def update_class_embeddings(self, class_name: str):
        """Update embeddings for a specific class by processing all images in its folder"""
        if class_name not in OFFICIAL_CLASSES:
            raise ValueError(f"Unknown class: {class_name}")
        
        class_dir = os.path.join(DATASET_DIR, class_name)
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)
            return
        
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        image_files = [
            f for f in os.listdir(class_dir)
            if os.path.splitext(f)[1].lower() in image_extensions
        ]
        
        if not image_files:
            self.embeddings_cache[class_name] = np.array([])
            self.image_paths_cache[class_name] = []
            return
        
        embeddings = []
        paths = []
        
        for img_file in image_files:
            img_path = os.path.join(class_dir, img_file)
            try:
                image = Image.open(img_path).convert("RGB")
                embedding = self.generate_embedding(image)
                embeddings.append(embedding)
                paths.append(img_path)
                print(f"  Processed: {img_file}")
            except Exception as e:
                print(f"  Error processing {img_file}: {e}")
        
        if embeddings:
            embeddings_array = np.array(embeddings)
            self.embeddings_cache[class_name] = embeddings_array
            self.image_paths_cache[class_name] = paths
            
            np.save(self._get_embedding_path(class_name), embeddings_array)
            with open(self._get_metadata_path(class_name), 'w') as f:
                json.dump(paths, f)
    
    def update_all_embeddings(self):
        """Update embeddings for all classes"""
        print("Updating all embeddings...")
        for class_name in OFFICIAL_CLASSES:
            print(f"Processing class: {class_name}")
            self.update_class_embeddings(class_name)
        print("All embeddings updated")
    
    def add_image_to_class(self, class_name: str, image_bytes: bytes, filename: str) -> str:
        """Add a new image to a class and update embeddings"""
        import uuid
        import re
        
        if class_name not in OFFICIAL_CLASSES:
            raise ValueError(f"Unknown class: {class_name}")
        
        class_dir = os.path.join(DATASET_DIR, class_name)
        os.makedirs(class_dir, exist_ok=True)
        
        safe_filename = os.path.basename(filename)
        safe_filename = re.sub(r'[^\w\-.]', '_', safe_filename)
        
        base_name = os.path.splitext(safe_filename)[0] or 'image'
        ext = os.path.splitext(safe_filename)[1] or '.jpg'
        
        if ext.lower() not in {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}:
            ext = '.jpg'
        
        unique_id = uuid.uuid4().hex[:8]
        new_filename = f"{base_name}_{unique_id}{ext}"
        new_path = os.path.join(class_dir, new_filename)
        
        from io import BytesIO
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        image.save(new_path)
        
        embedding = self.generate_embedding(image)
        
        if len(self.embeddings_cache[class_name]) == 0:
            self.embeddings_cache[class_name] = np.array([embedding])
        else:
            self.embeddings_cache[class_name] = np.vstack([
                self.embeddings_cache[class_name],
                embedding
            ])
        
        self.image_paths_cache[class_name].append(new_path)
        
        np.save(self._get_embedding_path(class_name), self.embeddings_cache[class_name])
        with open(self._get_metadata_path(class_name), 'w') as f:
            json.dump(self.image_paths_cache[class_name], f)
        
        return new_path
    
    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    
    def classify_image(self, image_bytes: bytes) -> Dict:
        """Classify an image and return results with top 3 classifications"""
        query_embedding = self.generate_embedding_from_bytes(image_bytes)
        
        class_scores = {}
        
        for class_name in OFFICIAL_CLASSES:
            class_embeddings = self.embeddings_cache.get(class_name, np.array([]))
            
            if len(class_embeddings) == 0:
                class_scores[class_name] = {
                    "avg_similarity": 0.0,
                    "max_similarity": 0.0,
                    "num_samples": 0
                }
                continue
            
            similarities = []
            for emb in class_embeddings:
                sim = self.cosine_similarity(query_embedding, emb)
                similarities.append(sim)
            
            class_scores[class_name] = {
                "avg_similarity": float(np.mean(similarities)),
                "max_similarity": float(np.max(similarities)),
                "num_samples": len(similarities)
            }
        
        sorted_classes = sorted(
            class_scores.items(),
            key=lambda x: x[1]["avg_similarity"],
            reverse=True
        )
        
        top_class = sorted_classes[0][0] if sorted_classes else None
        top_score = sorted_classes[0][1]["avg_similarity"] if sorted_classes else 0.0
        
        top_3 = [
            {
                "classification": name,
                "similarity": round(scores["avg_similarity"] * 100, 2),
                "max_similarity": round(scores["max_similarity"] * 100, 2),
                "samples_count": scores["num_samples"]
            }
            for name, scores in sorted_classes[:3]
        ]
        
        has_samples = any(
            len(self.embeddings_cache.get(c, [])) > 0
            for c in OFFICIAL_CLASSES
        )
        
        return {
            "classification": top_class if has_samples else None,
            "confidence": round(top_score * 100, 2) if has_samples else 0.0,
            "top_3": top_3,
            "has_dataset": has_samples,
            "message": "Classification successful" if has_samples else "No images in dataset. Please upload training images first."
        }
    
    def get_class_info(self) -> List[Dict]:
        """Get information about all classes and their image counts"""
        result = []
        for class_name in OFFICIAL_CLASSES:
            class_dir = os.path.join(DATASET_DIR, class_name)
            image_count = 0
            
            if os.path.exists(class_dir):
                image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
                image_count = len([
                    f for f in os.listdir(class_dir)
                    if os.path.splitext(f)[1].lower() in image_extensions
                ])
            
            result.append({
                "name": class_name,
                "image_count": image_count,
                "has_embeddings": len(self.embeddings_cache.get(class_name, [])) > 0
            })
        
        return result

engine = None

def get_engine() -> EmbeddingEngine:
    """Get or create the embedding engine singleton"""
    global engine
    if engine is None:
        engine = EmbeddingEngine()
    return engine
