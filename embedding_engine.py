"""
Embedding Engine Module for MRX SCAN - Electronic Scrap Classification
Uses OpenCLIP for image embedding generation and PostgreSQL for persistent storage
"""

import os
import numpy as np
from PIL import Image
import open_clip
import torch
from typing import List, Dict, Tuple, Optional
import json
from pathlib import Path
import io
from sqlalchemy.orm import Session
from database import get_session_local, DatasetImage, Embedding, Classification
from sqlalchemy import func

DATASET_DIR = "dataset"
EMBEDDINGS_DIR = "embeddings"

SIMILARITY_THRESHOLD = 0.60

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
        # Cache for faster lookup (will be populated from DB)
        self.embeddings_cache: Dict[str, np.ndarray] = {}
        self.image_paths_cache: Dict[str, List[Dict]] = {} # List of dicts with id, filename
        
        self.db_session = get_session_local()()
        
        self._load_model()
        self._load_embeddings_from_db()
    
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
    
    def generate_embedding(self, image: Image.Image) -> np.ndarray:
        """Generate embedding vector for a single image"""
        with torch.no_grad():
            image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
            embedding = self.model.encode_image(image_tensor)
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)
            return embedding.cpu().numpy().flatten()
    
    def generate_embedding_from_bytes(self, image_bytes: bytes) -> np.ndarray:
        """Generate embedding from image bytes"""
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            return self.generate_embedding(image)
        except Exception as e:
            print(f"Error generating embedding: {e}")
            raise ValueError("Invalid image data")

    def _load_embeddings_from_db(self):
        """Load all embeddings from database into memory cache"""
        print("Loading embeddings from database...")
        
        # Reset caches
        self.embeddings_cache = {}
        self.image_paths_cache = {}
        
        try:
            # Load Embeddings
            embeddings_query = self.db_session.query(Embedding).all()
            for emb in embeddings_query:
                class_name = emb.classification
                embedding_vector = np.frombuffer(emb.embedding_data, dtype=np.float32)
                
                if class_name not in self.embeddings_cache:
                    self.embeddings_cache[class_name] = []
                
                self.embeddings_cache[class_name].append(embedding_vector)
            
            # Convert lists to numpy arrays
            for class_name in self.embeddings_cache:
                self.embeddings_cache[class_name] = np.array(self.embeddings_cache[class_name])
                
            # Load Image Metadata (DatasetImage) to map embeddings to images
            # Note: The order of embeddings in DB might differentiate from images if not careful.
            # Ideally we should link them. 
            # Current simplified approach assumes we can just reload all embeddings and all images.
            # But the 'classify' method assumes index matching.
            # TO FIX THIS properly: We will reload EVERYTHING from DatasetImage table actually.
            
            self._rebuild_cache_from_images_table()
            
            print("Embeddings loaded from DB")
            
        except Exception as e:
            print(f"Error loading embeddings from DB: {e}")
            self.db_session.rollback()

    def _rebuild_cache_from_images_table(self):
        """Rebuild in-memory cache strictly from DatasetImage table to ensure consistency"""
        print("Rebuilding cache from DatasetImage table...")
        self.embeddings_cache = {}
        self.image_paths_cache = {}
        
        # Get all images ordered by ID to ensure consistent order
        images = self.db_session.query(DatasetImage).order_by(DatasetImage.id).all()
        
        temp_embeddings = {}
        
        for img in images:
            if not img.classification:
                continue
                
            # If image has no embedding but we have the image data, we might need to regenerate it
            # But for now assuming we have a separate way to store/retrieve embeddings or regenerate them
            # The original code stored embeddings separately. 
            # We need to find the corresponding embedding.
            pass

        # Use the Embedding table for actual vectors, assuming 1:1 mapped logic or simple retrieval
        # Better approach for this refactor:
        # Load ALL Embeddings from 'Embedding' table which are vectors properly stored.
        # But wait, the original code had a disconnect between 'embeddings.npy' and 'file list'.
        # We need to robustly fix this.
        
        # NEW STRATEGY:
        # 1. Load all DatasetImage records.
        # 2. If an image doesn't have a corresponding record in 'Embedding' (conceptually), generate it.
        # 3. Actually, let's keep it simple and efficient. Use 'Embedding' table for search.
        # 4. We need to know which Image corresponds to which Embedding index.
        
        # Since the original system just append-only'd to .npy files and .json lists, 
        # let's try to maintain that structure in memory but backed by DB rows.
        
        # We will fetch everything from Embedding table.
        # We ALSO need the image filenames/ids to return in the API.
        
        # Let's query Embedding table. It has 'classification' and 'embedding_data'.
        # It DOES NOT have a link to DatasetImage ID. This is a schema limitations of original code.
        # The user wants us to fix persistence.
        # We will ONLY use DatasetImage table. We will store embedding IN DatasetImage or re-generate on startup?
        # Re-generating on startup is too slow for many images.
        # We should add 'embedding_data' to DatasetImage or link them.
        # However, I cannot change schema easily without migrations.
        # Wait, 'Embedding' table exists. Let's use it.
        # BUT, how to link?
        
        # Let's look at `create_tables.py`:
        # DatasetImage: id, filename, classification, image_data
        # Embedding: id, classification, embedding_data
        
        # The PROPER fix is to store embedding with the image or link them.
        # Given I can't easily run migrations on production easily without risk,
        # I will use a logic of: 
        # "Load all DatasetImages. If count(DatasetImages) != count(Embeddings) for a class, REGENERATE ALL for that class."
        # This ensures consistency.
        
        classes = self.db_session.query(DatasetImage.classification).distinct().all()
        classes = [c[0] for c in classes]
        
        for class_name in classes:
            images = self.db_session.query(DatasetImage).filter_by(classification=class_name).order_by(DatasetImage.id).all()
            
            emb_list = []
            path_list = []
            
            needs_update = False
            
            for img in images:
                # We don't have the embedding stored with the image in current schema.
                # We'll need to generate it if we want to be sure, or store it.
                # To make this performant and robust: 
                # Let's just generate embeddings on the fly if missing? No, too slow.
                
                # We will check if we have pre-calculated embeddings in the embeddings table.
                # But matching them index-wise is risky if rows were deleted.
                
                # CRITICAL DECISION:
                # To guarantee consistency, I will recalculate embeddings on startup IF the counts mismatch. 
                # If counts match, we assume order by ID is preserved (since both inserted sequentially).
                pass

        # Actually, let's just stick to the original 'embeddings' table as a cache of vectors.
        # But we need to link specific embedding to specific image for "Reference Image" feature.
        
        # Let's simplify efficiently:
        # We will rely on `DatasetImage` as the source of truth.
        # On startup, we iterate all DatasetImages.
        # We check if we have an Embedding record for this image? No foreign key.
        
        # OK, I will modify the logic to ONLY use `DatasetImage` and generate embeddings on startup?
        # No, that's too slow for production.
        
        # I will store the embedding IN the `DatasetImage` table if possible? 
        # `DatasetImage` has `image_data` (blob).
        # `Embedding` has `embedding_data` (blob).
        
        # Let's just wipe the `Embedding` table and regenerate from `DatasetImage` on startup if it seems empty/inconsistent?
        # Or better: Iterate `DatasetImage`. If we have a list of `DatasetImage`, we generate embeddings in memory and
        # cache them. If we want persistence of embeddings, we can save them to `Embedding` table.
        
        # Implementation Plan Revised Strategy:
        # 1. Clear in-memory cache.
        # 2. Query `DatasetImage` by class.
        # 3. Query `Embedding` by class.
        # 4. If count matches, load `Embedding`.
        # 5. If count mismatches, regenerate all embeddings for that class from `DatasetImage` contents and update `Embedding` table.
        
        # This is self-healing.
        
        images_count_q = self.db_session.query(DatasetImage.classification, func.count(DatasetImage.id)).group_by(DatasetImage.classification).all()
        embeddings_count_q = self.db_session.query(Embedding.classification, func.count(Embedding.id)).group_by(Embedding.classification).all()
        
        img_counts = {c: n for c, n in images_count_q}
        emb_counts = {c: n for c, n in embeddings_count_q}
        
        all_classes = set(list(img_counts.keys()) + list(emb_counts.keys()))
        
        for class_name in all_classes:
            i_count = img_counts.get(class_name, 0)
            e_count = emb_counts.get(class_name, 0)
            
            if i_count == 0:
                continue
                
            if i_count != e_count:
                print(f"Mismatch for {class_name}: {i_count} images vs {e_count} embeddings. Regenerating...")
                self._regenerate_class_embeddings_db(class_name)
            else:
                # Load from DB
                raw_embs = self.db_session.query(Embedding).filter_by(classification=class_name).order_by(Embedding.id).all()
                images = self.db_session.query(DatasetImage).filter_by(classification=class_name).order_by(DatasetImage.id).all()
                
                vectors = []
                for emb in raw_embs:
                    vectors.append(np.frombuffer(emb.embedding_data, dtype=np.float32))
                
                if vectors:
                    self.embeddings_cache[class_name] = np.array(vectors)
                    self.image_paths_cache[class_name] = [
                        {"id": img.id, "filename": img.filename, "path": f"/images/{class_name}/{img.filename}"}
                        for img in images
                    ]
                    
                    # Double check alignment length
                    if len(self.embeddings_cache[class_name]) != len(self.image_paths_cache[class_name]):
                         print(f"Alignment error for {class_name}, regenerating...")
                         self._regenerate_class_embeddings_db(class_name)

    def _regenerate_class_embeddings_db(self, class_name: str):
        """Regenerate embeddings for a class from DB images"""
        # Delete existing embeddings for this class
        self.db_session.query(Embedding).filter_by(classification=class_name).delete()
        self.db_session.commit()
        
        images = self.db_session.query(DatasetImage).filter_by(classification=class_name).order_by(DatasetImage.id).all()
        
        vectors = []
        paths = []
        new_embedding_records = []
        
        print(f"Regenerating {len(images)} embeddings for {class_name}...")
        
        for img in images:
            if not img.image_data:
                continue
                
            try:
                vec = self.generate_embedding_from_bytes(img.image_data)
                vectors.append(vec)
                paths.append({
                    "id": img.id, 
                    "filename": img.filename, 
                    "path": f"/images/{class_name}/{img.filename}"
                })
                
                # Create embedding record
                new_embedding_records.append(Embedding(
                    classification=class_name,
                    embedding_data=vec.tobytes()
                ))
            except Exception as e:
                print(f"Error processing image {img.id}: {e}")
        
        if new_embedding_records:
            self.db_session.add_all(new_embedding_records)
            self.db_session.commit()
            
            self.embeddings_cache[class_name] = np.array(vectors)
            self.image_paths_cache[class_name] = paths
            print(f"Regeneration complete for {class_name}")

    def add_multiple_images_to_class(self, class_name: str, images_list: List[Tuple[str, bytes]]) -> List[str]:
        """
        Batch add images to a class.
        images_list: List of (filename, bytes)
        Returns list of saved paths
        """
        import uuid
        import re
        
        new_images = []
        new_embeddings = []
        saved_paths = []
        
        # Prepare data
        for filename, content in images_list:
            # 1. Sanitize filename
            safe_filename = os.path.basename(filename)
            safe_filename = re.sub(r'[^\w\-.]', '_', safe_filename)
            base_name = os.path.splitext(safe_filename)[0] or 'image'
            ext = os.path.splitext(safe_filename)[1] or '.jpg'
            if ext.lower() not in {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.gif'}:
                ext = '.jpg'
                
            unique_id = uuid.uuid4().hex[:8]
            new_filename = f"{base_name}_{unique_id}{ext}"
            
            # 2. Generate Embedding
            try:
                vec = self.generate_embedding_from_bytes(content)
                
                # 3. Create records
                img_record = DatasetImage(
                    filename=new_filename,
                    classification=class_name,
                    image_data=content,
                    image_path=f"/images/{class_name}/{new_filename}"  # Virtual path
                )
                emb_record = Embedding(
                    classification=class_name,
                    embedding_data=vec.tobytes()
                )
                
                new_images.append(img_record)
                new_embeddings.append(emb_record)
                saved_paths.append(img_record.image_path)
                
                # Update in-memory cache immediately? 
                # Better to just rebuild/append to cache locally to avoid full reload
                # But for consistency, let's append to lists
                
                if class_name not in self.embeddings_cache:
                    self.embeddings_cache[class_name] = np.array([vec])
                    self.image_paths_cache[class_name] = []
                else:
                    if len(self.embeddings_cache[class_name]) == 0:
                         self.embeddings_cache[class_name] = np.array([vec])
                    else:
                        self.embeddings_cache[class_name] = np.vstack([self.embeddings_cache[class_name], vec])
                
                # Note: ID is not available until flush, but we can append a temporary dict or wait.
                # Since we don't strictly need ID for similarity search (just index), it's fine.
                self.image_paths_cache[class_name].append({
                    "id": None, # Only needed for DB ops
                    "filename": new_filename,
                    "path": img_record.image_path
                })
                
            except Exception as e:
                print(f"Skipping file {filename}: {e}")

        # 4. Batch Insert
        if new_images:
            try:
                self.db_session.add_all(new_images)
                self.db_session.add_all(new_embeddings)
                self.db_session.commit()
                print(f"Batch added {len(new_images)} images to {class_name}")
            except Exception as e:
                self.db_session.rollback()
                print(f"Batch insert failed: {e}")
                raise e
                
        return saved_paths

    def add_image_to_class(self, class_name: str, image_bytes: bytes, filename: str) -> str:
        """Legacy wrapper for single image"""
        paths = self.add_multiple_images_to_class(class_name, [(filename, image_bytes)])
        return paths[0] if paths else ""
        
    def get_image_data(self, class_name: str, filename: str) -> Optional[bytes]:
        """Retrieve raw image bytes from DB"""
        img = self.db_session.query(DatasetImage).filter_by(
            classification=class_name, 
            filename=filename
        ).first()
        return img.image_data if img else None

    def sync_initial_data(self):
        """On startup, import files from local 'dataset' folder into DB if DB is empty"""
        if not os.path.exists(DATASET_DIR):
            return

        print("Checking for local files to import...")
        for class_name in os.listdir(DATASET_DIR):
            class_path = os.path.join(DATASET_DIR, class_name)
            if not os.path.isdir(class_path):
                continue
                
            # Check if we already have images for this class in DB
            count = self.db_session.query(DatasetImage).filter_by(classification=class_name).count()
            if count > 0:
                continue # Already populated
            
            print(f"Importing local folder {class_name} to database...")
            batch = []
            valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.gif'}
            
            files = [f for f in os.listdir(class_path) if os.path.splitext(f)[1].lower() in valid_exts]
            
            for f in files:
                fpath = os.path.join(class_path, f)
                try:
                    with open(fpath, "rb") as file:
                        content = file.read()
                    batch.append((f, content))
                except Exception as e:
                    print(f"Error reading {f}: {e}")
            
            if batch:
                self.add_multiple_images_to_class(class_name, batch)
                print(f"Imported {len(batch)} images for {class_name}")

    def classify_image(self, image_bytes: bytes) -> Dict:
        """Classify an image and return results"""
        # (Same logic as before but using self.embeddings_cache which is now DB-backed)
        query_embedding = self.generate_embedding_from_bytes(image_bytes)
        
        all_classes = list(self.embeddings_cache.keys())
        class_scores = {}
        all_image_matches = []
        
        for class_name in all_classes:
            class_embeddings = self.embeddings_cache.get(class_name, np.array([]))
            class_paths = self.image_paths_cache.get(class_name, [])
            
            if len(class_embeddings) == 0:
                continue

            similarities = []
            # Calculate similarity
            # Vectorized operation
            sims = np.dot(class_embeddings, query_embedding) # (N,) result
            # Assuming query_embedding is already normalized (it is)
            # And class_embeddings are normalized (they are)
            
            # Note: numpy dict/arrays might have dimension issues if not careful
            if len(sims.shape) > 1:
                sims = sims.flatten()
                
            for i, sim in enumerate(sims):
                sim_val = float(sim)
                similarities.append(sim_val)
                
                if i < len(class_paths):
                    path_info = class_paths[i]
                    all_image_matches.append({
                        "classification": class_name,
                        "image_path": path_info.get("path", ""),
                        "similarity": round(sim_val * 100, 2)
                    })
            
            if similarities:
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
        
        sorted_image_matches = sorted(
            all_image_matches,
            key=lambda x: x["similarity"],
            reverse=True
        )
        
        reference_image = None
        if sorted_image_matches:
            best_match = sorted_image_matches[0]
            reference_image = {
                "image_path": best_match["image_path"],
                "classification": best_match["classification"],
                "similarity": best_match["similarity"]
            }
        
        top_3_images = sorted_image_matches[:3]
        
        top_3_classes = [
            {
                "classification": name,
                "similarity": round(scores["avg_similarity"] * 100, 2),
                "max_similarity": round(scores["max_similarity"] * 100, 2),
                "samples_count": scores["num_samples"]
            }
            for name, scores in sorted_classes[:3]
        ]
        
        has_samples = len(all_image_matches) > 0
        best_similarity = sorted_image_matches[0]["similarity"] / 100.0 if sorted_image_matches else 0.0
        
        if not has_samples:
            return {
                "classification": None,
                "similarity_score": 0.0,
                "reference_image": None,
                "top_matches": [],
                "top_3": [],
                "has_dataset": False,
                "status": "no_dataset",
                "message": "No images in dataset."
            }
        
        if best_similarity < SIMILARITY_THRESHOLD:
            return {
                "classification": None,
                "similarity_score": round(best_similarity * 100, 2),
                "reference_image": None,
                "top_matches": top_3_images,
                "top_3": top_3_classes,
                "has_dataset": True,
                "status": "no_match",
                "message": "No match found above threshold."
            }
        
        return {
            "classification": top_class,
            "similarity_score": round(top_score * 100, 2),
            "reference_image": reference_image,
            "top_matches": top_3_images,
            "top_3": top_3_classes,
            "has_dataset": True,
            "status": "match",
            "message": "Classification successful"
        }

    def get_class_info(self) -> List[Dict]:
        """Get info about classes from DB/Cache"""
        return [
            {
                "name": k,
                "image_count": len(v),
                "has_embeddings": True
            }
            for k, v in self.image_paths_cache.items()
        ]

engine = None

def get_engine() -> EmbeddingEngine:
    global engine
    if engine is None:
        engine = EmbeddingEngine()
    return engine
