"""
Vectorisation Pipeline for CrisisMMD Dataset (Lightweight Version).

This script generates 512-dimensional CLIP embeddings for disaster imagery.
Uses sentence-transformers for lower memory footprint.

Author: Rashid
Supervisor: XinHui Ma
Project: Visualising Natural Disaster Image Embeddings
"""

import os
import sys
import numpy as np
from PIL import Image
from tqdm import tqdm
import json
from pathlib import Path
from datetime import datetime


# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# Configuration
DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "clean_data"
OUTPUT_DIR = PROJECT_ROOT / "data" / "embeddings"
BATCH_SIZE = 16


def load_model():
    """Load CLIP model using sentence-transformers (lighter memory footprint)."""
    print("Step 1: Loading CLIP Model...")
    
    from sentence_transformers import SentenceTransformer
    
    print("  Using sentence-transformers (memory-efficient)")
    print("  Downloading clip-ViT-B-32...")
    
    model = SentenceTransformer("clip-ViT-B-32")
    
    print("  Model loaded successfully")
    return model


def find_images(root_dir):
    """Find all image files in directory."""
    print("Step 2: Scanning for Images...")
    
    image_paths = []
    extensions = {".jpg", ".jpeg", ".png"}
    
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if Path(file).suffix.lower() in extensions:
                image_paths.append(os.path.join(root, file))
    
    print("  Found {} images".format(len(image_paths)))
    return image_paths


def process_images(model, image_paths):
    """Generate embeddings for all images."""
    print("Step 3: Generating Embeddings...")
    print("  Batch size: {}".format(BATCH_SIZE))
    
    embeddings = []
    valid_paths = []
    corrupt_files = []
    
    for i in tqdm(range(0, len(image_paths), BATCH_SIZE), desc="  Processing"):
        batch_paths = image_paths[i:i + BATCH_SIZE]
        batch_images = []
        batch_valid_paths = []
        
        for path in batch_paths:
            try:
                img = Image.open(path).convert("RGB")
                batch_images.append(img)
                batch_valid_paths.append(path)
            except Exception as e:
                corrupt_files.append((path, str(e)))
        
        if batch_images:
            try:
                batch_embeddings = model.encode(batch_images, convert_to_numpy=True)
                embeddings.append(batch_embeddings)
                valid_paths.extend(batch_valid_paths)
            except Exception as e:
                for path in batch_valid_paths:
                    corrupt_files.append((path, str(e)))
    
    if embeddings:
        embeddings = np.vstack(embeddings)
    else:
        embeddings = np.array([])
    
    print("  Generated {} embeddings".format(len(embeddings)))
    print("  Skipped {} corrupt files".format(len(corrupt_files)))
    
    return embeddings, valid_paths, corrupt_files


def save_results(embeddings, paths, corrupt_files):
    """Save embeddings and metadata."""
    print("Step 4: Saving Results...")
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save embeddings
    np.save(OUTPUT_DIR / "embeddings.npy", embeddings)
    print("  Saved embeddings.npy ({} x {})".format(*embeddings.shape))
    
    # Save filenames
    with open(OUTPUT_DIR / "filenames.json", "w") as f:
        json.dump(paths, f, indent=2)
    print("  Saved filenames.json")
    
    # Save metadata
    metadata = {
        "generated_at": datetime.now().isoformat(),
        "n_images": len(paths),
        "embedding_dim": embeddings.shape[1] if len(embeddings) > 0 else 0,
        "model": "clip-ViT-B-32",
        "batch_size": BATCH_SIZE
    }
    with open(OUTPUT_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    # Save corrupt files log
    if corrupt_files:
        with open(OUTPUT_DIR / "corrupt_files.log", "w") as f:
            for path, error in corrupt_files:
                f.write("{}: {}\n".format(path, error))


def main():
    """Main entry point."""
    print("CrisisMMD Embedding Generator (Lightweight)")
    print("Started: {}".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    print("")
    
    if not DATASET_PATH.exists():
        print("ERROR: Dataset not found at {}".format(DATASET_PATH))
        print("Run clean_data.py first")
        sys.exit(1)
    
    model = load_model()
    image_paths = find_images(DATASET_PATH)
    
    if not image_paths:
        print("ERROR: No images found")
        sys.exit(1)
    
    embeddings, valid_paths, corrupt_files = process_images(model, image_paths)
    save_results(embeddings, valid_paths, corrupt_files)
    
    print("")
    print("Pipeline complete!")
    print("Next: python -m src.umap_reduction")


if __name__ == "__main__":
    main()
