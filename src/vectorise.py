"""
Vectorisation Pipeline for CrisisMMD Dataset.

This script generates 512-dimensional CLIP embeddings using the native
Transformers library to ensure compatibility with the production app.

Author: Rashid
Supervisor: XinHui Ma
Project: Visualising Natural Disaster Image Embeddings
"""

import sys
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import json
from pathlib import Path
from datetime import datetime
from typing import Tuple, List, Any
from transformers import CLIPProcessor, CLIPModel

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import config

# Configuration
DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "clean_data"
OUTPUT_DIR = PROJECT_ROOT / "data" / "embeddings"


def load_model() -> Tuple[CLIPModel, CLIPProcessor, str]:
    """Load CLIP model using Hugging Face Transformers.
    
    Returns:
        Tuple containing the model, processor, and device string.
    """
    print("Step 1: Loading CLIP Model...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("  Inference Device: {}".format(device))
    
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    print("  Model loaded successfully")
    return model, processor, device


def find_images(root_dir: Path) -> List[str]:
    """Find all image files in directory.
    
    Args:
        root_dir: Root directory to search for images.
        
    Returns:
        List of image file paths.
    """
    print("Step 2: Scanning for Images...")
    
    image_paths = []
    extensions = set(config.IMAGE_EXTENSIONS)

    # Use pathlib rglob for cleaner iteration
    for ext in extensions:
        for path in root_dir.rglob("*{}".format(ext)):
            image_paths.append(str(path))
            
    print("  Found {} images".format(len(image_paths)))
    return image_paths


def process_images(
    model: CLIPModel, 
    processor: CLIPProcessor, 
    device: str, 
    image_paths: List[str]
) -> Tuple[np.ndarray, List[str], List[Tuple[str, str]]]:
    """Generate embeddings for all images.
    
    Args:
        model: CLIP model instance.
        processor: CLIP processor instance.
        device: Device string ('cuda' or 'cpu').
        image_paths: List of image file paths.
        
    Returns:
        Tuple of (embeddings array, valid paths, corrupt files list).
    """
    batch_size = config.VECTORISE_BATCH_SIZE
    print("Step 3: Generating Embeddings...")
    print("  Batch size: {}".format(batch_size))

    embeddings = []
    valid_paths = []
    corrupt_files = []

    for i in tqdm(range(0, len(image_paths), batch_size), desc="  Processing"):
        batch_paths = image_paths[i:i + batch_size]
        batch_images = []
        batch_valid_paths = []
        
        # 1. Load Images
        for path in batch_paths:
            try:
                img = Image.open(path).convert("RGB")
                batch_images.append(img)
                batch_valid_paths.append(path)
            except Exception as e:
                corrupt_files.append((path, str(e)))
        
        if not batch_images:
            continue
            
        # 2. Process via CLIP
        try:
            inputs = processor(images=batch_images, return_tensors="pt", padding=True).to(device)
            
            with torch.no_grad():
                outputs = model.get_image_features(**inputs)
            
            # Normalise which is needed for Cosine Similarity
            outputs = outputs / outputs.norm(p=2, dim=-1, keepdim=True)
            
            embeddings.append(outputs.cpu().numpy())
            valid_paths.extend(batch_valid_paths)
            
        except Exception as e:
            print("  Batch Error: {}".format(e))
            for path in batch_valid_paths:
                corrupt_files.append((path, str(e)))
    
    if embeddings:
        embeddings = np.vstack(embeddings)
    else:
        embeddings = np.array([])
    
    print("  Generated {} embeddings".format(len(embeddings)))
    return embeddings, valid_paths, corrupt_files


def save_results(
    embeddings: np.ndarray, 
    paths: List[str], 
    corrupt_files: List[Tuple[str, str]]
) -> None:
    """Save embeddings and metadata.
    
    Args:
        embeddings: NumPy array of CLIP embeddings.
        paths: List of valid image paths.
        corrupt_files: List of (path, error) tuples for failed images.
    """
    print("Step 4: Saving Results...")
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    np.save(OUTPUT_DIR / "embeddings.npy", embeddings)
    print("  Saved embeddings.npy ({})".format(embeddings.shape))
    
    with open(OUTPUT_DIR / "filenames.json", "w") as f:
        json.dump(paths, f, indent=2)
    print("  Saved filenames.json")


def main() -> None:
    """Main entry point for the vectorisation pipeline."""
    print("Embedding Generator using Transformers")
    
    if not DATASET_PATH.exists():
        print("Error: Dataset not found at {}".format(DATASET_PATH))
        sys.exit(1)
    
    model, processor, device = load_model()
    image_paths = find_images(DATASET_PATH)
    
    embeddings, valid_paths, corrupt_files = process_images(model, processor, device, image_paths)
    save_results(embeddings, valid_paths, corrupt_files)
    
    print("\nPipeline complete! Now run UMAP reduction.")


if __name__ == "__main__":
    main()
