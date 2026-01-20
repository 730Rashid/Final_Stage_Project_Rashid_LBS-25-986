"""
Vectorisation Pipeline for CrisisMMD Dataset.

This script generates 512-dimensional CLIP embeddings for disaster imagery.
It forms the 'offline' component of the Offline-Online architecture described
in my thesis. The embeddings are saved to disk and can then be used by the
online Dash frontend for interactive visualisation and semantic search.

Author: Rashid
Supervisor: XinHui Ma
Project: Visualising Natural Disaster Image Embeddings
"""

import os
import sys
import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm
import json
from pathlib import Path
from datetime import datetime


# Add project root to path so we can import our own modules
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# Configuration
# Change DATASET_PATH to point to your images folder
DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "CrisisMMD_v2.0" / "data_image"

# Output directory for embeddings
OUTPUT_DIR = PROJECT_ROOT / "data" / "embeddings"

# Batch size for processing
# I set this to 16 because my MX450 GPU only has 2GB VRAM
# If you have a better GPU, you can increase this to 32 or 64
BATCH_SIZE = 16


def load_model():
    """
    Load the CLIP model and processor.
    
    This function automatically detects the best available hardware
    (CUDA GPU, Apple Silicon, or CPU) and loads the model accordingly.
    
    Returns:
        tuple: The model, processor, and device string.
    """
    print("Step 1: Loading CLIP Model...")
    
    # Check what hardware is available
    if torch.cuda.is_available():
        device = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print("  GPU detected: {}".format(gpu_name))
        print("  VRAM available: {:.1f} GB".format(gpu_memory))
        print("  Using batch size: {}".format(BATCH_SIZE))
    elif torch.backends.mps.is_available():
        device = "mps"
        print("  Apple Silicon detected")
    else:
        device = "cpu"
        print("  No GPU found, running on CPU (this will be slower)")

    # Download and load the model from HuggingFace
    print("  Downloading model weights (this may take a moment)...")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    # Set to evaluation mode since we are not training
    model.eval()
    
    print("Model loaded successfully")
    print("")
    return model, processor, device


def find_images(root_dir):
    """
    Recursively find all image files in a directory.
    
    This function walks through all subdirectories and collects paths
    to any .jpg, .jpeg, or .png files it finds.
    
    Args:
        root_dir: Path to the root directory to scan.
        
    Returns:
        list: A list of paths to image files.
    """
    print("Step 2: Scanning for images...")
    print("  Directory: {}".format(root_dir))
    
    image_paths = []
    valid_extensions = (".png", ".jpg", ".jpeg")
    
    root_path = Path(root_dir)
    
    # Check the directory actually exists
    if not root_path.exists():
        print("  ERROR: Directory does not exist")
        print("  Please check the path and try again")
        return []
    
    # Walk through all files recursively
    for file_path in root_path.rglob("*"):
        if file_path.suffix.lower() in valid_extensions:
            image_paths.append(str(file_path))
    
    print("  Found {:,} images".format(len(image_paths)))
    
    # Show a breakdown by category if there are subdirectories
    if image_paths:
        subdirs = {}
        for p in image_paths:
            rel = Path(p).relative_to(root_path)
            if len(rel.parts) > 1:
                subdir = rel.parts[0]
                subdirs[subdir] = subdirs.get(subdir, 0) + 1
        
        if subdirs:
            print("  Breakdown by event type:")
            sorted_subdirs = sorted(subdirs.items(), key=lambda x: -x[1])
            for subdir, count in sorted_subdirs[:8]:
                print("    {}: {:,}".format(subdir, count))
            if len(subdirs) > 8:
                print("    ... and {} more categories".format(len(subdirs) - 8))
    
    print("")
    return image_paths


def process_images(model, processor, device, image_paths):
    """
    Process images through CLIP to generate embeddings.
    
    This is the main vectorisation loop. It processes images in batches
    to make efficient use of GPU memory, and handles any corrupt files
    gracefully by logging them and continuing.
    
    Args:
        model: The CLIP model.
        processor: The CLIP processor for image preprocessing.
        device: The device to run inference on.
        image_paths: List of paths to images.
        
    Returns:
        tuple: (embeddings array, list of valid paths, list of corrupt files)
    """
    print("Step 3: Generating embeddings...")
    print("  Processing {:,} images in batches of {}".format(len(image_paths), BATCH_SIZE))
    print("")
    
    all_embeddings = []
    valid_image_paths = []
    corrupt_files = []
    
    start_time = datetime.now()
    total_batches = (len(image_paths) + BATCH_SIZE - 1) // BATCH_SIZE
    
    # Process images in batches
    for i in tqdm(range(0, len(image_paths), BATCH_SIZE), 
                  desc="  Processing", 
                  unit="batch",
                  total=total_batches):
        
        batch_paths = image_paths[i:i + BATCH_SIZE]
        batch_images = []
        batch_valid_paths = []

        # Try to load each image in the batch
        for path in batch_paths:
            try:
                img = Image.open(path).convert("RGB")
                batch_images.append(img)
                batch_valid_paths.append(path)
            except Exception as e:
                corrupt_files.append((path, str(e)))

        # Skip empty batches
        if not batch_images:
            continue

        # Run the batch through CLIP
        try:
            inputs = processor(
                images=batch_images, 
                return_tensors="pt", 
                padding=True
            ).to(device)
            
            # We use no_grad because we are not training the model
            with torch.no_grad():
                outputs = model.get_image_features(**inputs)
            
            # Normalise the vectors for cosine similarity
            outputs = outputs / outputs.norm(p=2, dim=-1, keepdim=True)
            
            embeddings = outputs.cpu().numpy()
            
            all_embeddings.append(embeddings)
            valid_image_paths.extend(batch_valid_paths)
            
            # Clear GPU cache periodically to avoid memory buildup
            if device == "cuda" and i % (BATCH_SIZE * 10) == 0:
                torch.cuda.empty_cache()
                
        except Exception as e:
            tqdm.write("  Warning: Batch {} failed with error: {}".format(i // BATCH_SIZE, e))

    # Calculate how long it took
    elapsed = datetime.now() - start_time
    
    print("")
    print("  Processing complete")
    print("  Successfully processed: {:,} images".format(len(valid_image_paths)))
    print("  Corrupt or skipped: {}".format(len(corrupt_files)))
    print("  Total time: {}".format(elapsed))
    
    if valid_image_paths:
        speed = len(valid_image_paths) / elapsed.total_seconds()
        print("  Speed: {:.1f} images per second".format(speed))
    
    print("")
    
    # Stack all the batch results into one array
    if all_embeddings:
        final_embeddings = np.vstack(all_embeddings)
        return final_embeddings, valid_image_paths, corrupt_files
    else:
        return None, None, corrupt_files


def save_results(embeddings, paths, corrupt_files, output_dir):
    """
    Save the generated embeddings and metadata to disk.
    
    This creates several output files:
    - embeddings.npy: The actual vectors (N x 512 matrix)
    - filenames.json: Maps each row to its source image path
    - corrupt_files.log: List of files that could not be processed
    - metadata.json: Information about how the embeddings were generated
    
    Args:
        embeddings: The embedding matrix.
        paths: List of image paths corresponding to each row.
        corrupt_files: List of (path, error) tuples for failed files.
        output_dir: Directory to save outputs to.
    """
    print("Step 4: Saving results...")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save the embeddings
    embeddings_file = output_path / "embeddings.npy"
    np.save(embeddings_file, embeddings)
    file_size_mb = embeddings.nbytes / (1024 ** 2)
    print("  Saved embeddings to: {}".format(embeddings_file))
    print("  Shape: {} ({:.1f} MB)".format(embeddings.shape, file_size_mb))
    
    # Save the filenames
    filenames_file = output_path / "filenames.json"
    with open(filenames_file, "w", encoding="utf-8") as f:
        json.dump(paths, f, indent=2)
    print("  Saved filenames to: {}".format(filenames_file))
    
    # Save the corrupt files log
    if corrupt_files:
        log_file = output_path / "corrupt_files.log"
        with open(log_file, "w", encoding="utf-8") as f:
            for path, error in corrupt_files:
                f.write("{}\t{}\n".format(path, error))
        print("  Saved error log to: {}".format(log_file))
    
    # Save metadata for reproducibility
    metadata = {
        "generated_at": datetime.now().isoformat(),
        "model": "openai/clip-vit-base-patch32",
        "embedding_dimension": 512,
        "total_images": len(paths),
        "corrupt_files": len(corrupt_files),
        "batch_size": BATCH_SIZE,
        "normalised": True
    }
    
    metadata_file = output_path / "metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)
    print("  Saved metadata to: {}".format(metadata_file))
    
    print("")
    print("Pipeline Complete. Total vectors: {:,}".format(len(embeddings)))
    print("")
    print("  Next steps:")
    print("  1. Run UMAP dimensionality reduction")
    print("  2. Launch the Dash visualisation frontend")
    print("")


def main():
    """
    Main entry point for the vectorisation pipeline.
    """
    print("")
    print("Starting CrisisMMD Embedding Generator...")
    print("Timestamp: {}".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    
    # Load the model
    model, processor, device = load_model()
    
    # Find all images
    all_files = find_images(DATASET_PATH)
    
    if len(all_files) == 0:
        print("ERROR: No images found")
        print("Please check that DATASET_PATH is correct: {}".format(DATASET_PATH))
        sys.exit(1)

    # Generate embeddings
    embeddings, paths, corrupt = process_images(model, processor, device, all_files)

    # Save everything
    if embeddings is not None:
        save_results(embeddings, paths, corrupt, OUTPUT_DIR)
    else:
        print("ERROR: Failed to generate any embeddings")
        sys.exit(1)


if __name__ == "__main__":
    main()
