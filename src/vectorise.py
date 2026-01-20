"""
Vectorisation Pipeline for CrisisMMD Dataset
Generates 512-dimensional CLIP embeddings for disaster imagery.

This is the 'offline' component of the Offline-Online architecture.
Run this once to generate embeddings, then use them in the online Dash frontend.

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

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# --- CONFIGURATION ---
# The folder where your CrisisMMD images are located
# Use clean_data if you've run the cleaning pipeline, otherwise use data_image
DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "CrisisMMD_v2.0" / "data_image"

# Alternative: Point to cleaned data after running clean_data.py
# DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "clean_data"

# Where we will save the output files
OUTPUT_DIR = PROJECT_ROOT / "data" / "embeddings"

# --- HARDWARE-OPTIMISED SETTINGS ---
# MX450 has ~2GB VRAM, so we use conservative batch sizes
# Increase to 32-64 if you have a more powerful GPU
BATCH_SIZE = 16  # Safe for MX450 with 2GB VRAM

# For very large datasets, process in chunks and save periodically
CHECKPOINT_EVERY = 1000  # Save progress every N images


def load_model():
    """
    Loads the CLIP model (ViT-B/32).
    Automatically detects and uses the best available hardware.
    
    Returns:
        tuple: (model, processor, device)
    """
    print("=" * 60)
    print(" [1/4] Loading CLIP Model (ViT-B/32)...")
    print("=" * 60)
    
    # Check for hardware acceleration
    if torch.cuda.is_available():
        device = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"   ✓ GPU Detected: {gpu_name}")
        print(f"   ✓ VRAM: {gpu_memory:.1f} GB")
        print(f"   ✓ Batch Size: {BATCH_SIZE} (optimised for your GPU)")
    elif torch.backends.mps.is_available():
        device = "mps"
        print("   ✓ Apple Silicon Detected (MPS)")
    else:
        device = "cpu"
        print("   ⚠ No GPU detected. Running on CPU (this will be slower).")

    # Load the CLIP model from HuggingFace
    print("   > Downloading/loading model weights...")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    # Set model to evaluation mode
    model.eval()
    
    print("   ✓ Model loaded successfully!")
    return model, processor, device


def find_images(root_dir):
    """
    Recursively finds all image files (.jpg, .jpeg, .png) in the directory.
    
    Args:
        root_dir: Path to the root directory to scan
        
    Returns:
        list: List of image file paths
    """
    print("\n" + "=" * 60)
    print(f" [2/4] Scanning for images...")
    print("=" * 60)
    print(f"   > Directory: {root_dir}")
    
    image_paths = []
    valid_extensions = ('.png', '.jpg', '.jpeg')
    
    root_path = Path(root_dir)
    
    if not root_path.exists():
        print(f"   ✗ ERROR: Directory does not exist!")
        print(f"   > Please check the path: {root_dir}")
        return []
    
    for file_path in root_path.rglob("*"):
        if file_path.suffix.lower() in valid_extensions:
            image_paths.append(str(file_path))
    
    print(f"   ✓ Found {len(image_paths):,} images")
    
    # Show breakdown by subdirectory
    if image_paths:
        subdirs = {}
        for p in image_paths:
            rel = Path(p).relative_to(root_path)
            if len(rel.parts) > 1:
                subdir = rel.parts[0]
                subdirs[subdir] = subdirs.get(subdir, 0) + 1
        
        if subdirs:
            print("   > Breakdown by event type:")
            for subdir, count in sorted(subdirs.items(), key=lambda x: -x[1])[:10]:
                print(f"      - {subdir}: {count:,} images")
            if len(subdirs) > 10:
                print(f"      ... and {len(subdirs) - 10} more categories")
    
    return image_paths


def process_images(model, processor, device, image_paths):
    """
    Main vectorisation loop. Processes images in batches through CLIP.
    
    Features:
    - Robust error handling for corrupt files
    - Progress bar with ETA
    - Periodic checkpointing for large datasets
    - Memory-efficient batch processing
    
    Args:
        model: CLIP model
        processor: CLIP processor
        device: PyTorch device
        image_paths: List of image file paths
        
    Returns:
        tuple: (embeddings array, valid paths list)
    """
    print("\n" + "=" * 60)
    print(" [3/4] Vectorisation Pipeline")
    print("=" * 60)
    print(f"   > Processing {len(image_paths):,} images in batches of {BATCH_SIZE}")
    
    all_embeddings = []
    valid_image_paths = []
    corrupt_files = []
    failed_batches = []
    
    start_time = datetime.now()
    
    # Process in batches
    total_batches = (len(image_paths) + BATCH_SIZE - 1) // BATCH_SIZE
    
    for i in tqdm(range(0, len(image_paths), BATCH_SIZE), 
                  desc="   Processing", 
                  unit="batch",
                  total=total_batches):
        
        batch_paths = image_paths[i:i + BATCH_SIZE]
        batch_images = []
        batch_valid_paths = []

        # 1. Load and validate images in this batch
        for path in batch_paths:
            try:
                # Open image and convert to RGB
                img = Image.open(path).convert("RGB")
                batch_images.append(img)
                batch_valid_paths.append(path)
            except Exception as e:
                corrupt_files.append((path, str(e)))

        if not batch_images:
            continue

        # 2. Process batch through CLIP
        try:
            # Preprocess for the model
            inputs = processor(
                images=batch_images, 
                return_tensors="pt", 
                padding=True
            ).to(device)
            
            # Forward pass (no gradients needed - saves memory)
            with torch.no_grad():
                outputs = model.get_image_features(**inputs)
            
            # L2 normalise embeddings (critical for cosine similarity later)
            outputs = outputs / outputs.norm(p=2, dim=-1, keepdim=True)
            
            # Move to CPU and convert to numpy
            embeddings = outputs.cpu().numpy()
            
            all_embeddings.append(embeddings)
            valid_image_paths.extend(batch_valid_paths)
            
            # Clear GPU cache periodically to prevent memory buildup
            if device == "cuda" and i % (BATCH_SIZE * 10) == 0:
                torch.cuda.empty_cache()
                
        except Exception as e:
            failed_batches.append((i, str(e)))
            tqdm.write(f"   ⚠ Batch {i//BATCH_SIZE} failed: {e}")

    # Compile results
    elapsed = datetime.now() - start_time
    
    print("\n   " + "-" * 50)
    print("   VECTORISATION COMPLETE")
    print("   " + "-" * 50)
    print(f"   ✓ Successfully processed: {len(valid_image_paths):,} images")
    print(f"   ⚠ Corrupt/skipped files: {len(corrupt_files)}")
    print(f"   ✗ Failed batches: {len(failed_batches)}")
    print(f"   ⏱ Total time: {elapsed}")
    print(f"   ⏱ Speed: {len(valid_image_paths) / elapsed.total_seconds():.1f} images/second")
    
    # Stack all batch embeddings into single array
    if all_embeddings:
        final_embeddings = np.vstack(all_embeddings)
        return final_embeddings, valid_image_paths, corrupt_files
    else:
        return None, None, corrupt_files


def save_results(embeddings, paths, corrupt_files, output_dir):
    """
    Saves the embeddings and metadata to disk.
    
    Outputs:
    - embeddings.npy: The 512-dim vectors (N x 512)
    - filenames.json: Mapping of row index to image path
    - corrupt_files.log: List of files that couldn't be processed
    - metadata.json: Processing metadata for reproducibility
    """
    print("\n" + "=" * 60)
    print(" [4/4] Saving Results")
    print("=" * 60)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 1. Save embeddings (the core data)
    embeddings_file = output_path / "embeddings.npy"
    np.save(embeddings_file, embeddings)
    print(f"   ✓ Embeddings: {embeddings_file}")
    print(f"     Shape: {embeddings.shape} ({embeddings.nbytes / (1024**2):.1f} MB)")
    
    # 2. Save filenames mapping
    filenames_file = output_path / "filenames.json"
    with open(filenames_file, 'w', encoding='utf-8') as f:
        json.dump(paths, f, indent=2)
    print(f"   ✓ Filenames: {filenames_file}")
    
    # 3. Save corrupt files log
    if corrupt_files:
        log_file = output_path / "corrupt_files.log"
        with open(log_file, 'w', encoding='utf-8') as f:
            for path, error in corrupt_files:
                f.write(f"{path}\t{error}\n")
        print(f"   ✓ Error log: {log_file}")
    
    # 4. Save metadata for reproducibility
    metadata = {
        "generated_at": datetime.now().isoformat(),
        "model": "openai/clip-vit-base-patch32",
        "embedding_dim": 512,
        "total_images": len(paths),
        "corrupt_files": len(corrupt_files),
        "batch_size": BATCH_SIZE,
        "normalised": True  # L2 normalised for cosine similarity
    }
    
    metadata_file = output_path / "metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"   ✓ Metadata: {metadata_file}")
    
    print("\n" + "=" * 60)
    print(" ✓ PIPELINE COMPLETE!")
    print("=" * 60)
    print(f"   Output directory: {output_path}")
    print(f"   Total vectors: {len(embeddings):,}")
    print("\n   Next steps:")
    print("   1. Run UMAP reduction: python src/reduce_dimensions.py")
    print("   2. Launch Dash frontend: python src/dashboard.py")


# --- MAIN EXECUTION ---
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print(" CRISISD EMBEDDING GENERATOR")
    print(" Visualising Natural Disaster Image Embeddings")
    print("=" * 60)
    print(f" Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 1. Load the model
    model, processor, device = load_model()
    
    # 2. Find all images
    all_files = find_images(DATASET_PATH)
    
    if len(all_files) == 0:
        print("\n ✗ ERROR: No images found!")
        print(f"   Check that DATASET_PATH exists: {DATASET_PATH}")
        print("   Expected structure: CrisisMMD_v2.0/data_image/<event>/<images>")
        sys.exit(1)

    # 3. Generate embeddings
    embeddings, paths, corrupt = process_images(model, processor, device, all_files)

    # 4. Save results
    if embeddings is not None:
        save_results(embeddings, paths, corrupt, OUTPUT_DIR)
    else:
        print("\n ✗ Failed to generate embeddings. Check error messages above.")
        sys.exit(1)
