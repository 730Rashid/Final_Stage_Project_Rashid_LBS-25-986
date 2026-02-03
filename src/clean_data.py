"""
Data Cleaning Pipeline for CrisisMMD Dataset.

This script filters out corrupt, tiny, and irrelevant images before
vectorisation. Running this first ensures that only high-quality images
are processed by the embedding pipeline, which improves the quality of
the final visualisation.

Author: Rashid
Supervisor: XinHui Ma
Project: Visualising Natural Disaster Image Embeddings
"""

import os
import sys
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import shutil
import json
from datetime import datetime
from collections import defaultdict


# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# Configuration
# Input folder containing raw CrisisMMD images, some of the images may have noise and some unrelated images that has nothing to do with Natural Disasters.
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "CrisisMMD_v2.0" / "data_image"

# Output folder for cleaned images
CLEAN_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "clean_data"

# Quality thresholds
# Images smaller than this will be rejected
MIN_WIDTH = 200
MIN_HEIGHT = 200

# Images with extreme aspect ratios will be rejected
# This filters out banners, sidebars, and other non-standard shapes
MAX_ASPECT_RATIO = 4.0
MIN_ASPECT_RATIO = 0.25

# Minimum file size in bytes
# Very small files are often placeholder images
MIN_FILE_SIZE = 5000


def setup_folders():
    """
    Create the output directory if it does not exist.
    """
    if not CLEAN_DATA_PATH.exists():
        CLEAN_DATA_PATH.mkdir(parents=True, exist_ok=True)
        print("  Created output folder: {}".format(CLEAN_DATA_PATH))


def is_valid_image(file_path):
    """
    Check if an image file meets our quality requirements.
    
    This function performs several checks:
    1. File size is above the minimum threshold
    2. File is not corrupt and can be opened
    3. Resolution is at least MIN_WIDTH x MIN_HEIGHT
    4. Aspect ratio is within acceptable bounds
    
    Args:
        file_path: Path to the image file.
        
    Returns:
        tuple: (is_valid, reason) where is_valid is a boolean and
               reason is a string explaining why it was rejected.
    """
    path = Path(file_path)
    
    # Check file size
    try:
        file_size = path.stat().st_size
        if file_size < MIN_FILE_SIZE:
            return False, "File too small ({} bytes)".format(file_size)
    except Exception:
        return False, "Cannot read file"
    
    # Check if file is corrupt
    try:
        with Image.open(file_path) as img:
            img.verify()
    except Exception as e:
        return False, "Corrupt file ({})".format(type(e).__name__)
    
    # Check resolution and aspect ratio of the image and try to make it compatiable for parsing.

    try:
        with Image.open(file_path) as img:
            width, height = img.size
            
            if width < MIN_WIDTH or height < MIN_HEIGHT:
                return False, "Resolution too low ({}x{})".format(width, height)
            
            aspect_ratio = width / height

            if aspect_ratio > MAX_ASPECT_RATIO:
                return False, "Too wide (ratio: {:.1f})".format(aspect_ratio)
            if aspect_ratio < MIN_ASPECT_RATIO:
                return False, "Too tall (ratio: {:.1f})".format(aspect_ratio)
                
    except Exception as e:
        return False, "Cannot read dimensions ({})".format(type(e).__name__)
    
    return True, "Valid"


def clean_dataset():
    """
    Main cleaning function.
    
    This scans through all images in the raw data folder, validates each one,
    and copies valid images to the clean data folder. The original folder
    structure is preserved so that event categories remain intact.
    """
    print("Data Cleaning Pipeline...")
    print("Timestamp: {}".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))) # We check the time to make sure it is runnning in real time.
    
    # Check input folder exists
    if not RAW_DATA_PATH.exists():
        print("Error: Raw data folder does not exist")
        print("Path: {}".format(RAW_DATA_PATH))
        print("Please download the dataset first")
        
        return
    
    setup_folders()
    
    print("Scanning for images...")
    
    # Find all image files from all formats.
    image_files = []
    
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]:
        image_files.extend(RAW_DATA_PATH.rglob(ext))
    
    total_images = len(image_files)

    print("  Found {:,} images to process".format(total_images))
    print("")
    
    if total_images == 0:
        print("ERROR: No images found in the source folder")
        return
    
    # Process each image
    print("Step 2: Validating images...")
    
    stats = {
        "valid": 0,
        "skipped": defaultdict(int),
        "by_category": defaultdict(lambda: {"valid": 0, "skipped": 0})
    }
    
    errors_log = []
    
    for src_path in tqdm(image_files, desc="  Checking", unit="img"):
        # Get relative path to preserve folder structure
        rel_path = src_path.relative_to(RAW_DATA_PATH)
        
        # Extract category from the folder name
        if len(rel_path.parts) > 1:
            category = rel_path.parts[0]
        else:
            category = "uncategorised"
        
        # Validate the image
        is_valid, reason = is_valid_image(src_path)
        
        if is_valid:
            # Copy to clean folder
            dest_path = CLEAN_DATA_PATH / rel_path
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            
            try:
                shutil.copy2(src_path, dest_path)
                stats["valid"] += 1
                stats["by_category"][category]["valid"] += 1

            except Exception as e:
                stats["skipped"]["Copy error"] += 1
                stats["by_category"][category]["skipped"] += 1
                errors_log.append("{}\tCopy error: {}".format(src_path, e))
        
        else:
            stats["skipped"][reason] += 1
            stats["by_category"][category]["skipped"] += 1
            errors_log.append("{}\t{}".format(src_path, reason))
    
    print("")
    
    # Print the report
    print("Generating cleaning report...")
    
    total_skipped = sum(stats["skipped"].values())
    
    print("")
    print("Summary:")
    print("Total processed:  {:,}".format(total_images))
    print("Valid images:     {:,} ({:.1f}%)".format(
        stats["valid"], 
        100 * stats["valid"] / total_images
    ))
    print("  Skipped:          {:,} ({:.1f}%)".format(
        total_skipped, 
        100 * total_skipped / total_images
    ))
    
    print("")
    print("Rejection Reasons:")
    sorted_reasons = sorted(stats["skipped"].items(), key=lambda x: -x[1])
    for reason, count in sorted_reasons:
        print("  {}: {:,}".format(reason, count))
    
    print("")
    print("Images by Category (Top 10):")
    sorted_cats = sorted(
        stats["by_category"].items(), 
        key=lambda x: x[1]["valid"], 
        reverse=True
    )[:10]
    for cat, counts in sorted_cats:
        total = counts["valid"] + counts["skipped"]
        if total > 0:
            pct = 100 * counts["valid"] / total
            print("  {}: {:,}/{:,} valid ({:.0f}%)".format(cat, counts["valid"], total, pct))
    
    # Save logs
    log_dir = PROJECT_ROOT / "data" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    error_log_path = log_dir / "cleaning_errors.log"
    with open(error_log_path, "w", encoding="utf-8") as f:
        for line in errors_log:
            f.write(line + "\n")
    
    summary = {
        "timestamp": datetime.now().isoformat(),
        "source": str(RAW_DATA_PATH),
        "destination": str(CLEAN_DATA_PATH),
        "total_processed": total_images,
        "valid": stats["valid"],
        "skipped": total_skipped,
        "rejection_reasons": dict(stats["skipped"]),
        "settings": {
            "min_resolution": "{}x{}".format(MIN_WIDTH, MIN_HEIGHT),
            "min_file_size": MIN_FILE_SIZE,
            "aspect_ratio_range": "{}-{}".format(MIN_ASPECT_RATIO, MAX_ASPECT_RATIO)
        }
    }
    
    summary_path = log_dir / "cleaning_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    print("Logs saved to: {}, {}".format(error_log_path, summary_path))
    
    print("")
    print("Cleaning Complete. Output: {}".format(CLEAN_DATA_PATH))
    print("  Clean data saved to: {}".format(CLEAN_DATA_PATH))
    print("")
    print("  Next step:")
    print("  Update DATASET_PATH in vectorise.py to point to the clean_data")
    print("  folder, then run: python src/vectorise.py in the terminal")
    print("")


if __name__ == "__main__":
    clean_dataset()
