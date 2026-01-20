"""
Data Cleaning Pipeline for CrisisMMD Dataset
Filters out corrupt, tiny, and irrelevant images before vectorisation.

Run this BEFORE vectorise.py to ensure high-quality embeddings.

Author: Rashid
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

# --- CONFIGURATION ---
# Input: Raw CrisisMMD data
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "CrisisMMD_v2.0" / "data_image"

# Output: Cleaned data ready for vectorisation
CLEAN_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "clean_data"

# --- QUALITY FILTERS ---
# Minimum resolution (pixels) - filters out tiny thumbnails and icons
MIN_WIDTH = 200
MIN_HEIGHT = 200

# Maximum aspect ratio - filters out banners and extreme shapes
# Aspect ratio = width / height
MAX_ASPECT_RATIO = 4.0   # Reject if width > 4x height
MIN_ASPECT_RATIO = 0.25  # Reject if height > 4x width

# Minimum file size (bytes) - filters out placeholder/stub images
MIN_FILE_SIZE = 5000  # 5KB minimum


def setup_folders():
    """Create the clean data directory if it doesn't exist."""
    if not CLEAN_DATA_PATH.exists():
        CLEAN_DATA_PATH.mkdir(parents=True, exist_ok=True)
        print(f"   ✓ Created output folder: {CLEAN_DATA_PATH}")
    return True


def is_valid_image(file_path):
    """
    Validates an image file for quality and usability.
    
    Checks:
    1. File is not corrupt (can be opened and verified)
    2. Resolution meets minimum requirements
    3. Aspect ratio is reasonable (not a banner/sidebar)
    4. File size is above minimum threshold
    
    Args:
        file_path: Path to the image file
        
    Returns:
        tuple: (is_valid: bool, reason: str)
    """
    path = Path(file_path)
    
    # Check 1: File size
    try:
        file_size = path.stat().st_size
        if file_size < MIN_FILE_SIZE:
            return False, f"Too small ({file_size} bytes)"
    except Exception:
        return False, "Cannot read file"
    
    # Check 2: Can open and verify (corruption check)
    try:
        with Image.open(file_path) as img:
            img.verify()
    except Exception as e:
        return False, f"Corrupt ({type(e).__name__})"
    
    # Check 3: Resolution and aspect ratio (need to reopen after verify)
    try:
        with Image.open(file_path) as img:
            width, height = img.size
            
            # Resolution check
            if width < MIN_WIDTH or height < MIN_HEIGHT:
                return False, f"Low resolution ({width}x{height})"
            
            # Aspect ratio check
            aspect_ratio = width / height
            if aspect_ratio > MAX_ASPECT_RATIO:
                return False, f"Too wide (ratio: {aspect_ratio:.1f})"
            if aspect_ratio < MIN_ASPECT_RATIO:
                return False, f"Too tall (ratio: {aspect_ratio:.1f})"
            
            # Check for grayscale or unusual modes
            # (optional - uncomment if you want only RGB images)
            # if img.mode not in ('RGB', 'RGBA'):
            #     return False, f"Unusual mode ({img.mode})"
                
    except Exception as e:
        return False, f"Cannot read dimensions ({type(e).__name__})"
    
    return True, "Valid"


def clean_dataset():
    """
    Main cleaning function. Scans raw data, validates each image,
    and copies valid images to the clean data folder.
    
    Preserves the original folder structure (event categories).
    """
    print("\n" + "=" * 60)
    print(" CRISISD DATA CLEANING PIPELINE")
    print("=" * 60)
    print(f" Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Setup
    if not RAW_DATA_PATH.exists():
        print(f"\n ✗ ERROR: Raw data path does not exist!")
        print(f"   Path: {RAW_DATA_PATH}")
        print("   Please download CrisisMMD dataset first.")
        return
    
    setup_folders()
    
    print(f"\n [1/3] Scanning for images...")
    print(f"   > Source: {RAW_DATA_PATH}")
    print(f"   > Destination: {CLEAN_DATA_PATH}")
    
    # Find all images
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
        image_files.extend(RAW_DATA_PATH.rglob(ext))
    
    total_images = len(image_files)
    print(f"   ✓ Found {total_images:,} images to process")
    
    if total_images == 0:
        print("\n ✗ No images found!")
        return
    
    # Process each image
    print(f"\n [2/3] Validating and copying images...")
    
    stats = {
        'valid': 0,
        'skipped': defaultdict(int),  # Reason -> count
        'by_category': defaultdict(lambda: {'valid': 0, 'skipped': 0})
    }
    
    errors_log = []
    
    for src_path in tqdm(image_files, desc="   Processing", unit="img"):
        # Get the relative path to preserve folder structure
        rel_path = src_path.relative_to(RAW_DATA_PATH)
        
        # Extract category (first folder in path)
        category = rel_path.parts[0] if len(rel_path.parts) > 1 else "uncategorised"
        
        # Validate image
        is_valid, reason = is_valid_image(src_path)
        
        if is_valid:
            # Copy to clean folder, preserving structure
            dest_path = CLEAN_DATA_PATH / rel_path
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            
            try:
                shutil.copy2(src_path, dest_path)
                stats['valid'] += 1
                stats['by_category'][category]['valid'] += 1
            except Exception as e:
                stats['skipped'][f"Copy error: {type(e).__name__}"] += 1
                stats['by_category'][category]['skipped'] += 1
                errors_log.append(f"{src_path}\tCopy error: {e}")
        else:
            stats['skipped'][reason] += 1
            stats['by_category'][category]['skipped'] += 1
            errors_log.append(f"{src_path}\t{reason}")
    
    # Print report
    print("\n" + "=" * 60)
    print(" [3/3] CLEANING REPORT")
    print("=" * 60)
    
    total_skipped = sum(stats['skipped'].values())
    
    print(f"\n   SUMMARY")
    print(f"   " + "-" * 40)
    print(f"   Total processed:  {total_images:,}")
    print(f"   ✓ Valid images:   {stats['valid']:,} ({100*stats['valid']/total_images:.1f}%)")
    print(f"   ✗ Skipped:        {total_skipped:,} ({100*total_skipped/total_images:.1f}%)")
    
    print(f"\n   REJECTION REASONS")
    print(f"   " + "-" * 40)
    for reason, count in sorted(stats['skipped'].items(), key=lambda x: -x[1]):
        print(f"   {reason}: {count:,}")
    
    print(f"\n   BY CATEGORY (Top 10)")
    print(f"   " + "-" * 40)
    sorted_cats = sorted(
        stats['by_category'].items(), 
        key=lambda x: x[1]['valid'], 
        reverse=True
    )[:10]
    for cat, counts in sorted_cats:
        total = counts['valid'] + counts['skipped']
        pct = 100 * counts['valid'] / total if total > 0 else 0
        print(f"   {cat}: {counts['valid']:,}/{total:,} valid ({pct:.0f}%)")
    
    # Save logs
    log_dir = PROJECT_ROOT / "data" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Detailed error log
    error_log_path = log_dir / "cleaning_errors.log"
    with open(error_log_path, 'w', encoding='utf-8') as f:
        for line in errors_log:
            f.write(line + "\n")
    
    # Summary JSON
    summary = {
        "timestamp": datetime.now().isoformat(),
        "source": str(RAW_DATA_PATH),
        "destination": str(CLEAN_DATA_PATH),
        "total_processed": total_images,
        "valid": stats['valid'],
        "skipped": total_skipped,
        "rejection_reasons": dict(stats['skipped']),
        "settings": {
            "min_resolution": f"{MIN_WIDTH}x{MIN_HEIGHT}",
            "min_file_size": MIN_FILE_SIZE,
            "aspect_ratio_range": f"{MIN_ASPECT_RATIO}-{MAX_ASPECT_RATIO}"
        }
    }
    
    summary_path = log_dir / "cleaning_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n   LOGS SAVED")
    print(f"   " + "-" * 40)
    print(f"   Error log: {error_log_path}")
    print(f"   Summary: {summary_path}")
    
    print("\n" + "=" * 60)
    print(" ✓ CLEANING COMPLETE!")
    print("=" * 60)
    print(f"   Clean data ready at: {CLEAN_DATA_PATH}")
    print(f"\n   Next step:")
    print(f"   Update DATASET_PATH in vectorise.py to point to clean_data,")
    print(f"   then run: python src/vectorise.py")


if __name__ == "__main__":
    clean_dataset()
