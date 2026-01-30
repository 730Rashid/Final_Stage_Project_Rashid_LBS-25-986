"""
Pre-Flight Environment Verification Script.

Run this script before presenting to verify that all dependencies
are correctly installed and the GPU/CUDA environment is configured.

Usage:
    python verify_install.py

Author: Rashid
Supervisor: XinHui Ma
Project: Visualising Natural Disaster Image Embeddings
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


def print_header():
    """Print the verification header."""
    print("")
    print("=" * 60)
    print("  DISASTER VISUALISATION - ENVIRONMENT VERIFICATION")
    print("=" * 60)
    print("")


def check_python_version():
    """Check Python version is 3.9+."""
    print("1. Python Version")
    version = sys.version_info
    version_str = "{}.{}.{}".format(version.major, version.minor, version.micro)
    
    if version.major >= 3 and version.minor >= 9:
        print("   ✅ Python {} (Required: 3.9+)".format(version_str))
        return True
    else:
        print("   ❌ Python {} (Required: 3.9+)".format(version_str))
        return False


def check_torch():
    """Check PyTorch installation and CUDA availability."""
    print("\n2. PyTorch & CUDA")
    try:
        import torch
        print("   ✅ PyTorch {}".format(torch.__version__))
        
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print("   ✅ CUDA Available: {} ({:.1f} GB VRAM)".format(device_name, vram))
            return True
        else:
            print("   ⚠️  CUDA Not Available (Using CPU - slower inference)")
            return True  # Still valid, just slower
    except ImportError:
        print("   ❌ PyTorch not installed")
        return False


def check_transformers():
    """Check Hugging Face Transformers for CLIP."""
    print("\n3. CLIP Model (Transformers)")
    try:
        from transformers import CLIPModel, CLIPProcessor
        print("   ✅ transformers library installed")
        
        # Check if model can be loaded (cached or downloadable)
        try:
            processor = CLIPProcessor.from_pretrained(
                "openai/clip-vit-base-patch32",
                local_files_only=True
            )
            print("   ✅ CLIP model cached locally")
        except Exception:
            print("   ⚠️  CLIP model not cached (will download on first run)")
        
        return True
    except ImportError:
        print("   ❌ transformers not installed")
        return False


def check_umap():
    """Check UMAP installation."""
    print("\n4. UMAP (Dimensionality Reduction)")
    try:
        import umap
        print("   ✅ umap-learn {}".format(umap.__version__))
        return True
    except ImportError:
        print("   ❌ umap-learn not installed")
        return False


def check_sklearn():
    """Check scikit-learn installation."""
    print("\n5. Scikit-learn (Clustering & Metrics)")
    try:
        import sklearn
        print("   ✅ scikit-learn {}".format(sklearn.__version__))
        return True
    except ImportError:
        print("   ❌ scikit-learn not installed")
        return False


def check_dash():
    """Check Dash/Plotly installation."""
    print("\n6. Dash & Plotly (Web App)")
    try:
        import dash
        import plotly
        print("   ✅ Dash {}".format(dash.__version__))
        print("   ✅ Plotly {}".format(plotly.__version__))
        return True
    except ImportError as e:
        print("   ❌ Missing: {}".format(e))
        return False


def check_data_files():
    """Check if required data files exist."""
    print("\n7. Data Files")
    
    files_to_check = [
        ("data/embeddings/embeddings.npy", "CLIP Embeddings"),
        ("data/embeddings/filenames.json", "Image Filenames"),
        ("data/visualisation/umap_coords.npy", "UMAP Coordinates"),
        ("data/visualisation/umap_data.json", "UMAP Data (for app)"),
    ]
    
    all_exist = True
    for rel_path, description in files_to_check:
        full_path = PROJECT_ROOT / rel_path
        if full_path.exists():
            size_mb = full_path.stat().st_size / (1024 * 1024)
            print("   ✅ {} ({:.1f} MB)".format(description, size_mb))
        else:
            print("   ❌ {} (Missing: {})".format(description, rel_path))
            all_exist = False
    
    return all_exist


def check_image_folder():
    """Check if processed images exist."""
    print("\n8. Processed Images")
    
    image_folder = PROJECT_ROOT / "data" / "processed" / "clean_data"
    
    if not image_folder.exists():
        print("   ❌ Clean data folder not found")
        print("      Run: python src/clean_data.py")
        return False
    
    # Count images
    image_count = 0
    for ext in ["*.jpg", "*.jpeg", "*.png"]:
        image_count += len(list(image_folder.rglob(ext)))
    
    if image_count > 0:
        print("   ✅ Found {:,} processed images".format(image_count))
        return True
    else:
        print("   ❌ No images found in clean_data folder")
        return False


def print_summary(results):
    """Print final summary."""
    print("")
    print("=" * 60)
    print("  VERIFICATION SUMMARY")
    print("=" * 60)
    
    passed = sum(results.values())
    total = len(results)
    
    if passed == total:
        print("")
        print("  ✅ ALL CHECKS PASSED ({}/{})".format(passed, total))
        print("")
        print("  Ready to run:")
        print("    python src/app_demo.py")
        print("")
    else:
        print("")
        print("  ⚠️  SOME CHECKS FAILED ({}/{})".format(passed, total))
        print("")
        print("  Failed checks:")
        for name, passed in results.items():
            if not passed:
                print("    ❌ {}".format(name))
        print("")
        print("  Fix the issues above, then run this script again.")
        print("")
    
    print("=" * 60)


def main():
    """Run all verification checks."""
    print_header()
    
    results = {}
    
    results["Python Version"] = check_python_version()
    results["PyTorch & CUDA"] = check_torch()
    results["Transformers CLIP"] = check_transformers()
    results["UMAP"] = check_umap()
    results["Scikit-learn"] = check_sklearn()
    results["Dash & Plotly"] = check_dash()
    results["Data Files"] = check_data_files()
    results["Processed Images"] = check_image_folder()
    
    print_summary(results)
    
    # Return exit code
    if all(results.values()):
        return 0
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main())
