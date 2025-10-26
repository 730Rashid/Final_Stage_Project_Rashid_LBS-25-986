"""
Installation Verification Script
Checks that all critical packages are installed and working correctly.
Also verifies GPU availability and provides system information.

Usage:
    python verify_install.py
"""

import sys
from typing import List, Tuple


def check_import(package_name: str, import_name: str = None) -> bool:
    """
    Try importing a package and report status.
    
    Args:
        package_name: Display name of the package
        import_name: Actual import name (if different from package_name)
    
    Returns:
        True if import successful, False otherwise
    """
    if import_name is None:
        import_name = package_name.lower().replace('-', '_')
    
    try:
        __import__(import_name)
        print(f"✓ {package_name}")
        return True
    except ImportError as e:
        print(f"✗ {package_name}: {e}")
        return False
    except Exception as e:
        print(f"✗ {package_name}: Unexpected error - {e}")
        return False


def check_gpu() -> bool:
    """
    Check if PyTorch can access the GPU.
    
    Returns:
        True if GPU available, False otherwise
    """
    try:
        import torch
        
        print("\n" + "="*70)
        print("GPU INFORMATION")
        print("="*70)
        
        if torch.cuda.is_available():
            print(f"✓ CUDA Available: Yes")
            print(f"  Device Count: {torch.cuda.device_count()}")
            print(f"  Current Device: {torch.cuda.current_device()}")
            print(f"  Device Name: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA Version: {torch.version.cuda}")
            
            # Get device properties
            device_props = torch.cuda.get_device_properties(0)
            print(f"  Total Memory: {device_props.total_memory / (1024**3):.2f} GB")
            print(f"  Compute Capability: {device_props.major}.{device_props.minor}")
            print(f"  Multiprocessors: {device_props.multi_processor_count}")
            
            # Test GPU with a simple operation
            try:
                x = torch.randn(100, 100).cuda()
                y = x @ x
                del x, y
                torch.cuda.empty_cache()
                print(f"  GPU Test: Simple operations working")
            except Exception as e:
                print(f"  GPU Test: Error during test - {e}")
                return False
            
            return True
        else:
            print("CUDA Available: No")
            print("PyTorch will use CPU for computations")
            print("This is normal if you don't have a compatible GPU otherwise please check your hardware and drivers")
            return False
            
    except ImportError:
        print("\n PyTorch not installed - cannot check for GPU")
        return False
    except Exception as e:
        print(f"\n GPU check failed: {e}")
        return False


def check_clip() -> bool:
    """
    Specifically check CLIP installation and load a model.
    
    Returns:
        True if CLIP works, False otherwise
    """
    try:
        import clip
        import torch
        
        print("\n" + "="*70)
        print("CLIP MODEL CHECK")
        print("="*70)
        
        # List available models
        available_models = clip.available_models()
        print(f"Available CLIP models: {', '.join(available_models)}")
        
        # Try loading the default model
        print("Loading CLIP ViT-B/32 model...")
        model, preprocess = clip.load("ViT-B/32", device="cpu")
        print("✓ CLIP model loaded successfully")
        
        # Clean up
        del model, preprocess
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return True
        
    except ImportError:
        print("\n CLIP not installed")
        print("  Install with: pip install git+https://github.com/openai/CLIP.git")
        return False
    except Exception as e:
        print(f"\n CLIP check failed: {e}")
        return False


def get_system_info() -> dict:
    """
    Get system information.
    
    Returns:
        Dictionary with system information
    """
    import platform
    import os
    
    info = {
        'python_version': platform.python_version(),
        'platform': platform.platform(),
        'processor': platform.processor(),
        'cpu_count': os.cpu_count(),
    }
    
    return info


def print_system_info():
    """Print system information."""
    info = get_system_info()
    
    print("\n" + "="*70)
    print("SYSTEM INFORMATION")
    print("="*70)
    print(f"Python Version: {info['python_version']}")
    print(f"Platform: {info['platform']}")
    print(f"Processor: {info['processor']}")
    print(f"CPU Count: {info['cpu_count']}")


def check_package_versions():
    """Check and print versions of key packages."""
    print("\n" + "="*70)
    print("PACKAGE VERSIONS")
    print("="*70)
    
    packages_to_check = [
        ('torch', 'PyTorch'),
        ('torchvision', 'TorchVision'),
        ('transformers', 'Transformers'),
        ('numpy', 'NumPy'),
        ('pandas', 'Pandas'),
        ('sklearn', 'Scikit-learn'),
        ('umap', 'UMAP'),
        ('hdbscan', 'HDBSCAN'),
        ('PIL', 'Pillow'),
        ('cv2', 'OpenCV'),
        ('h5py', 'H5Py'),
        ('plotly', 'Plotly'),
        ('dash', 'Dash'),
        ('streamlit', 'Streamlit'),
    ]
    
    for import_name, display_name in packages_to_check:
        try:
            module = __import__(import_name)
            version = getattr(module, '__version__', 'unknown')
            print(f"{display_name:20s} {version}")
        except ImportError:
            print(f"{display_name:20s} NOT INSTALLED")
        except Exception:
            print(f"{display_name:20s} ERROR")


def main():
    """Main verification function."""
    
    print("="*70)
    print("Distaster Visualisation Installation Verification")
    print("="*70)
    print("\nChecking critical packages...\n")
    
    # Define packages to check
    packages: List[Tuple[str, str]] = [
        ("PyTorch", "torch"),
        ("TorchVision", "torchvision"),
        ("Transformers", "transformers"),
        ("CLIP", "clip"),
        ("NumPy", "numpy"),
        ("Pandas", "pandas"),
        ("Scikit-learn", "sklearn"),
        ("UMAP", "umap"),
        ("HDBSCAN", "hdbscan"),
        ("Pillow", "PIL"),
        ("OpenCV", "cv2"),
        ("H5Py", "h5py"),
        ("Plotly", "plotly"),
        ("Dash", "dash"),
        ("Streamlit", "streamlit"),
        ("TQDM", "tqdm"),
        ("Python-dotenv", "dotenv"),
    ]
    
    # Check all packages
    results = [check_import(name, import_name) for name, import_name in packages]
    
    # Check GPU
    gpu_available = check_gpu()
    
    # Check CLIP specifically
    clip_works = check_clip()
    
    # Print system info
    print_system_info()
    
    # Print package versions
    check_package_versions()
    
    # Final summary
    print("\n" + "="*70)
    print("Verification Summary")
    print("="*70)
    print(f"Packages checked: {len(results)}")
    print(f"Successfully installed: {sum(results)}/{len(results)}")
    print(f"GPU available: {'Yes' if gpu_available else 'No (CPU only)'}")
    print(f"CLIP working: {'Yes' if clip_works else 'No'}")
    
    # Overall status
    print("\n" + "="*70)
    if all(results) and clip_works:
        print("All packages installed successfully!")
        print("="*70)
        print("\nYou're ready to proceed to Phase 2!")
        print("\nNext steps:")
        print("1. Test configuration: python -m config.settings")
        print("2. Test GPU utils: python -m src.utils.gpu_utils")
        print("3. Test file utils: python -m src.utils.file_utils")
        print("4. Make your first commit: git add . && git commit -m 'Phase 1 complete'")
        return 0
    else:
        print("Error some files failed to install correctly")
        print("="*70)
        print("\nPlease check the errors above and:")
        print("1. Make sure you activated your virtual environment")
        print("2. Try: pip install -r requirements.txt")
        print("3. For CLIP: pip install git+https://github.com/openai/CLIP.git")
        print("4. For GPU issues: Verify NVIDIA drivers with 'nvidia-smi'")
        return 1


if __name__ == "__main__":
    sys.exit(main())