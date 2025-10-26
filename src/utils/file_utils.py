"""
File Utilities
Helper functions for file operations, path handling, and data management.
"""

import shutil
from pathlib import Path
from typing import List, Optional, Union
import json
from config.logging_config import get_logger

logger = get_logger(__name__)


def ensure_dir(path: Union[str, Path]) -> Path:
    """
    Ensure a directory exists, create it if it doesn't.
    
    Args:
        path: Directory path
    
    Returns:
        Path object of the directory
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_image_files(
    directory: Union[str, Path],
    extensions: Optional[List[str]] = None,
    recursive: bool = True
) -> List[Path]:
    """
    Get all image files from a directory.
    
    Args:
        directory: Directory to search
        extensions: List of file extensions (e.g., ['.jpg', '.png'])
        recursive: Whether to search subdirectories
    
    Returns:
        List of image file paths
    """
    directory = Path(directory)
    
    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff']
    
    # Normalise extensions (ensure lowercase and leading dot)
    extensions = [ext.lower() if ext.startswith('.') else f'.{ext.lower()}' 
                  for ext in extensions]
    
    image_files = []
    
    if recursive:
        for ext in extensions:
            image_files.extend(directory.rglob(f'*{ext}'))
    else:
        for ext in extensions:
            image_files.extend(directory.glob(f'*{ext}'))
    
    logger.info(f"Found {len(image_files)} image files in {directory}")
    return sorted(image_files)


def get_file_size_mb(file_path: Union[str, Path]) -> float:
    """
    Get file size in megabytes.
    
    Args:
        file_path: Path to the file
    
    Returns:
        File size in MB
    """
    return Path(file_path).stat().st_size / (1024 * 1024)


def filter_by_size(
    files: List[Path],
    max_size_mb: Optional[float] = None,
    min_size_mb: Optional[float] = None
) -> List[Path]:
    """
    Filter files by size.
    
    Args:
        files: List of file paths
        max_size_mb: Maximum file size in MB
        min_size_mb: Minimum file size in MB
    
    Returns:
        Filtered list of file paths
    """
    filtered = []
    
    for file in files:
        size_mb = get_file_size_mb(file)
        
        if max_size_mb is not None and size_mb > max_size_mb:
            logger.debug(f"Skipping {file.name}: too large ({size_mb:.2f} MB)")
            continue
        
        if min_size_mb is not None and size_mb < min_size_mb:
            logger.debug(f"Skipping {file.name}: too small ({size_mb:.2f} MB)")
            continue
        
        filtered.append(file)
    
    logger.info(f"Filtered {len(files)} files to {len(filtered)} files")
    return filtered


def save_json(data: dict, file_path: Union[str, Path], indent: int = 2):
    """
    Save dictionary to JSON file.
    
    Args:
        data: Dictionary to save
        file_path: Output file path
        indent: JSON indentation level
    """
    file_path = Path(file_path)
    ensure_dir(file_path.parent)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)
    
    logger.info(f"Saved JSON to {file_path}")


def load_json(file_path: Union[str, Path]) -> dict:
    """
    Load JSON file.
    
    Args:
        file_path: Path to JSON file
    
    Returns:
        Loaded dictionary
    """
    file_path = Path(file_path)
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    logger.info(f"Loaded JSON from {file_path}")
    return data


def copy_file(src: Union[str, Path], dst: Union[str, Path]):
    """
    Copy a file from source to destination.
    
    Args:
        src: Source file path
        dst: Destination file path
    """
    src = Path(src)
    dst = Path(dst)
    ensure_dir(dst.parent)
    shutil.copy2(src, dst)
    logger.debug(f"Copied {src.name} to {dst}")


def move_file(src: Union[str, Path], dst: Union[str, Path]):
    """
    Move a file from source to destination.
    
    Args:
        src: Source file path
        dst: Destination file path
    """
    src = Path(src)
    dst = Path(dst)
    ensure_dir(dst.parent)
    shutil.move(str(src), str(dst))
    logger.debug(f"Moved {src.name} to {dst}")


def get_relative_path(file_path: Path, base_path: Path) -> Path:
    """
    Get relative path from base path.
    
    Args:
        file_path: Full file path
        base_path: Base directory path
    
    Returns:
        Relative path
    """
    try:
        return file_path.relative_to(base_path)
    except ValueError:
        # If not relative, return the original path
        return file_path


def clean_filename(filename: str) -> str:
    """
    Clean a filename by removing/replacing invalid characters.
    
    Args:
        filename: Original filename
    
    Returns:
        Cleaned filename
    """
    # Replace invalid characters
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    
    # Remove leading/trailing spaces and dots
    filename = filename.strip(' .')
    
    return filename


def get_disk_usage(path: Union[str, Path]) -> dict:
    """
    Get disk usage statistics for a path.
    
    Args:
        path: Directory or file path
    
    Returns:
        Dictionary with total, used, and free space in GB
    """
    path = Path(path)
    stat = shutil.disk_usage(path)
    
    return {
        'total_gb': stat.total / (1024**3),
        'used_gb': stat.used / (1024**3),
        'free_gb': stat.free / (1024**3),
        'percent_used': (stat.used / stat.total) * 100
    }


# Example usage
if __name__ == "__main__":
    from config.settings import config
    
    # Test directory creation
    test_dir = config.DATA_DIR / "test"
    ensure_dir(test_dir)
    print(f"Created directory: {test_dir}")
    
    # Test disk usage
    usage = get_disk_usage(config.PROJECT_ROOT)
    print(f"\nDisk usage:")
    print(f"Total: {usage['total_gb']:.2f} GB")
    print(f"Used: {usage['used_gb']:.2f} GB")
    print(f"Free: {usage['free_gb']:.2f} GB")
    print(f"Usage: {usage['percent_used']:.1f}%")
    
    # Clean up test directory
    test_dir.rmdir()
    print(f"\nRemoved test directory")