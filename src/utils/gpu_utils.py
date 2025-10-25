"""
GPU Utilities
Helper functions for GPU memory management and device handling.
"""

import torch
from typing import Optional, Dict
from config.logging_config import get_logger

logger = get_logger(__name__)


def get_device(device: Optional[str] = None) -> torch.device:
    """
    Get the appropriate PyTorch device (CPU or CUDA).
    
    Args:
        device: Specific device string ('cuda', 'cuda:0', 'cpu', etc.)
                If None, automatically selects CUDA if available
    
    Returns:
        PyTorch device object
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    device_obj = torch.device(device)
    logger.info(f"Using device: {device_obj}")
    
    return device_obj


def get_gpu_info() -> Dict[str, any]:
    """
    Get detailed information about available GPUs.
    
    Returns:
        Dictionary containing GPU information
    """
    info = {
        'cuda_available': torch.cuda.is_available(),
        'device_count': 0,
        'current_device': None,
        'devices': []
    }
    
    if torch.cuda.is_available():
        info['device_count'] = torch.cuda.device_count()
        info['current_device'] = torch.cuda.current_device()
        
        for i in range(info['device_count']):
            device_props = torch.cuda.get_device_properties(i)
            device_info = {
                'id': i,
                'name': device_props.name,
                'total_memory_gb': device_props.total_memory / (1024**3),
                'major': device_props.major,
                'minor': device_props.minor,
                'multi_processor_count': device_props.multi_processor_count
            }
            info['devices'].append(device_info)
    
    return info


def print_gpu_info():
    """Print GPU information in a formatted way."""
    info = get_gpu_info()
    
    print("="*60)
    print("GPU INFORMATION")
    print("="*60)
    
    if info['cuda_available']:
        print(f"CUDA Available: Yes")
        print(f"Number of GPUs: {info['device_count']}")
        print(f"Current Device: {info['current_device']}")
        print("-"*60)
        
        for device in info['devices']:
            print(f"\nGPU {device['id']}: {device['name']}")
            print(f"  Memory: {device['total_memory_gb']:.2f} GB")
            print(f"  Compute Capability: {device['major']}.{device['minor']}")
            print(f"  Multiprocessors: {device['multi_processor_count']}")
    else:
        print("CUDA Available: No")
        print("Using CPU for computations")
    
    print("="*60)


def get_gpu_memory_info(device: Optional[int] = None) -> Dict[str, float]:
    """
    Get current GPU memory usage.
    
    Args:
        device: GPU device ID (None for current device)
    
    Returns:
        Dictionary with memory information in GB
    """
    if not torch.cuda.is_available():
        return {'allocated': 0.0, 'reserved': 0.0, 'free': 0.0}
    
    if device is None:
        device = torch.cuda.current_device()
    
    allocated = torch.cuda.memory_allocated(device) / (1024**3)
    reserved = torch.cuda.memory_reserved(device) / (1024**3)
    total = torch.cuda.get_device_properties(device).total_memory / (1024**3)
    free = total - allocated
    
    return {
        'allocated_gb': allocated,
        'reserved_gb': reserved,
        'total_gb': total,
        'free_gb': free,
        'utilization_percent': (allocated / total) * 100
    }


def clear_gpu_memory():
    """Clear GPU cache to free up memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("GPU cache cleared")


def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # For deterministic behavior (may reduce performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    logger.info(f"Random seed set to {seed}")


def estimate_batch_size(
    model_memory_gb: float,
    available_memory_gb: float,
    safety_factor: float = 0.8
) -> int:
    """
    Estimate appropriate batch size based on available GPU memory.
    
    Args:
        model_memory_gb: Memory used by model in GB
        available_memory_gb: Total available GPU memory in GB
        safety_factor: Safety factor (0.8 = use 80% of available memory)
    
    Returns:
        Recommended batch size
    """
    usable_memory = available_memory_gb * safety_factor
    memory_per_sample = 0.1  # Rough estimate: 100MB per image
    
    batch_size = int((usable_memory - model_memory_gb) / memory_per_sample)
    batch_size = max(1, batch_size)  # Ensure at least 1
    
    # Round to nearest power of 2 for efficiency
    batch_size = 2 ** int(torch.log2(torch.tensor(batch_size)))
    
    logger.info(f"Recommended batch size: {batch_size}")
    return batch_size


def monitor_gpu_memory(func):
    """
    Decorator to monitor GPU memory usage of a function.
    
    Usage:
        @monitor_gpu_memory
        def my_function():
            ...
    """
    def wrapper(*args, **kwargs):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            before = get_gpu_memory_info()
            logger.info(f"GPU memory before {func.__name__}: {before['allocated_gb']:.2f} GB")
        
        result = func(*args, **kwargs)
        
        if torch.cuda.is_available():
            after = get_gpu_memory_info()
            peak = torch.cuda.max_memory_allocated() / (1024**3)
            logger.info(f"GPU memory after {func.__name__}: {after['allocated_gb']:.2f} GB")
            logger.info(f"Peak GPU memory during {func.__name__}: {peak:.2f} GB")
        
        return result
    
    return wrapper


def get_optimal_num_workers() -> int:
    """
    Get optimal number of DataLoader workers based on CPU cores.
    
    Returns:
        Recommended number of workers
    """
    import os
    
    num_cpus = os.cpu_count() or 1
    # Use half the CPUs, but at least 1 and at most 8
    num_workers = min(max(num_cpus // 2, 1), 8)
    
    logger.info(f"Recommended num_workers: {num_workers}")
    return num_workers


# Example usage
if __name__ == "__main__":
    print_gpu_info()
    
    if torch.cuda.is_available():
        print("\n" + "="*60)
        print("MEMORY INFORMATION")
        print("="*60)
        
        mem_info = get_gpu_memory_info()
        print(f"Allocated: {mem_info['allocated_gb']:.2f} GB")
        print(f"Reserved: {mem_info['reserved_gb']:.2f} GB")
        print(f"Free: {mem_info['free_gb']:.2f} GB")
        print(f"Total: {mem_info['total_gb']:.2f} GB")
        print(f"Utilization: {mem_info['utilization_percent']:.1f}%")
        
        print("\n" + "="*60)
        print("RECOMMENDATIONS")
        print("="*60)
        
        batch_size = estimate_batch_size(2.0, mem_info['total_gb'])
        print(f"Recommended batch size: {batch_size}")
        
        num_workers = get_optimal_num_workers()
        print(f"Recommended num_workers: {num_workers}")
    
    # Test seed setting
    print("\n" + "="*60)
    set_seed(42)
    print("Random seed set for reproducibility")