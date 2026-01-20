"""
GPU Utilities Module.

Helper functions for GPU memory management and device handling.
These utilities help optimise performance on different hardware
configurations, particularly useful for my MX450 GPU with limited VRAM.

Author: Rashid
Supervisor: XinHui Ma
Project: Visualising Natural Disaster Image Embeddings
"""

import torch
from typing import Optional, Dict
from config.logging_config import get_logger


logger = get_logger(__name__)


def get_device(device: Optional[str] = None) -> torch.device:
    """
    Get the appropriate PyTorch device.
    
    This function automatically selects the best available device
    (CUDA GPU, Apple Silicon, or CPU) for running computations.
    
    Args:
        device: Specific device string (e.g. 'cuda', 'cpu').
                If None, automatically selects the best option.
    
    Returns:
        PyTorch device object.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    device_obj = torch.device(device)
    logger.info("Using device: {}".format(device_obj))
    
    return device_obj


def get_gpu_info() -> Dict[str, any]:
    """
    Get detailed information about available GPUs.
    
    Returns:
        Dictionary containing GPU information including name,
        memory, and compute capability.
    """
    info = {
        "cuda_available": torch.cuda.is_available(),
        "device_count": 0,
        "current_device": None,
        "devices": []
    }
    
    if torch.cuda.is_available():
        info["device_count"] = torch.cuda.device_count()
        info["current_device"] = torch.cuda.current_device()
        
        for i in range(info["device_count"]):
            device_props = torch.cuda.get_device_properties(i)
            device_info = {
                "id": i,
                "name": device_props.name,
                "total_memory_gb": device_props.total_memory / (1024 ** 3),
                "major": device_props.major,
                "minor": device_props.minor,
                "multi_processor_count": device_props.multi_processor_count
            }
            info["devices"].append(device_info)
    
    return info


def print_gpu_info():
    """
    Print GPU information in a formatted way.
    
    Useful for debugging and checking hardware configuration.
    """
    info = get_gpu_info()
    
    print("GPU Information:")
    
    if info["cuda_available"]:
        print("CUDA Available: Yes")
        print("Number of GPUs: {}".format(info["device_count"]))
        print("Current Device: {}".format(info["current_device"]))
        
        for device in info["devices"]:
            print("")
            print("GPU {}: {}".format(device["id"], device["name"]))
            print("  Memory: {:.2f} GB".format(device["total_memory_gb"]))
            print("  Compute Capability: {}.{}".format(device["major"], device["minor"]))
            print("  Multiprocessors: {}".format(device["multi_processor_count"]))
    else:
        print("CUDA Available: No")
        print("Using CPU for computations")


def get_gpu_memory_info(device: Optional[int] = None) -> Dict[str, float]:
    """
    Get current GPU memory usage.
    
    Args:
        device: GPU device ID. If None, uses the current device.
    
    Returns:
        Dictionary with memory information in GB.
    """
    if not torch.cuda.is_available():
        return {"allocated": 0.0, "reserved": 0.0, "free": 0.0}
    
    if device is None:
        device = torch.cuda.current_device()
    
    allocated = torch.cuda.memory_allocated(device) / (1024 ** 3)
    reserved = torch.cuda.memory_reserved(device) / (1024 ** 3)
    total = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
    free = total - allocated
    
    return {
        "allocated_gb": allocated,
        "reserved_gb": reserved,
        "total_gb": total,
        "free_gb": free,
        "utilisation_percent": (allocated / total) * 100
    }


def clear_gpu_memory():
    """
    Clear the GPU cache to free up memory.
    
    Useful when processing large batches of images and memory
    starts to accumulate.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("GPU cache cleared")


def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility.
    
    This ensures that results are consistent across runs,
    which is important for academic research.
    
    Args:
        seed: Random seed value.
    """
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Enable deterministic behaviour (may reduce performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    logger.info("Random seed set to {}".format(seed))


def estimate_batch_size(
    model_memory_gb: float,
    available_memory_gb: float,
    safety_factor: float = 0.8
) -> int:
    """
    Estimate an appropriate batch size based on available GPU memory.
    
    This is particularly useful for my MX450 with only 2GB VRAM,
    where choosing the right batch size is critical.
    
    Args:
        model_memory_gb: Memory used by the model in GB.
        available_memory_gb: Total available GPU memory in GB.
        safety_factor: Fraction of memory to use (0.8 = 80%).
    
    Returns:
        Recommended batch size.
    """
    usable_memory = available_memory_gb * safety_factor
    memory_per_sample = 0.1  # Rough estimate: 100MB per image
    
    batch_size = int((usable_memory - model_memory_gb) / memory_per_sample)
    batch_size = max(1, batch_size)
    
    # Round to nearest power of 2 for efficiency
    batch_size = 2 ** int(torch.log2(torch.tensor(batch_size)))
    
    logger.info("Recommended batch size: {}".format(batch_size))
    return batch_size


def monitor_gpu_memory(func):
    """
    Decorator to monitor GPU memory usage of a function.
    
    Useful for profiling and debugging memory issues.
    
    Usage:
        @monitor_gpu_memory
        def my_function():
            ...
    """
    def wrapper(*args, **kwargs):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            before = get_gpu_memory_info()
            logger.info("GPU memory before {}: {:.2f} GB".format(
                func.__name__, before["allocated_gb"]
            ))
        
        result = func(*args, **kwargs)
        
        if torch.cuda.is_available():
            after = get_gpu_memory_info()
            peak = torch.cuda.max_memory_allocated() / (1024 ** 3)
            logger.info("GPU memory after {}: {:.2f} GB".format(
                func.__name__, after["allocated_gb"]
            ))
            logger.info("Peak GPU memory during {}: {:.2f} GB".format(
                func.__name__, peak
            ))
        
        return result
    
    return wrapper


def get_optimal_num_workers() -> int:
    """
    Get the optimal number of DataLoader workers based on CPU cores.
    
    Returns:
        Recommended number of workers.
    """
    import os
    
    num_cpus = os.cpu_count() or 1
    # Use half the CPUs, but at least 1 and at most 8
    num_workers = min(max(num_cpus // 2, 1), 8)
    
    logger.info("Recommended num_workers: {}".format(num_workers))
    return num_workers


if __name__ == "__main__":
    print_gpu_info()
    
    if torch.cuda.is_available():
        print("")
        print("Memory Information:")
        
        mem_info = get_gpu_memory_info()
        print("Allocated: {:.2f} GB".format(mem_info["allocated_gb"]))
        print("Reserved: {:.2f} GB".format(mem_info["reserved_gb"]))
        print("Free: {:.2f} GB".format(mem_info["free_gb"]))
        print("Total: {:.2f} GB".format(mem_info["total_gb"]))
        print("Utilisation: {:.1f}%".format(mem_info["utilisation_percent"]))
        
        print("")
        print("Recommendations:")
        
        batch_size = estimate_batch_size(2.0, mem_info["total_gb"])
        print("Recommended batch size: {}".format(batch_size))
        
        num_workers = get_optimal_num_workers()
        print("Recommended num_workers: {}".format(num_workers))
    
    print("")
    set_seed(42)
    print("Random seed set for reproducibility")
