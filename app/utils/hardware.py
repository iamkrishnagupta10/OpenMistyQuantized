"""
Hardware detection and optimization utilities.
"""

import os
import platform
import psutil
import torch
from typing import Dict, Any

from app.logger import logger
from app.config import config


def check_hardware_compatibility() -> Dict[str, Any]:
    """
    Check hardware compatibility and return information about the system.
    
    Returns:
        Dict[str, Any]: Hardware information
    """
    # Get system information
    system = platform.system()
    processor = platform.processor()
    machine = platform.machine()
    
    # Get memory information
    memory = psutil.virtual_memory()
    total_memory_gb = memory.total / (1024 ** 3)
    available_memory_gb = memory.available / (1024 ** 3)
    
    # Get CPU information
    cpu_count = psutil.cpu_count(logical=False)
    cpu_count_logical = psutil.cpu_count(logical=True)
    
    # Check for CUDA availability
    cuda_available = torch.cuda.is_available()
    cuda_device_count = torch.cuda.device_count() if cuda_available else 0
    cuda_device_name = torch.cuda.get_device_name(0) if cuda_available and cuda_device_count > 0 else "N/A"
    
    # Check for MPS (Metal Performance Shaders) availability on macOS
    mps_available = False
    if system == "Darwin" and hasattr(torch, 'mps') and hasattr(torch.mps, 'is_available'):
        mps_available = torch.mps.is_available()
    
    # Determine the best device for inference
    if cuda_available:
        device = "cuda"
        device_name = cuda_device_name
    elif mps_available:
        device = "mps"
        device_name = f"Apple {machine} (MPS)"
    else:
        device = "cpu"
        device_name = processor
    
    # Apply hardware-specific optimizations
    if config.hardware.optimize_for_hardware:
        # Set environment variables for better performance
        if device == "cpu":
            # Set number of threads for CPU inference
            cpu_threads = config.hardware.cpu_threads
            if cpu_threads == 0:
                # Auto-detect: use physical cores or half of logical cores, whichever is higher
                cpu_threads = max(cpu_count or 1, (cpu_count_logical or 2) // 2)
            
            os.environ["OMP_NUM_THREADS"] = str(cpu_threads)
            os.environ["MKL_NUM_THREADS"] = str(cpu_threads)
            torch.set_num_threads(cpu_threads)
            logger.info(f"Set CPU threads to {cpu_threads}")
        
        # Set memory limits
        max_memory_gb = config.hardware.max_memory_gb
        if max_memory_gb == 0:
            # Auto-detect: use 80% of available memory
            max_memory_gb = available_memory_gb * 0.8
        
        # Log hardware information
        logger.info(f"System: {system} {machine}")
        logger.info(f"Processor: {processor}")
        logger.info(f"Memory: {total_memory_gb:.2f} GB total, {available_memory_gb:.2f} GB available")
        logger.info(f"CPU: {cpu_count} physical cores, {cpu_count_logical} logical cores")
        logger.info(f"CUDA available: {cuda_available}")
        if cuda_available:
            logger.info(f"CUDA devices: {cuda_device_count}")
            logger.info(f"CUDA device: {cuda_device_name}")
        logger.info(f"MPS available: {mps_available}")
        logger.info(f"Using device: {device}")
    
    # Return hardware information
    return {
        "system": system,
        "processor": processor,
        "machine": machine,
        "total_memory_gb": total_memory_gb,
        "available_memory_gb": available_memory_gb,
        "cpu_count": cpu_count,
        "cpu_count_logical": cpu_count_logical,
        "cuda_available": cuda_available,
        "cuda_device_count": cuda_device_count,
        "cuda_device_name": cuda_device_name,
        "mps_available": mps_available,
        "device": device,
        "device_name": device_name,
    }


def get_optimal_batch_size(model_size_gb: float) -> int:
    """
    Calculate the optimal batch size based on available memory and model size.
    
    Args:
        model_size_gb (float): Size of the model in GB
        
    Returns:
        int: Optimal batch size
    """
    memory = psutil.virtual_memory()
    available_memory_gb = memory.available / (1024 ** 3)
    
    # Reserve memory for the model and system
    usable_memory_gb = available_memory_gb - model_size_gb - 2.0
    
    # Estimate memory needed per batch item (this is a rough estimate)
    memory_per_batch_gb = 0.5
    
    # Calculate batch size
    batch_size = max(1, int(usable_memory_gb / memory_per_batch_gb))
    
    return min(batch_size, 32)  # Cap at 32 to avoid excessive memory usage 