"""
Memory management utilities for GPU/CUDA operations.
Centralizes memory cleanup and error handling patterns.
"""
import gc
import torch
from typing import Dict, Any
from contextlib import contextmanager


class MemoryManager:
    """Centralized memory management for GPU operations."""
    
    @staticmethod
    def get_device_info() -> Dict[str, Any]:
        """Get comprehensive device and memory information."""
        if torch.cuda.is_available():
            device_props = torch.cuda.get_device_properties(0)
            allocated = torch.cuda.memory_allocated() / (1024**2)  # MB
            cached = torch.cuda.memory_reserved() / (1024**2)  # MB
            total = device_props.total_memory / (1024**2)  # MB
            free = total - allocated
            
            return {
                'device': 'cuda',
                'device_name': device_props.name,
                'allocated_mb': allocated,
                'cached_mb': cached,
                'free_mb': free,
                'total_mb': total,
                'total_gb': total / 1024
            }
        else:
            return {
                'device': 'cpu',
                'device_name': 'CPU',
                'allocated_mb': 0,
                'cached_mb': 0,
                'free_mb': float('inf'),
                'total_mb': float('inf'),
                'total_gb': float('inf')
            }
    
    @staticmethod
    def clear_cuda_cache() -> None:
        """Clear CUDA cache and run garbage collection."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    @staticmethod
    def cleanup_gpu_memory():
        """Legacy cleanup method for GPU memory - redirects to clear_cuda_cache."""
        MemoryManager.clear_cuda_cache()
    
    @staticmethod
    def select_optimal_device(min_gpu_memory_gb: float = 4.0) -> str:
        """Select the optimal device based on available memory."""
        if not torch.cuda.is_available():
            return "cpu"
        
        try:
            info = MemoryManager.get_device_info()
            if info['total_gb'] < min_gpu_memory_gb:
                return "cpu"
            return "cuda"
        except Exception as e:
            print(f"Error checking GPU memory: {e}, falling back to CPU")
            return "cpu"
    
    @staticmethod
    def get_recommended_image_size(device: str) -> int:
        """Get recommended maximum image size based on available memory."""
        if device == "cpu":
            return 1024  # CPU can handle larger sizes with slower processing
        
        info = MemoryManager.get_device_info()
        gpu_memory_gb = info['total_gb']
        
        if gpu_memory_gb < 4.0:
            return 512
        elif gpu_memory_gb < 6.0:
            return 768
        else:
            return 1024


def setup_memory_optimization():
    """Setup memory optimization for better performance."""
    import os
    
    # Set PyTorch CUDA memory management for better GPU memory handling
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    # Clear cache at startup
    MemoryManager.clear_cuda_cache()

class ErrorHandler:
    """Centralized error handling for memory and GPU operations."""
    
    @staticmethod
    def handle_memory_error(error: Exception, operation: str = "operation") -> str:
        """Handle memory-related errors with appropriate fallback suggestions."""
        error_str = str(error).lower()
        
        if "memory" in error_str or "cuda" in error_str:
            MemoryManager.clear_cuda_cache()
            return f"Memory error during {operation}. Memory cleared, consider using CPU mode or smaller images."
        elif "device" in error_str:
            return f"Device error during {operation}. Try switching to CPU mode."
        else:
            return f"Unexpected error during {operation}: {error}"
    
    @staticmethod
    def safe_gpu_operation(operation, fallback_operation=None, operation_name="GPU operation"):
        """
        Safely execute a GPU operation with automatic fallback.
        
        Args:
            operation: Function to execute on GPU
            fallback_operation: Function to execute if GPU fails (optional)
            operation_name: Name for logging purposes
        """
        try:
            return operation()
        except torch.cuda.OutOfMemoryError as e:
            print(f"CUDA out of memory during {operation_name}: {e}")
            MemoryManager.clear_cuda_cache()
            
            if fallback_operation:
                print(f"Attempting fallback for {operation_name}")
                return fallback_operation()
            else:
                raise
        except Exception as e:
            error_msg = ErrorHandler.handle_memory_error(e, operation_name)
            print(error_msg)
            
            if fallback_operation and ("memory" in str(e).lower() or "cuda" in str(e).lower()):
                print(f"Attempting fallback for {operation_name}")
                return fallback_operation()
            else:
                raise


class ResourceManager:
    """Manages computational resources and their lifecycles."""
    
    def __init__(self):
        self.active_models = {}
        self.memory_threshold = 500  # MB
    
    def register_model(self, name: str, model, device: str):
        """Register a model for resource tracking."""
        self.active_models[name] = {
            'model': model,
            'device': device,
            'temp_cpu_mode': False
        }
    
    def move_to_cpu_if_needed(self, model_name: str) -> bool:
        """Move model to CPU if memory is critically low."""
        if model_name not in self.active_models:
            return False
        
        model_info = self.active_models[model_name]
        if model_info['device'] != 'cuda':
            return False
        
        memory_info = MemoryManager.get_device_info()
        if memory_info['free_mb'] < self.memory_threshold:            
            model_info['model'].to('cpu')
            model_info['temp_cpu_mode'] = True
            MemoryManager.clear_cuda_cache()
            return True
        
        return False
    
    def move_back_to_gpu_if_possible(self, model_name: str) -> bool:
        """Move model back to GPU if sufficient memory is available."""
        if model_name not in self.active_models:
            return False
        
        model_info = self.active_models[model_name]
        if not model_info.get('temp_cpu_mode', False):
            return False
        
        memory_info = MemoryManager.get_device_info()
        if memory_info['free_mb'] > 1000:  # At least 1GB free
            model_info['model'].to('cuda')
            model_info['temp_cpu_mode'] = False
            return True
        
        return False
