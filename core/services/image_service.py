"""
Image service for managing image operations and state.
"""
from typing import Optional
import numpy as np

from utils.memory_utils import MemoryManager, ErrorHandler
from processing.file_manager import load_image


class ImageService:
    """Service for managing image operations and state."""
    
    def __init__(self):
        self.current_image: Optional[np.ndarray] = None
        self.current_image_path: Optional[str] = None
        self.image_processor = None
        self.crop_rotate_ui = None
    
    def load_image(self, file_path: str) -> Optional[np.ndarray]:
        """Load an image from file."""
        try:
            image = load_image(file_path)
            self.current_image = image
            self.current_image_path = file_path
            
            # Clear GPU memory before loading new image
            MemoryManager.clear_cuda_cache()
            
            return image
        except Exception as e:
            error_msg = ErrorHandler.handle_memory_error(e, "image loading")
            print(error_msg)
            return None
    
    def create_image_processor(self, image: np.ndarray):
        """Create an image processor for the given image."""
        from processing.image_processor import ImageProcessor
        self.image_processor = ImageProcessor(image.copy())
        return self.image_processor
    
    def create_crop_rotate_ui(self, image: np.ndarray, processor):
        """Create crop/rotate UI for the given image."""
        from ui.interactions.crop_rotate import CropRotateUI
        self.crop_rotate_ui = CropRotateUI(image, processor)
        return self.crop_rotate_ui
    
    def get_current_image(self) -> Optional[np.ndarray]:
        """Get the current loaded image."""
        return self.current_image
    
    def get_processed_image(self) -> Optional[np.ndarray]:
        """Get the processed image from the image processor."""
        if self.image_processor:
            # ImageProcessor uses apply_all_edits() to get the processed image
            if hasattr(self.image_processor, 'apply_all_edits'):
                return self.image_processor.apply_all_edits()
            elif hasattr(self.image_processor, 'get_processed_image'):
                return self.image_processor.get_processed_image()
            elif hasattr(self.image_processor, 'current'):
                return self.image_processor.current
        elif self.current_image is not None:
            # Return the current image if no processor is available
            return self.current_image
        return None
