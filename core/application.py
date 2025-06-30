"""
Core application service layer.
Centralizes business logic and coordinates between different subsystems.
"""
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from utils.memory_utils import MemoryManager
from core.services.image_service import ImageService
from core.services.mask_service import MaskService
from core.services.segmentation_service import SegmentationService


class ApplicationService:
    """Main application service that coordinates all other services."""
    
    def __init__(self):
        self.image_service = ImageService()
        self.mask_service = MaskService()
        self._segmentation_service = None  # Lazy-loaded to avoid auto-initialization
        self.ui_state = {}
        self.main_window = None
    
    @property
    def segmentation_service(self):
        """Lazy-load segmentation service only when explicitly requested."""
        if self._segmentation_service is None:
            self._segmentation_service = SegmentationService(self.main_window)
        return self._segmentation_service
    
    def load_image(self, file_path: str, create_ui_components: bool = False) -> Optional[np.ndarray]:
        """Load an image and optionally set up UI components."""
        # Clear existing segmenter to free memory
        self.segmentation_service.reset_segmenter()
        
        # Load the image
        image = self.image_service.load_image(file_path)
        if image is None:
            return None
        
        # Clear existing masks
        self.mask_service.clear_all_masks()
        
        # Create processor (always needed)
        processor = self.image_service.create_image_processor(image)
        
        # Only create UI components if requested
        if create_ui_components:
            try:
                self.image_service.create_crop_rotate_ui(image, processor)
            except Exception as e:
                print(f"Warning: Could not create UI components: {e}")
        
        return image
    
    def perform_automatic_segmentation(self) -> Tuple[List[Dict[str, Any]], List[str]]:
        """Perform automatic segmentation on current image."""
        current_image = self.image_service.get_current_image()
        if current_image is None:
            return [], []
        
        # Use crop/rotate UI's processed image if available
        if (self.image_service.crop_rotate_ui and 
            hasattr(self.image_service.crop_rotate_ui, 'original_image')):
            image_to_segment = self.image_service.crop_rotate_ui.original_image
        else:
            image_to_segment = current_image
        
        masks = self.segmentation_service.segment_image(image_to_segment)
        
        # Replace all existing masks
        self.mask_service.replace_all_masks(masks)
        
        return self.mask_service.get_masks(), self.mask_service.get_mask_names()
    
    def perform_box_segmentation(self, box: List[int]) -> Tuple[List[Dict[str, Any]], List[str]]:
        """Perform box-guided segmentation on current image."""
        current_image = self.image_service.get_current_image()
        if current_image is None:
            return [], []
        
        # Use crop/rotate UI's processed image if available
        if (self.image_service.crop_rotate_ui and 
            hasattr(self.image_service.crop_rotate_ui, 'original_image')):
            image_to_segment = self.image_service.crop_rotate_ui.original_image
        else:
            image_to_segment = current_image
        
        new_masks = self.segmentation_service.segment_with_box(image_to_segment, box)
        
        # Add to existing masks (accumulate)
        self.mask_service.add_masks(new_masks)
        
        return self.mask_service.get_masks(), self.mask_service.get_mask_names()
    
    def delete_mask(self, mask_index: int) -> bool:
        """Delete a mask."""
        return self.mask_service.delete_mask(mask_index)
    
    def rename_mask(self, mask_index: int, new_name: str) -> bool:
        """Rename a mask."""
        return self.mask_service.rename_mask(mask_index, new_name)
    
    def clear_all_masks(self) -> None:
        """Clear all masks and reset image processor to original state."""
        # Clear mask service data
        self.mask_service.clear_all_masks()
        
        # Reset image processor to original state if available
        if (self.image_service and self.image_service.image_processor):
            processor = self.image_service.image_processor
            
            # Disable mask editing
            processor.set_mask_editing(False)
            
            # Reset base image to original image (removes all committed mask edits)
            if hasattr(processor, 'original') and processor.original is not None:
                processor.base_image = processor.original.copy()
                processor.current = processor.original.copy()
                processor.clear_optimization_cache()
    
    def get_mask_service(self) -> MaskService:
        """Get the mask service."""
        return self.mask_service
    
    def get_segmentation_service(self) -> SegmentationService:
        """Get the segmentation service."""
        return self.segmentation_service
    
    def cleanup(self):
        """Cleanup all services and free resources."""
        try:
            # Cleanup segmentation service (frees GPU memory)
            if hasattr(self.segmentation_service, 'cleanup'):
                self.segmentation_service.cleanup()
            
            # Clear CUDA cache
            MemoryManager.clear_cuda_cache()
            
            # Clear all masks
            self.mask_service.clear_all_masks()
            
        except Exception as e:
            print(f"Cleanup error: {e}")
