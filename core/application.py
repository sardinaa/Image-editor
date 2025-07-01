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
from core.services.generative_service import GenerativeService


class ApplicationService:
    """Main application service that coordinates all other services."""
    
    def __init__(self):
        self.image_service = ImageService()
        self.mask_service = MaskService()
        self._segmentation_service = None  # Lazy-loaded to avoid auto-initialization
        self._generative_service = None
        self.ui_state = {}
        self.main_window = None
    
    @property
    def segmentation_service(self):
        """Lazy-load segmentation service only when explicitly requested."""
        if self._segmentation_service is None:
            self._segmentation_service = SegmentationService(self.main_window)
        return self._segmentation_service
    
    @property
    def generative_service(self):
        """Lazy-load generative service for image synthesis."""
        if self._generative_service is None:
            # Only use inpainting models
            inpainting_model = "runwayml/stable-diffusion-inpainting"
            
            self._generative_service = GenerativeService(model_path=inpainting_model)
        return self._generative_service
    
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
    
    def reimagine_mask(self, mask_index: int, prompt: str, **generate_kwargs) -> Optional[np.ndarray]:
        """Regenerate content within the specified mask using a generative model."""
        masks = self.mask_service.get_masks()
        if not (0 <= mask_index < len(masks)):
            print(f"Error: mask_index {mask_index} out of range (0-{len(masks)-1})")
            return None

        mask_data = masks[mask_index]
        original_mask = mask_data.get("segmentation")
        if original_mask is None:
            print("Error: No segmentation data in mask")
            return None

        # Get the base image for inpainting (before processing effects)
        base_image = self._get_base_image_for_inpainting()
        if base_image is None:
            print("Error: Could not get base image for inpainting")
            return None

        # Transform mask if needed to match the base image coordinate system
        transformed_mask = self._transform_mask_for_inpainting(original_mask)
        
        # Validate mask dimensions match image
        if transformed_mask.shape[:2] != base_image.shape[:2]:
            print(f"Warning: Mask shape {transformed_mask.shape[:2]} doesn't match image shape {base_image.shape[:2]}")
            # Resize mask to match image
            import cv2
            transformed_mask = cv2.resize(
                transformed_mask.astype(np.uint8), 
                (base_image.shape[1], base_image.shape[0]), 
                interpolation=cv2.INTER_NEAREST
            ).astype(bool)

        # Ensure mask has valid content
        if not np.any(transformed_mask):
            print("Error: Mask is empty after transformation")
            return None

        print(f"Inpainting with image shape: {base_image.shape}, mask shape: {transformed_mask.shape}")
        print(f"Mask area: {np.sum(transformed_mask)} pixels ({np.sum(transformed_mask)/transformed_mask.size*100:.1f}%)")
        
        # Perform inpainting
        result = self.generative_service.reimagine(base_image, transformed_mask, prompt, **generate_kwargs)
        if result is None:
            print("Error: Generative service returned None")
            return None

        # Ensure result and image have compatible formats
        if base_image.ndim == 3 and result.ndim == 3:
            if base_image.shape[2] == 4 and result.shape[2] == 3:
                import cv2
                result = cv2.cvtColor(result, cv2.COLOR_RGB2RGBA)
            elif base_image.shape[2] == 3 and result.shape[2] == 4:
                import cv2
                result = cv2.cvtColor(result, cv2.COLOR_RGBA2RGB)

        # Ensure result dimensions match image
        if result.shape[:2] != base_image.shape[:2]:
            import cv2
            result = cv2.resize(result, (base_image.shape[1], base_image.shape[0]))

        # Create final mask for blending
        if transformed_mask.dtype != bool:
            mask_bool = transformed_mask > 0
        else:
            mask_bool = transformed_mask

        # Apply inpainting result
        blended = base_image.copy()
        if base_image.shape[2] == 4:
            # Handle RGBA images
            blended[mask_bool] = result[mask_bool]
        else:
            # Handle RGB images
            blended[mask_bool] = result[mask_bool]

        # Save the inpainting result to PNG
        self._save_inpainting_result(blended, result, mask_bool, prompt)

        # Update image processor to use the blended result
        self._update_image_processor_with_result(blended)

        return blended
    
    def _get_base_image_for_inpainting(self) -> Optional[np.ndarray]:
        """Get the appropriate base image for inpainting, avoiding processed effects that might interfere."""
        # Priority 1: Use crop/rotate UI's original image if available (this has crop/rotate/flip applied)
        if (self.image_service.crop_rotate_ui and 
            hasattr(self.image_service.crop_rotate_ui, 'original_image') and
            self.image_service.crop_rotate_ui.original_image is not None):
            return self.image_service.crop_rotate_ui.original_image.copy()
        
        # Priority 2: Use the image processor's base image (has committed edits but not current slider values)
        if (self.image_service.image_processor and 
            hasattr(self.image_service.image_processor, 'base_image') and
            self.image_service.image_processor.base_image is not None):
            return self.image_service.image_processor.base_image.copy()
        
        # Priority 3: Use the raw current image
        if self.image_service.current_image is not None:
            return self.image_service.current_image.copy()
        
        return None
    
    def _transform_mask_for_inpainting(self, mask: np.ndarray) -> np.ndarray:
        """Transform mask coordinates to match the inpainting image coordinate system."""
        # For now, return the mask as-is since we're using the original image
        # In the future, this could handle coordinate transformations if needed
        return mask.copy()
    
    def _update_image_processor_with_result(self, blended_image: np.ndarray) -> None:
        """Update the image processor with the inpainting result."""
        try:
            if self.image_service.image_processor:
                proc = self.image_service.image_processor
                # Update base image to include the inpainting changes
                proc.base_image = blended_image.copy()
                proc.current = blended_image.copy()
                proc.clear_optimization_cache()
            
            # Update current image
            self.image_service.current_image = blended_image.copy()
            
            # Update crop/rotate UI if available
            if self.image_service.crop_rotate_ui:
                self.image_service.crop_rotate_ui.original_image = blended_image.copy()
                
        except Exception as e:
            print(f"Warning: Could not update image processor with result: {e}")

    def _save_inpainting_result(self, blended_image: np.ndarray, generated_content: np.ndarray, 
                               mask: np.ndarray, prompt: str) -> None:
        """Save inpainting results to PNG files."""
        import cv2
        import os
        from datetime import datetime
        
        # Create output directory if it doesn't exist
        output_dir = "inpainting_results"
        os.makedirs(output_dir, exist_ok=True)
        
        # Create timestamp for unique filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Clean prompt for filename (remove special characters)
        clean_prompt = "".join(c for c in prompt if c.isalnum() or c in (' ', '-', '_')).rstrip()
        clean_prompt = clean_prompt.replace(' ', '_')[:50]  # Limit length and replace spaces
        
        # Save the full blended result
        blended_filename = f"{output_dir}/inpainted_full_{timestamp}_{clean_prompt}.png"
        
        # Convert from RGB/RGBA to BGR for OpenCV saving
        if blended_image.shape[2] == 4:  # RGBA
            blended_bgr = cv2.cvtColor(blended_image, cv2.COLOR_RGBA2BGRA)
        else:  # RGB
            blended_bgr = cv2.cvtColor(blended_image, cv2.COLOR_RGB2BGR)
            
        cv2.imwrite(blended_filename, blended_bgr)
        print(f"Saved full inpainting result to: {blended_filename}")
        
        # Save just the generated content area
        generated_only = np.zeros_like(blended_image)
        generated_only[mask] = generated_content[mask]
        
        generated_filename = f"{output_dir}/inpainted_mask_only_{timestamp}_{clean_prompt}.png"
        
        if generated_only.shape[2] == 4:  # RGBA
            generated_bgr = cv2.cvtColor(generated_only, cv2.COLOR_RGBA2BGRA)
        else:  # RGB
            generated_bgr = cv2.cvtColor(generated_only, cv2.COLOR_RGB2BGR)
            
        cv2.imwrite(generated_filename, generated_bgr)
        print(f"Saved generated content only to: {generated_filename}")

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

            if hasattr(self.generative_service, 'cleanup'):
                self.generative_service.cleanup()
            
            # Clear CUDA cache
            MemoryManager.clear_cuda_cache()
            
            # Clear all masks
            self.mask_service.clear_all_masks()
            
        except Exception as e:
            print(f"Cleanup error: {e}")
