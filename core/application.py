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
            # Use local inpainting model checkpoint
            import os
            from pathlib import Path
            
            # Get the absolute path to the local model
            project_root = Path(__file__).parent.parent
            local_model_path = project_root / "assets" / "models" / "512-inpainting-ema.ckpt"
            
            if local_model_path.exists():
                inpainting_model = str(local_model_path)
                print(f"Using local inpainting model: {inpainting_model}")
            else:
                # Fallback to HuggingFace model if local not found
                inpainting_model = "runwayml/stable-diffusion-inpainting"
                print(f"Local model not found, using HuggingFace model: {inpainting_model}")
            
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
    
    def reimagine_mask(self, mask_index: int, prompt: str, negative_prompt: str = "", **generate_kwargs) -> Optional[np.ndarray]:
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

        # CRITICAL: Free up GPU memory before inpainting
        print("Preparing GPU memory for inpainting...")
        self._prepare_gpu_memory_for_inpainting()

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
        result = self.generative_service.reimagine(base_image, transformed_mask, prompt, negative_prompt=negative_prompt, **generate_kwargs)
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
        self._save_inpainting_result(blended, result, mask_bool, prompt, negative_prompt)

        # Update image processor to use the blended result
        self._update_image_processor_with_result(blended)

        return blended
    
    def _get_base_image_for_inpainting(self) -> Optional[np.ndarray]:
        """Get the appropriate base image for inpainting - should be the original unprocessed image."""
        # Priority 1: Use the image processor's original image (unprocessed)
        if (self.image_service.image_processor and 
            hasattr(self.image_service.image_processor, 'original') and
            self.image_service.image_processor.original is not None):
            return self.image_service.image_processor.original.copy()
        
        # Priority 2: Use crop/rotate UI's original image if available (has crop/rotate/flip applied)
        if (self.image_service.crop_rotate_ui and 
            hasattr(self.image_service.crop_rotate_ui, 'original_image') and
            self.image_service.crop_rotate_ui.original_image is not None):
            return self.image_service.crop_rotate_ui.original_image.copy()
        
        # Priority 3: Use the raw current image as fallback
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
                # With the new architecture, we need to update the original image
                # and clear any existing edits since the inpainting result becomes the new base
                proc.original = blended_image.copy()
                # Clear all edits since they've been baked into the new image
                proc.committed_global_edits = proc._get_default_parameters()
                proc.committed_mask_edits.clear()
                proc.reset_current_parameters()
                proc.clear_optimization_cache()
            
            # Update current image
            self.image_service.current_image = blended_image.copy()
            
            # Update crop/rotate UI if available
            if self.image_service.crop_rotate_ui:
                self.image_service.crop_rotate_ui.original_image = blended_image.copy()
                
        except Exception as e:
            print(f"Warning: Could not update image processor with result: {e}")

    def _save_inpainting_result(self, blended_image: np.ndarray, generated_content: np.ndarray, 
                               mask: np.ndarray, prompt: str, negative_prompt: str = "") -> None:
        """Save inpainting results to PNG files."""
        import cv2
        import os
        from datetime import datetime
        
        # Create output directory if it doesn't exist
        output_dir = "inpainting_results"
        os.makedirs(output_dir, exist_ok=True)
        
        # Create timestamp for unique filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Clean prompts for filename (remove special characters)
        clean_positive = "".join(c for c in prompt if c.isalnum() or c in (' ', '-', '_')).rstrip()
        clean_positive = clean_positive.replace(' ', '_')[:30] if clean_positive else "no_prompt"
        
        clean_negative = "".join(c for c in negative_prompt if c.isalnum() or c in (' ', '-', '_')).rstrip()
        clean_negative = clean_negative.replace(' ', '_')[:20] if clean_negative else ""
        
        # Create filename with both prompts
        if clean_negative:
            filename_suffix = f"{clean_positive}_neg_{clean_negative}"
        else:
            filename_suffix = clean_positive
        
        # Save the full blended result
        blended_filename = f"{output_dir}/inpainted_full_{timestamp}_{filename_suffix}.png"
        
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
        
        generated_filename = f"{output_dir}/inpainted_mask_only_{timestamp}_{filename_suffix}.png"
        
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
            
            # Clear all committed mask edits (removes all mask edits from the processor)
            if hasattr(processor, 'committed_mask_edits'):
                processor.committed_mask_edits.clear()
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
    
    def _prepare_gpu_memory_for_inpainting(self):
        """Prepare GPU memory for inpainting by freeing up resources from other models."""
        try:
            print("Freeing GPU memory before inpainting...")
            
            # 1. Clean up segmentation service if it exists and has resources
            if self._segmentation_service is not None:
                if hasattr(self._segmentation_service, 'segmenter') and self._segmentation_service.segmenter is not None:
                    print("Moving segmentation model to CPU to free GPU memory...")
                    self._segmentation_service.segmenter.cleanup_memory()
            
            # 2. Force garbage collection and CUDA cache cleanup
            MemoryManager.clear_cuda_cache()
            
            # 3. Check available memory
            memory_info = MemoryManager.get_device_info()
            print(f"GPU memory after cleanup: {memory_info['free_mb']:.0f}MB free of {memory_info['total_mb']:.0f}MB total")
            
            # 4. If still low on memory, try more aggressive cleanup
            if memory_info['free_mb'] < 2000:  # Less than 2GB free
                print("Low GPU memory detected, performing aggressive cleanup...")
                
                # Clear any cached states in image processor
                if (self.image_service and 
                    self.image_service.image_processor and 
                    hasattr(self.image_service.image_processor, 'clear_optimization_cache')):
                    self.image_service.image_processor.clear_optimization_cache()
                
                # Additional cleanup
                MemoryManager.clear_cuda_cache()
                
                # Re-check memory
                memory_info = MemoryManager.get_device_info()
                print(f"GPU memory after aggressive cleanup: {memory_info['free_mb']:.0f}MB free")
                
                if memory_info['free_mb'] < 1500:  # Still less than 1.5GB
                    print("WARNING: GPU memory still low. Inpainting may fail or be very slow.")
                    print("Consider closing other GPU-intensive applications or using a smaller model.")
            
        except Exception as e:
            print(f"Error during GPU memory preparation: {e}")
            # Continue anyway - this is best-effort cleanup
