"""
Core application service layer.
Centralizes business logic and coordinates between different subsystems.
"""
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import traceback

from utils.memory_utils import MemoryManager, ErrorHandler, ResourceManager
from ui.segmentation import ImageSegmenter
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
        from ui.crop_rotate import CropRotateUI
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

class MaskService:
    """Service for managing mask operations and state."""
    
    def __init__(self):
        self.layer_masks: List[Dict[str, Any]] = []
        self.mask_names: List[str] = []
        self.mask_editing_enabled = False
        self.current_mask_index = -1
        self.mask_parameters: Dict[int, Dict[str, Any]] = {}
        self.global_parameters: Optional[Dict[str, Any]] = None
    
    def add_masks(self, masks: List[Dict[str, Any]], names: Optional[List[str]] = None) -> None:
        """Add new masks to the collection."""
        start_index = len(self.layer_masks)
        self.layer_masks.extend(masks)
        
        # Add names for new masks
        if names and len(names) >= len(masks):
            self.mask_names.extend(names[:len(masks)])
        else:
            for idx in range(len(masks)):
                self.mask_names.append(f"Mask {start_index + idx + 1}")
    
    def replace_all_masks(self, masks: List[Dict[str, Any]], names: Optional[List[str]] = None) -> None:
        """Replace all existing masks with new ones."""
        self.layer_masks = masks
        
        if names and len(names) >= len(masks):
            self.mask_names = names[:len(masks)]
        else:
            self.mask_names = [f"Mask {idx + 1}" for idx in range(len(masks))]
    
    def clear_all_masks(self) -> None:
        """Clear all masks."""
        self.layer_masks.clear()
        self.mask_names.clear()
        self.mask_parameters.clear()
        self.mask_editing_enabled = False
        self.current_mask_index = -1
    
    def delete_mask(self, mask_index: int) -> bool:
        """Delete a specific mask."""
        if 0 <= mask_index < len(self.layer_masks):
            self.layer_masks.pop(mask_index)
            if mask_index < len(self.mask_names):
                self.mask_names.pop(mask_index)
            
            # Remove parameters for this mask
            if mask_index in self.mask_parameters:
                del self.mask_parameters[mask_index]
            
            # Adjust parameter indices for masks after the deleted one
            new_params = {}
            for idx, params in self.mask_parameters.items():
                if idx > mask_index:
                    new_params[idx - 1] = params
                elif idx < mask_index:
                    new_params[idx] = params
            self.mask_parameters = new_params
            
            # Adjust current mask index
            if self.current_mask_index == mask_index:
                self.current_mask_index = -1
                self.mask_editing_enabled = False
            elif self.current_mask_index > mask_index:
                self.current_mask_index -= 1
            
            return True
        return False
    
    def rename_mask(self, mask_index: int, new_name: str) -> bool:
        """Rename a specific mask."""
        if 0 <= mask_index < len(self.mask_names):
            self.mask_names[mask_index] = new_name
            return True
        return False
    
    def save_mask_parameters(self, mask_index: int, parameters: Dict[str, Any]) -> None:
        """Save parameters for a specific mask."""
        if 0 <= mask_index < len(self.layer_masks):
            self.mask_parameters[mask_index] = parameters.copy()
    
    def get_masks(self) -> List[Dict[str, Any]]:
        """Get all masks."""
        return self.layer_masks
    
    def get_mask_names(self) -> List[str]:
        """Get all mask names."""
        return self.mask_names

class SegmentationService:
    """Service for managing segmentation operations."""
    
    def __init__(self, main_window=None):
        self.segmenter = None
        self.resource_manager = ResourceManager()
        self.segmentation_mode = False
        self.pending_selection = None
        self.main_window = main_window
    
    def get_segmenter(self):
        """Get or create segmenter instance with memory-efficient approach."""
        if self.segmenter is None:
            try:  
                # Check available memory first
                memory_info = MemoryManager.get_device_info()
                device = MemoryManager.select_optimal_device(min_gpu_memory_gb=4.0)
                
                # If GPU doesn't have enough memory, force CPU mode
                if device == "cuda" and memory_info['free_mb'] < 3000:  # 3GB threshold
                    device = "cpu"
                
                self.main_window._update_status(f"Selected device to apply segmentation: {device}")
                
                # Clear memory before creating segmenter
                MemoryManager.clear_cuda_cache()
                
                self.segmenter = ImageSegmenter(device=device)
                
                # Register with resource manager
                self.resource_manager.register_model(
                    "sam_segmenter", 
                    self.segmenter.model, 
                    device
                )
                
            except Exception as e:
                error_msg = ErrorHandler.handle_memory_error(e, "segmenter creation")
                print(error_msg)
                
                # Try fallback to CPU if GPU failed
                try:
                    MemoryManager.clear_cuda_cache()
                    self.segmenter = ImageSegmenter(device="cpu")
                    self.main_window._update_status("ImageSegmenter created successfully on CPU as fallback")
                except Exception as cpu_e:
                    print(f"CPU fallback also failed: {cpu_e}")
                    return None
        
        return self.segmenter
    
    def segment_image(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Perform automatic segmentation on an image."""
        segmenter = self.get_segmenter()
        if segmenter is None:
            return []
        
        try:
            # Clean up memory before segmentation
            self.cleanup_memory()
            
            return ErrorHandler.safe_gpu_operation(
                operation=lambda: segmenter.segment(image),
                fallback_operation=lambda: self._fallback_segment(image),
                operation_name="automatic segmentation"
            )
        except Exception as e:
            error_msg = ErrorHandler.handle_memory_error(e, "automatic segmentation")
            print(error_msg)
            return []
    
    def segment_with_box(self, image: np.ndarray, box: List[int]) -> List[Dict[str, Any]]:
        """Perform box-guided segmentation on an image."""
        segmenter = self.get_segmenter()
        if segmenter is None:
            return []
        
        try:
            # Clean up memory before segmentation
            self.cleanup_memory()
            
            return ErrorHandler.safe_gpu_operation(
                operation=lambda: segmenter.segment_with_box(image, box),
                fallback_operation=lambda: self._fallback_segment_with_box(image, box),
                operation_name="box segmentation"
            )
        except Exception as e:
            error_msg = ErrorHandler.handle_memory_error(e, "box segmentation")
            print(error_msg)
            return []
    
    def _fallback_segment(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Fallback segmentation method."""
        if self.segmenter and hasattr(self.segmenter, '_fallback_segment'):
            return self.segmenter._fallback_segment(image)
        return []
    
    def _fallback_segment_with_box(self, image: np.ndarray, box: List[int]) -> List[Dict[str, Any]]:
        """Fallback box segmentation method."""
        if self.segmenter and hasattr(self.segmenter, '_fallback_segment_with_box'):
            return self.segmenter._fallback_segment_with_box(image, box)
        return []
    
    def cleanup_memory(self) -> None:
        """Clean up segmentation memory."""
        if self.segmenter and hasattr(self.segmenter, 'cleanup_memory'):
            self.segmenter.cleanup_memory()
        
        self.resource_manager.move_to_cpu_if_needed("sam_segmenter")
        MemoryManager.clear_cuda_cache()
    
    def reset_segmenter(self) -> None:
        """Reset the segmenter instance."""
        if self.segmenter:
            self.cleanup_memory()
            self.segmenter = None
    
    def enable_segmentation_mode(self) -> bool:
        """Enable segmentation mode."""
        self.segmentation_mode = True
        self.pending_selection = None
        return True
    
    def disable_segmentation_mode(self) -> None:
        """Disable segmentation mode."""
        self.segmentation_mode = False
        self.pending_selection = None
    
    def transform_texture_coordinates_to_image(self, box, crop_rotate_ui):
        """Transform texture coordinates to image coordinates, accounting for flips."""
        try:
            if not crop_rotate_ui or crop_rotate_ui.original_image is None:
                return None
                
            image = crop_rotate_ui.original_image
            h, w = image.shape[:2]
            
            # Calculate the texture offsets where the image is centered within the texture
            texture_w = crop_rotate_ui.texture_w
            texture_h = crop_rotate_ui.texture_h
            offset_x = (texture_w - w) // 2
            offset_y = (texture_h - h) // 2
            
            # Get flip states
            flip_horizontal = False
            flip_vertical = False
            if hasattr(crop_rotate_ui, 'get_flip_states'):
                flip_states = crop_rotate_ui.get_flip_states()
                flip_horizontal = flip_states.get('flip_horizontal', False)
                flip_vertical = flip_states.get('flip_vertical', False)
            
            # Convert all box coordinates to Python scalars first
            box_scalars = []
            for i, coord in enumerate(box):
                try:
                    if hasattr(coord, 'item'):  # numpy scalar
                        scalar_val = coord.item()
                    else:
                        scalar_val = float(coord)
                    box_scalars.append(scalar_val)
                except Exception as e:
                    print(f"DEBUG: Error converting box[{i}] = {coord}: {e}")
                    return None
            
            # Convert box coordinates from texture space to image space
            x1 = max(0, min(w-1, int(box_scalars[0] - offset_x)))
            y1 = max(0, min(h-1, int(box_scalars[1] - offset_y)))
            x2 = max(0, min(w-1, int(box_scalars[2] - offset_x)))
            y2 = max(0, min(h-1, int(box_scalars[3] - offset_y)))
            
            # Apply flip transformations to the box coordinates
            if flip_horizontal:
                # Flip X coordinates: x_new = image_width - x_old
                x1_flipped = w - 1 - x2  # Right becomes left
                x2_flipped = w - 1 - x1  # Left becomes right
                x1, x2 = x1_flipped, x2_flipped
            
            if flip_vertical:
                # Flip Y coordinates: y_new = image_height - y_old
                y1_flipped = h - 1 - y2  # Bottom becomes top
                y2_flipped = h - 1 - y1  # Top becomes bottom
                y1, y2 = y1_flipped, y2_flipped
            
            # Ensure coordinates are in correct order (x1 < x2, y1 < y2)
            if x1 > x2:
                x1, x2 = x2, x1
            if y1 > y2:
                y1, y2 = y2, y1
            
            scaled_box = [x1, y1, x2, y2]
            
            # Ensure all coordinates are integers (not numpy types)
            scaled_box = [int(coord) for coord in scaled_box]
            
            # Validate box dimensions
            if scaled_box[2] <= scaled_box[0] or scaled_box[3] <= scaled_box[1]:
                # Try to create a minimal valid box
                center_x = (scaled_box[0] + scaled_box[2]) // 2
                center_y = (scaled_box[1] + scaled_box[3]) // 2
                min_size = 10
                scaled_box = [
                    max(0, center_x - min_size//2),
                    max(0, center_y - min_size//2),
                    min(w, center_x + min_size//2),
                    min(h, center_y + min_size//2)
                ]

            return scaled_box
            
        except Exception as e:
            print(f"DEBUG: Exception in transform_texture_coordinates_to_image: {e}")
            traceback.print_exc()
            return None
    
    def create_default_bounding_box(self, crop_rotate_ui):
        """Create a default bounding box as fallback."""
        if not crop_rotate_ui or crop_rotate_ui.original_image is None:
            return None
            
        h, w = crop_rotate_ui.original_image.shape[:2]
        offset_x = (crop_rotate_ui.texture_w - w) // 2
        offset_y = (crop_rotate_ui.texture_h - h) // 2
        
        # Create a box in the center that's 80% of the image size
        box_w = int(w * 0.8)
        box_h = int(h * 0.8)
        box_x = offset_x + (w - box_w) // 2
        box_y = offset_y + (h - box_h) // 2
        
        return {
            "x": box_x,
            "y": box_y,
            "w": box_w,
            "h": box_h
        }
    
    def confirm_segmentation_selection(self, pending_box, crop_rotate_ui, app_service):
        """Confirm segmentation selection and perform segmentation."""        
        if not self.segmentation_mode:
            return False, "Segmentation mode is not active"
            
        if not pending_box:
            # Try to create a default box
            pending_box = self.create_default_bounding_box(crop_rotate_ui)
            if not pending_box:
                return False, "No area selected and could not create default selection"
        
        # Validate box dimensions
        if pending_box["w"] < 10 or pending_box["h"] < 10:
            return False, f"Selection area too small: {pending_box['w']}x{pending_box['h']} (minimum 10x10)"
        
        # Box format: [x1, y1, x2, y2]
        box = [pending_box["x"], pending_box["y"], 
               pending_box["x"] + pending_box["w"], 
               pending_box["y"] + pending_box["h"]]
        
        # Transform coordinates
        scaled_box = self.transform_texture_coordinates_to_image(box, crop_rotate_ui)
        if not scaled_box:
            return False, "Could not transform coordinates"
        
        # Validate transformed box
        try:
            # Ensure all coordinates are scalar values and finite
            for i, coord in enumerate(scaled_box):
                if not isinstance(coord, (int, float, np.integer, np.floating)):
                    return False, f"Invalid coordinate type: {type(coord)}"
                # Convert numpy scalars to Python scalars for safe comparison
                coord_val = float(coord) if hasattr(coord, 'item') else coord
                if not np.isfinite(coord_val):
                    return False, f"Non-finite coordinate: {coord_val}"
        except Exception as e:
            return False, f"Coordinate validation error: {str(e)}"
        
        try:
            if scaled_box[2] - scaled_box[0] < 10 or scaled_box[3] - scaled_box[1] < 10:
                return False, "Selection too small relative to image"
        except Exception as e:
            return False, f"Box size validation error: {str(e)}"
        
        # Perform segmentation
        try:
            masks, mask_names = app_service.perform_box_segmentation(scaled_box)
            if not masks or len(masks) == 0:
                return False, "No objects found in selection"
            
            return True, f"Created {len(masks)} total masks"
        except Exception as e:
            return False, f"Error: {str(e)}"


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
        """Clear all masks."""
        self.mask_service.clear_all_masks()
    
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
