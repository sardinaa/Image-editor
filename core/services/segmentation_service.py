"""
Segmentation service for managing segmentation operations.
"""
from typing import List, Dict, Any
import numpy as np
import traceback

from utils.memory_utils import MemoryManager, ErrorHandler, ResourceManager
from processing.segmentation import ImageSegmenter


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
