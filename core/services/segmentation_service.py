"""
Segmentation service for managing segmentation operations.
"""
from typing import List, Dict, Any
import numpy as np
import cv2
import traceback
import dearpygui.dearpygui as dpg

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
        """Transform texture coordinates to image coordinates, accounting for rotation and flips."""
        try:
            if not crop_rotate_ui or crop_rotate_ui.original_image is None:
                return None
                
            image = crop_rotate_ui.original_image
            h, w = image.shape[:2]
            
            # Get current rotation angle
            angle = 0
            if dpg.does_item_exist("rotation_slider"):
                angle = dpg.get_value("rotation_slider")
            
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
            

            # Use rotation path if angle != 0 and rotated attributes exist and are valid
            use_rotated = (
                angle != 0 and
                hasattr(crop_rotate_ui, 'rot_w') and hasattr(crop_rotate_ui, 'rot_h') and
                hasattr(crop_rotate_ui, 'offset_x') and hasattr(crop_rotate_ui, 'offset_y') and
                crop_rotate_ui.rot_w and crop_rotate_ui.rot_h and
                crop_rotate_ui.rot_w > 0 and crop_rotate_ui.rot_h > 0
            )

            if use_rotated:
                print(f"DEBUG: [FIXED] Using rotation path: angle={angle}, rot_w={crop_rotate_ui.rot_w}, rot_h={crop_rotate_ui.rot_h}, offsets=({crop_rotate_ui.offset_x},{crop_rotate_ui.offset_y})")
                return self._transform_coordinates_with_rotation(box_scalars, crop_rotate_ui, angle)
            else:
                print(f"DEBUG: [FIXED] Using non-rotation path: angle={angle}, w={w}, h={h}")
                return self._transform_coordinates_without_rotation(box_scalars, crop_rotate_ui, w, h)
                
        except Exception as e:
            print(f"DEBUG: Exception in transform_texture_coordinates_to_image: {e}")
            traceback.print_exc()
            return None
    
    def _transform_coordinates_with_rotation(self, box_scalars, crop_rotate_ui, angle):
        """Handle coordinate transformation when image is rotated."""
        try:
            image = crop_rotate_ui.original_image
            h, w = image.shape[:2]
            
            # Get rotated image dimensions from crop_rotate_ui
            if hasattr(crop_rotate_ui, 'rot_w') and hasattr(crop_rotate_ui, 'rot_h'):
                rot_w = crop_rotate_ui.rot_w
                rot_h = crop_rotate_ui.rot_h
                rot_offset_x = crop_rotate_ui.offset_x
                rot_offset_y = crop_rotate_ui.offset_y
            else:
                # Fallback: calculate rotated dimensions
                angle_rad = np.deg2rad(angle)
                cos_a = abs(np.cos(angle_rad))
                sin_a = abs(np.sin(angle_rad))
                rot_w = int(w * cos_a + h * sin_a)
                rot_h = int(h * cos_a + w * sin_a)
                rot_offset_x = (crop_rotate_ui.texture_w - rot_w) // 2
                rot_offset_y = (crop_rotate_ui.texture_h - rot_h) // 2
            
            # Get flip states
            flip_horizontal = False
            flip_vertical = False
            if hasattr(crop_rotate_ui, 'get_flip_states'):
                flip_states = crop_rotate_ui.get_flip_states()
                flip_horizontal = flip_states.get('flip_horizontal', False)
                flip_vertical = flip_states.get('flip_vertical', False)
            
            print(f"DEBUG: [ROTATION+FLIP] Transform order: texture -> rotated+flipped -> inverse_flips -> inverse_rotation -> original")
            print(f"DEBUG: [ROTATION+FLIP] Input box (texture): {box_scalars}")
            print(f"DEBUG: [ROTATION+FLIP] Rotation angle: {angle}, Flips: H={flip_horizontal}, V={flip_vertical}")
            print(f"DEBUG: [ROTATION+FLIP] Rotated dims: {rot_w}x{rot_h}, offsets: ({rot_offset_x},{rot_offset_y})")
            
            # Convert box coordinates from texture space to rotated image space
            x1_rot = box_scalars[0] - rot_offset_x
            y1_rot = box_scalars[1] - rot_offset_y
            x2_rot = box_scalars[2] - rot_offset_x
            y2_rot = box_scalars[3] - rot_offset_y
            
            print(f"DEBUG: [ROTATION+FLIP] Box in rotated+flipped space: [{x1_rot}, {y1_rot}, {x2_rot}, {y2_rot}]")
            
            # FIRST: Apply inverse flips to the coordinates in rotated space
            # This undoes the flips that were applied AFTER rotation in the display
            if flip_vertical:
                # Flip Y coordinates in rotated space: y_new = rotated_height - y_old
                y1_unflipped = rot_h - 1 - y2_rot  # Bottom becomes top
                y2_unflipped = rot_h - 1 - y1_rot  # Top becomes bottom
                y1_rot, y2_rot = y1_unflipped, y2_unflipped
                print(f"DEBUG: [ROTATION+FLIP] After undoing vertical flip: [{x1_rot}, {y1_rot}, {x2_rot}, {y2_rot}]")
            
            if flip_horizontal:
                # Flip X coordinates in rotated space: x_new = rotated_width - x_old
                x1_unflipped = rot_w - 1 - x2_rot  # Right becomes left
                x2_unflipped = rot_w - 1 - x1_rot  # Left becomes right
                x1_rot, x2_rot = x1_unflipped, x2_unflipped
                print(f"DEBUG: [ROTATION+FLIP] After undoing horizontal flip: [{x1_rot}, {y1_rot}, {x2_rot}, {y2_rot}]")
            
            # SECOND: Apply inverse rotation to get original image coordinates
            # Create rotation matrix (inverse rotation)
            center_rot = (rot_w / 2, rot_h / 2)
            M_inv = cv2.getRotationMatrix2D(center_rot, -angle, 1.0)
            
            # Transform bounding box corners
            corners_rot = np.array([
                [x1_rot, y1_rot, 1],
                [x2_rot, y1_rot, 1],
                [x1_rot, y2_rot, 1],
                [x2_rot, y2_rot, 1]
            ]).T
            
            corners_orig = M_inv @ corners_rot
            
            # Find bounding box in original image coordinates
            x_coords = corners_orig[0, :]
            y_coords = corners_orig[1, :]
            
            x1_orig = np.min(x_coords)
            y1_orig = np.min(y_coords)
            x2_orig = np.max(x_coords)
            y2_orig = np.max(y_coords)
            
            # Adjust coordinates to account for the fact that rotated image center
            # doesn't align with original image center
            center_orig = (w / 2, h / 2)
            x1_orig += center_orig[0] - center_rot[0]
            y1_orig += center_orig[1] - center_rot[1]
            x2_orig += center_orig[0] - center_rot[0]
            y2_orig += center_orig[1] - center_rot[1]
            
            print(f"DEBUG: [ROTATION+FLIP] After inverse rotation: [{x1_orig}, {y1_orig}, {x2_orig}, {y2_orig}]")
            
            # Clamp to image bounds and ensure correct order
            x1 = max(0, min(w-1, int(min(x1_orig, x2_orig))))
            y1 = max(0, min(h-1, int(min(y1_orig, y2_orig))))
            x2 = max(0, min(w-1, int(max(x1_orig, x2_orig))))
            y2 = max(0, min(h-1, int(max(y1_orig, y2_orig))))
            
            # Validate box dimensions
            if x2 <= x1 or y2 <= y1:
                # Create minimal valid box at center
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                min_size = 10
                x1 = max(0, center_x - min_size//2)
                y1 = max(0, center_y - min_size//2)
                x2 = min(w, center_x + min_size//2)
                y2 = min(h, center_y + min_size//2)
            
            final_box = [x1, y1, x2, y2]
            print(f"DEBUG: [ROTATION+FLIP] Final box in original space: {final_box}")
            
            return final_box
            
        except Exception as e:
            print(f"DEBUG: Error in rotation coordinate transformation: {e}")
            traceback.print_exc()
            return None
    
    def _transform_coordinates_without_rotation(self, box_scalars, crop_rotate_ui, w, h):
        """Handle coordinate transformation when image is not rotated."""
        try:
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
            
            print(f"DEBUG: [NO ROTATION] Transform: texture -> image -> apply_flips")
            print(f"DEBUG: [NO ROTATION] Input box (texture): {box_scalars}")
            print(f"DEBUG: [NO ROTATION] Image dims: {w}x{h}, texture offsets: ({offset_x},{offset_y})")
            print(f"DEBUG: [NO ROTATION] Flips: H={flip_horizontal}, V={flip_vertical}")
            
            # Convert box coordinates from texture space to image space
            x1 = max(0, min(w-1, int(box_scalars[0] - offset_x)))
            y1 = max(0, min(h-1, int(box_scalars[1] - offset_y)))
            x2 = max(0, min(w-1, int(box_scalars[2] - offset_x)))
            y2 = max(0, min(h-1, int(box_scalars[3] - offset_y)))
            
            print(f"DEBUG: [NO ROTATION] Box in image space (before flips): [{x1}, {y1}, {x2}, {y2}]")
            
            # Apply flip transformations to the box coordinates
            if flip_horizontal:
                # Flip X coordinates: x_new = image_width - x_old
                x1_flipped = w - 1 - x2  # Right becomes left
                x2_flipped = w - 1 - x1  # Left becomes right
                x1, x2 = x1_flipped, x2_flipped
                print(f"DEBUG: [NO ROTATION] After horizontal flip: [{x1}, {y1}, {x2}, {y2}]")
            
            if flip_vertical:
                # Flip Y coordinates: y_new = image_height - y_old
                y1_flipped = h - 1 - y2  # Bottom becomes top
                y2_flipped = h - 1 - y1  # Top becomes bottom
                y1, y2 = y1_flipped, y2_flipped
                print(f"DEBUG: [NO ROTATION] After vertical flip: [{x1}, {y1}, {x2}, {y2}]")
            
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

            print(f"DEBUG: [NO ROTATION] Final box in original space: {scaled_box}")
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
    
    def debug_coordinate_transformation(self, box, crop_rotate_ui):
        """Debug function to print coordinate transformation details."""
        try:
            if not crop_rotate_ui or crop_rotate_ui.original_image is None:
                print("DEBUG: No crop_rotate_ui or original_image")
                return
                
            print(f"DEBUG: Original box coordinates: {box}")
            
            # Get rotation angle
            angle = 0
            if dpg.does_item_exist("rotation_slider"):
                angle = dpg.get_value("rotation_slider")
            print(f"DEBUG: Rotation angle: {angle}")
            
            # Get crop mode
            crop_mode = dpg.get_value("crop_mode") if dpg.does_item_exist("crop_mode") else False
            print(f"DEBUG: Crop mode active: {crop_mode}")
            
            # Get image dimensions
            h, w = crop_rotate_ui.original_image.shape[:2]
            print(f"DEBUG: Original image dimensions: {w}x{h}")
            
            # Get texture dimensions
            print(f"DEBUG: Texture dimensions: {crop_rotate_ui.texture_w}x{crop_rotate_ui.texture_h}")
            
            # Get rotated dimensions if available
            if hasattr(crop_rotate_ui, 'rot_w') and hasattr(crop_rotate_ui, 'rot_h'):
                print(f"DEBUG: Rotated image dimensions: {crop_rotate_ui.rot_w}x{crop_rotate_ui.rot_h}")
                print(f"DEBUG: Rotation offsets: ({crop_rotate_ui.offset_x}, {crop_rotate_ui.offset_y})")
            
            # Get flip states
            if hasattr(crop_rotate_ui, 'get_flip_states'):
                flip_states = crop_rotate_ui.get_flip_states()
                print(f"DEBUG: Flip states: {flip_states}")
            
            # Apply transformation
            transformed_box = self.transform_texture_coordinates_to_image(box, crop_rotate_ui)
            print(f"DEBUG: Transformed box coordinates: {transformed_box}")
            
        except Exception as e:
            print(f"DEBUG: Error in coordinate transformation debug: {e}")
            traceback.print_exc()
