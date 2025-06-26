"""
Core application service layer.
Centralizes business logic and coordinates between different subsystems.
"""
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from pathlib import Path
import dearpygui.dearpygui as dpg
import cv2

# Import utilities
from utils.memory_utils import MemoryManager, ErrorHandler, ResourceManager
from utils.ui_helpers import UIStateManager, safe_item_check


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
            # Import here to avoid circular dependencies
            from processing.file_manager import load_image
            
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
    
    def save_image(self, file_path: str, image: Optional[np.ndarray] = None) -> bool:
        """Save an image to file."""
        try:
            # Import here to avoid circular dependencies
            from processing.file_manager import save_image
            
            image_to_save = image if image is not None else self.current_image
            if image_to_save is None:
                print("No image to save")
                return False
            
            save_image(file_path, image_to_save)
            return True
        except Exception as e:
            print(f"Error saving image: {e}")
            return False
    
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
    
    def get_current_path(self) -> Optional[str]:
        """Get the path of the current image."""
        return self.current_image_path
    
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
    
    def enable_mask_editing(self, mask_index: int, global_params: Dict[str, Any]) -> bool:
        """Enable editing for a specific mask."""
        if 0 <= mask_index < len(self.layer_masks):
            # Save global parameters if not already saved
            if not self.mask_editing_enabled:
                self.global_parameters = global_params.copy()
            
            # Save current mask parameters if switching masks
            if self.mask_editing_enabled and self.current_mask_index != mask_index:
                self.save_mask_parameters(self.current_mask_index, global_params)
            
            self.mask_editing_enabled = True
            self.current_mask_index = mask_index
            return True
        return False
    
    def disable_mask_editing(self, current_params: Dict[str, Any]) -> Dict[str, Any]:
        """Disable mask editing and return global parameters."""
        if self.mask_editing_enabled and self.current_mask_index >= 0:
            # Save current mask parameters
            self.save_mask_parameters(self.current_mask_index, current_params)
        
        self.mask_editing_enabled = False
        self.current_mask_index = -1
        
        # Return global parameters or current params if no global saved
        return self.global_parameters if self.global_parameters else current_params
    
    def save_mask_parameters(self, mask_index: int, parameters: Dict[str, Any]) -> None:
        """Save parameters for a specific mask."""
        if 0 <= mask_index < len(self.layer_masks):
            self.mask_parameters[mask_index] = parameters.copy()
    
    def load_mask_parameters(self, mask_index: int) -> Optional[Dict[str, Any]]:
        """Load parameters for a specific mask."""
        return self.mask_parameters.get(mask_index)
    
    def get_mask_count(self) -> int:
        """Get the number of masks."""
        return len(self.layer_masks)
    
    def get_masks(self) -> List[Dict[str, Any]]:
        """Get all masks."""
        return self.layer_masks
    
    def get_mask_names(self) -> List[str]:
        """Get all mask names."""
        return self.mask_names


class SegmentationService:
    """Service for managing segmentation operations."""
    
    def __init__(self):
        self.segmenter = None
        self.resource_manager = ResourceManager()
        self.segmentation_mode = False
        self.pending_selection = None
    
    def get_segmenter(self):
        """Get or create segmenter instance with memory-efficient approach."""
        if self.segmenter is None:
            try:
                # Import here to avoid circular dependencies
                from ui.segmentation import ImageSegmenter
                
                print("Creating ImageSegmenter instance...")
                
                # Check available memory first
                memory_info = MemoryManager.get_device_info()
                print(f"Available memory: {memory_info}")
                
                # Select device based on memory requirements
                # SAM model requires about 4GB of GPU memory
                device = MemoryManager.select_optimal_device(min_gpu_memory_gb=4.0)
                
                # If GPU doesn't have enough memory, force CPU mode
                if device == "cuda" and memory_info['free_mb'] < 3000:  # 3GB threshold
                    print(f"GPU has only {memory_info['free_mb']:.1f}MB free, forcing CPU mode")
                    device = "cpu"
                
                print(f"Selected device: {device}")
                
                # Clear memory before creating segmenter
                MemoryManager.clear_cuda_cache()
                
                self.segmenter = ImageSegmenter(device=device)
                
                # Register with resource manager
                self.resource_manager.register_model(
                    "sam_segmenter", 
                    self.segmenter.model, 
                    device
                )
                
                print(f"ImageSegmenter instance created successfully on {device}")
                
            except Exception as e:
                error_msg = ErrorHandler.handle_memory_error(e, "segmenter creation")
                print(error_msg)
                
                # Try fallback to CPU if GPU failed
                try:
                    print("Attempting CPU fallback for segmenter...")
                    MemoryManager.clear_cuda_cache()
                    self.segmenter = ImageSegmenter(device="cpu")
                    print("ImageSegmenter created successfully on CPU as fallback")
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
            print("Segmenter instance reset")
    
    def enable_segmentation_mode(self) -> bool:
        """Enable segmentation mode."""
        self.segmentation_mode = True
        self.pending_selection = None
        return True
    
    def disable_segmentation_mode(self) -> None:
        """Disable segmentation mode."""
        self.segmentation_mode = False
        self.pending_selection = None
    
    def set_pending_selection(self, selection: Dict[str, Any]) -> None:
        """Set pending segmentation selection."""
        self.pending_selection = selection
    
    def get_pending_selection(self) -> Optional[Dict[str, Any]]:
        """Get pending segmentation selection."""
        return self.pending_selection
    
    def validate_segmentation_requirements(self, crop_rotate_ui=None) -> bool:
        """Validate that segmentation requirements are met."""
        if not crop_rotate_ui or not hasattr(crop_rotate_ui, "original_image"):
            print("No image available for segmentation.")
            return False
            
        return True
    
    def transform_texture_coordinates_to_image(self, box, crop_rotate_ui):
        """Transform texture coordinates to image coordinates."""
        try:
            if not crop_rotate_ui or crop_rotate_ui.original_image is None:
                print("DEBUG: transform_texture_coordinates_to_image - No crop_rotate_ui or original_image")
                return None
                
            image = crop_rotate_ui.original_image
            h, w = image.shape[:2]
            
            # Calculate the texture offsets where the image is centered within the texture
            texture_w = crop_rotate_ui.texture_w
            texture_h = crop_rotate_ui.texture_h
            offset_x = (texture_w - w) // 2
            offset_y = (texture_h - h) // 2
            
            print(f"DEBUG: Image dimensions: {w}x{h}, Texture: {texture_w}x{texture_h}, Offsets: ({offset_x}, {offset_y})")
            print(f"DEBUG: Input box: {box}, types: {[type(x) for x in box]}")
            
            # Convert all box coordinates to Python scalars first
            box_scalars = []
            for i, coord in enumerate(box):
                try:
                    if hasattr(coord, 'item'):  # numpy scalar
                        scalar_val = coord.item()
                    else:
                        scalar_val = float(coord)
                    box_scalars.append(scalar_val)
                    print(f"DEBUG: Converted box[{i}]: {coord} -> {scalar_val}")
                except Exception as e:
                    print(f"DEBUG: Error converting box[{i}] = {coord}: {e}")
                    return None
            
            # Convert box coordinates from texture space to image space
            x1 = max(0, min(w-1, int(box_scalars[0] - offset_x)))
            y1 = max(0, min(h-1, int(box_scalars[1] - offset_y)))
            x2 = max(0, min(w-1, int(box_scalars[2] - offset_x)))
            y2 = max(0, min(h-1, int(box_scalars[3] - offset_y)))
            
            scaled_box = [x1, y1, x2, y2]
            print(f"DEBUG: Computed scaled_box: {scaled_box}")
            
            # Ensure all coordinates are integers (not numpy types)
            scaled_box = [int(coord) for coord in scaled_box]
            print(f"DEBUG: Final integer scaled_box: {scaled_box}")
            
            # Validate box dimensions
            if scaled_box[2] <= scaled_box[0] or scaled_box[3] <= scaled_box[1]:
                print(f"DEBUG: Invalid box dimensions after transformation: {scaled_box}")
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
                print(f"DEBUG: Created minimal valid box: {scaled_box}")
            
            print(f"DEBUG: Final transformed box: {scaled_box}")
            return scaled_box
            
        except Exception as e:
            print(f"DEBUG: Exception in transform_texture_coordinates_to_image: {e}")
            import traceback
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
        print(f"DEBUG: Starting segmentation confirmation, mode={self.segmentation_mode}")
        
        if not self.segmentation_mode:
            return False, "Segmentation mode is not active"
            
        if not pending_box:
            print("DEBUG: No pending box, creating default")
            # Try to create a default box
            pending_box = self.create_default_bounding_box(crop_rotate_ui)
            if not pending_box:
                return False, "No area selected and could not create default selection"
        
        print(f"DEBUG: Pending box: {pending_box}")
        
        # Validate box dimensions
        if pending_box["w"] < 10 or pending_box["h"] < 10:
            return False, f"Selection area too small: {pending_box['w']}x{pending_box['h']} (minimum 10x10)"
        
        # Box format: [x1, y1, x2, y2]
        box = [pending_box["x"], pending_box["y"], 
               pending_box["x"] + pending_box["w"], 
               pending_box["y"] + pending_box["h"]]
        print(f"DEBUG: Box coordinates: {box}")
        
        # Transform coordinates
        print("DEBUG: Transforming coordinates...")
        scaled_box = self.transform_texture_coordinates_to_image(box, crop_rotate_ui)
        if not scaled_box:
            return False, "Could not transform coordinates"
        
        print(f"DEBUG: Scaled box: {scaled_box}, types: {[type(x) for x in scaled_box]}")
        
        # Validate transformed box
        try:
            print("DEBUG: Starting coordinate validation...")
            # Ensure all coordinates are scalar values and finite
            for i, coord in enumerate(scaled_box):
                print(f"DEBUG: Validating coord {i}: {coord} (type: {type(coord)})")
                if not isinstance(coord, (int, float, np.integer, np.floating)):
                    return False, f"Invalid coordinate type: {type(coord)}"
                # Convert numpy scalars to Python scalars for safe comparison
                coord_val = float(coord) if hasattr(coord, 'item') else coord
                print(f"DEBUG: Coord {i} value: {coord_val}")
                if not np.isfinite(coord_val):
                    return False, f"Non-finite coordinate: {coord_val}"
            print("DEBUG: Coordinate validation passed")
        except Exception as e:
            print(f"DEBUG: Coordinate validation exception: {e}")
            return False, f"Coordinate validation error: {str(e)}"
        
        print("DEBUG: Checking box size...")
        try:
            if scaled_box[2] - scaled_box[0] < 10 or scaled_box[3] - scaled_box[1] < 10:
                return False, "Selection too small relative to image"
        except Exception as e:
            print(f"DEBUG: Box size check exception: {e}")
            return False, f"Box size validation error: {str(e)}"
        
        # Perform segmentation
        print("DEBUG: Starting segmentation...")
        try:
            masks, mask_names = app_service.perform_box_segmentation(scaled_box)
            if not masks or len(masks) == 0:
                return False, "No objects found in selection"
            
            return True, f"Created {len(masks)} total masks"
        except Exception as e:
            print(f"DEBUG: Segmentation exception: {e}")
            return False, f"Error: {str(e)}"


class ApplicationService:
    """Main application service that coordinates all other services."""
    
    def __init__(self):
        self.image_service = ImageService()
        self.mask_service = MaskService()
        self._segmentation_service = None  # Lazy-loaded to avoid auto-initialization
        self.ui_state = {}
    
    @property
    def segmentation_service(self):
        """Lazy-load segmentation service only when explicitly requested."""
        if self._segmentation_service is None:
            print("🔄 Initializing segmentation service on demand...")
            self._segmentation_service = SegmentationService()
            print("✓ Segmentation service ready")
        return self._segmentation_service
    
    def initialize(self):
        """Initialize the application services."""
        print("Application services initialized")
    
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
                crop_ui = self.image_service.create_crop_rotate_ui(image, processor)
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
    
    def get_image_service(self) -> ImageService:
        """Get the image service."""
        return self.image_service
    
    def get_mask_service(self) -> MaskService:
        """Get the mask service."""
        return self.mask_service
    
    def get_segmentation_service(self) -> SegmentationService:
        """Get the segmentation service."""
        return self.segmentation_service
    
    def setup_ui(self):
        """Setup the main UI components using modular architecture."""
        from ui.tool_panel_modular import ModularToolPanel
        from ui.main_window_production import ProductionMainWindow
        
        # Create the modular tool panel
        self.tool_panel = ModularToolPanel(
            update_callback=self.update_image_callback,
            mask_service=self.mask_service,
            app_service=self
        )
        
        # Create main window with updated callback structure
        self.main_window = ProductionMainWindow(self)
        
        # Setup the main window
        self.main_window.setup()
        
        # Replace the original tool panel with our modular one
        self.main_window.tool_panel = self.tool_panel
        self.tool_panel.setup()
    
    def update_image_callback(self):
        """Callback for updating the image when parameters change."""
        if not self.image_service.image_processor or not self.image_service.crop_rotate_ui:
            return
        
        # Get current tool parameters
        params = self.tool_panel.get_all_parameters() if hasattr(self, 'tool_panel') else {}
        
        # Apply parameters to processor
        processor = self.image_service.image_processor
        processor.exposure = params.get('exposure', 0)
        processor.illumination = params.get('illumination', 0.0)
        processor.contrast = params.get('contrast', 1.0)
        processor.shadow = params.get('shadow', 0)
        processor.highlights = params.get('highlights', 0)
        processor.whites = params.get('whites', 0)
        processor.blacks = params.get('blacks', 0)
        processor.saturation = params.get('saturation', 1.0)
        processor.texture = params.get('texture', 0)
        processor.grain = params.get('grain', 0)
        processor.temperature = params.get('temperature', 0)
        
        # Apply all edits
        updated_image = processor.apply_all_edits()
        
        # Apply curves if available
        curves_data = params.get('curves', None)
        if curves_data:
            if isinstance(curves_data, dict) and 'curves' in curves_data:
                curves = curves_data['curves']
                interpolation_mode = curves_data.get('interpolation_mode', 'Linear')
            else:
                curves = curves_data
                interpolation_mode = 'Linear'
            
            updated_image = processor.apply_rgb_curves(updated_image, curves, interpolation_mode)
        
        # Update crop/rotate UI
        crop_ui = self.image_service.crop_rotate_ui
        crop_ui.original_image = updated_image
        crop_ui.update_image(None, None, None)
        
        # Update histogram if tool panel exists
        if hasattr(self, 'tool_panel') and hasattr(self.tool_panel, 'update_histogram'):
            self.tool_panel.update_histogram(updated_image)
    
    def show_load_dialog(self):
        """Show load dialog callback."""
        if safe_item_check("file_dialog_load"):
            dpg.show_item("file_dialog_load")
    
    def show_save_dialog(self):
        """Show save dialog callback."""
        if safe_item_check("file_dialog_save"):
            dpg.show_item("file_dialog_save")
    
    def save_current_image(self, file_path: str) -> bool:
        """Save the current processed image."""
        if not self.image_service.image_processor or not self.image_service.crop_rotate_ui:
            print("No image to save.")
            return False
        
        try:
            # Get rotation angle and crop rectangle
            angle = dpg.get_value("rotation_slider") if safe_item_check("rotation_slider") else 0
            crop_ui = self.image_service.crop_rotate_ui
            
            offset_x = (crop_ui.texture_w - crop_ui.orig_w) // 2
            offset_y = (crop_ui.texture_h - crop_ui.orig_h) // 2
            
            rx = int(crop_ui.user_rect["x"] - offset_x)
            ry = int(crop_ui.user_rect["y"] - offset_y)
            rw = int(crop_ui.user_rect["w"])
            rh = int(crop_ui.user_rect["h"])
            
            # Apply crop and rotation
            cropped = self.image_service.image_processor.crop_rotate_flip(
                self.image_service.image_processor.current, 
                (rx, ry, rw, rh), 
                angle
            )
            
            # Save the image
            return self.image_service.save_image(file_path, cropped)
        except Exception as e:
            print(f"Error saving image: {e}")
            return False
    
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
