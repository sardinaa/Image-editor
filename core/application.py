"""
Core application service layer.
Centralizes business logic and coordinates between different subsystems.
"""
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from pathlib import Path
import dearpygui.dearpygui as dpg

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
        """Get or create segmenter instance."""
        if self.segmenter is None:
            try:
                # Import here to avoid circular dependencies
                from ui.segmentation import ImageSegmenter
                
                print("Creating ImageSegmenter instance...")
                device = MemoryManager.select_optimal_device()
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
        from ui.main_window import MainWindow
        
        # Create the modular tool panel
        self.tool_panel = ModularToolPanel(
            update_callback=self.update_image_callback,
            mask_service=self.mask_service,
            app_service=self
        )
        
        # Create main window with updated callback structure
        self.main_window = MainWindow(
            None,  # crop_rotate_ui - will be set when image is loaded
            self.update_image_callback,
            self.show_load_dialog,
            self.show_save_dialog
        )
        
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
