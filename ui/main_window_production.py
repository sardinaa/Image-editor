#!/usr/bin/env python3
"""
Production Main Window - Refactored using Service Layer Architecture
Replaces the original 1,093-line main_window.py with a modular, maintainable version.
"""

import dearpygui.dearpygui as dpg
import cv2
import numpy as np
import traceback
import os
from typing import Optional, List, Tuple

from core.application import ApplicationService
from ui.tool_panel_modular import ModularToolPanel
from ui.crop_rotate import CropRotateUI
# Removed ImageSegmenter import - segmentation now handled by SegmentationService
from ui.bounding_box_renderer import BoundingBoxRenderer, BoundingBox
from ui.mask_overlay_renderer import MaskOverlayRenderer
from ui.services.display_service import DisplayService
from ui.services.event_coordinator import EventCoordinator
from ui.services.export_service import ExportService
from ui.layout_manager import LayoutManager
from utils.ui_helpers import safe_item_check
from utils.memory_utils import MemoryManager


class ProductionMainWindow:
    """
    Production-ready main window using service layer architecture.
    
    This replaces the original main_window.py with a cleaner, more maintainable implementation
    that leverages the ApplicationService for business logic separation.
    
    CLEANED UP:
    - Removed redundant delegation methods that just call other services
    - Simplified status update handling with _update_status() helper
    - Removed unnecessary wrapper methods for menu callbacks
    - Consolidated error handling in _validate_segmentation_requirements()
    - Direct access to services instead of unnecessary proxy methods
    """
    
    def __init__(self, app_service: ApplicationService):
        self.app_service = app_service
        self.memory_manager = MemoryManager()
        
        # UI component references
        self.tool_panel: Optional[ModularToolPanel] = None
        self.crop_rotate_ui: Optional[CropRotateUI] = None
        # Removed self.segmenter - segmentation now handled by SegmentationService
        self.bbox_renderer: Optional[BoundingBoxRenderer] = None
        self.segmentation_bbox_renderer: Optional[BoundingBoxRenderer] = None
        self.mask_overlay_renderer: Optional[MaskOverlayRenderer] = None
        
        # Services
        self.display_service = DisplayService()
        self.event_coordinator = EventCoordinator(app_service, self)
        self.export_service = ExportService(app_service)
        self.layout_manager = LayoutManager(self)
        
        # Window tags and layout  
        self.window_tag = "main_window"
        self.central_panel_tag = "central_panel"
        self.right_panel_tag = "right_panel"
        self.image_plot_tag = "image_plot"
        # Unique axis tags to avoid conflicts with CropRotateUI
        self.x_axis_tag = "main_x_axis"
        self.y_axis_tag = "main_y_axis"
        
        # REMOVED: Legacy box selection state variables
        # Modern segmentation uses BoundingBoxRenderer instead
        
        # Segmentation state (UI-specific only)
        self.segmentation_mode = False
        self.segmentation_texture = None
        self.pending_segmentation_box = None
        # Removed layer_masks and mask_names - now managed by MaskService
    
    # Direct access to event_coordinator for display state - no wrapper needed
    
    def setup(self) -> None:
        """Setup the main window and all components."""
        try:
            self._calculate_layout()
            self._create_main_window()
            self._setup_handlers()
            
        except Exception as e:
            print(f"Error setting up main window: {e}")
            raise
    
    def _calculate_layout(self) -> None:
        """Calculate layout dimensions and store for backward compatibility."""
        dims = self.layout_manager.get_layout_dimensions()
        
        # Store dimensions for backward compatibility
        self.viewport_width = dims['viewport_width']
        self.viewport_height = dims['viewport_height'] 
        self.available_height = dims['available_height']
        self.right_panel_width = dims['tool_panel_width']
        self.central_panel_width = dims['central_panel_width']
        self.menu_bar_height = self.layout_manager.layout_config['menu_bar_height']
    
    def _create_main_window(self) -> None:
        """Create the main window structure."""
        self.layout_manager.setup_main_layout()
    
    def _setup_handlers(self) -> None:
        """Setup centralized event handlers through EventHandlers."""
        # Import here to avoid circular imports
        from ui.event_handlers import EventHandlers
        
        # Create centralized event handlers
        self.event_handlers = EventHandlers(self.app_service)
        
        if not dpg.does_item_exist("main_mouse_handlers"):
            with dpg.handler_registry(tag="main_mouse_handlers"):
                dpg.add_mouse_down_handler(callback=self.event_handlers.on_mouse_down)
                dpg.add_mouse_drag_handler(callback=self.event_handlers.on_mouse_drag)
                dpg.add_mouse_release_handler(callback=self.event_handlers.on_mouse_release)
                dpg.add_mouse_wheel_handler(callback=self.event_handlers.on_mouse_wheel)
                
                # Add keyboard handlers as well
                dpg.add_key_down_handler(callback=self.event_handlers.on_key_press)
    
    # Remove simple delegation methods - access services directly
    # _on_parameter_change -> use event_coordinator.handle_parameter_change directly
    # _collect_current_parameters -> use event_coordinator._collect_current_parameters directly  
    # _check_for_automatic_mask_reset -> use event_coordinator._check_for_automatic_mask_reset directly
    # _update_ui_after_mask_change -> use event_coordinator.update_ui_after_mask_change directly
    
    # Menu Callbacks - simplified to remove redundant wrappers
    def _open_image(self):
        """Open image file dialog."""
        import dearpygui.dearpygui as dpg
        dpg.show_item("file_open_dialog")
    
    def _export_image(self):
        """Export/save image file dialog."""
        import dearpygui.dearpygui as dpg
        dpg.show_item("file_save_dialog")
    
    def _reset_all_processing(self):
        """Reset all processing parameters."""
        if self.tool_panel:
            self.tool_panel.reset_all_parameters()
    
    def _clear_all_masks(self):
        """Clear all masks via application service."""
        if self.app_service:
            self.app_service.clear_all_masks()
            
            # Update UI
            if self.tool_panel and hasattr(self.tool_panel, 'masks_panel'):
                self.tool_panel.masks_panel.update_masks([], [])
            
            # Update mask overlays
            self.update_mask_overlays([])
    
    def _auto_segment(self):
        """Trigger automatic segmentation via application service."""
        if self.app_service:
            masks, names = self.app_service.perform_automatic_segmentation()
            
            # Update UI
            if self.tool_panel and hasattr(self.tool_panel, 'masks_panel'):
                self.tool_panel.masks_panel.update_masks(masks, names)
            
            # Update mask overlays
            self.update_mask_overlays(masks)
    
    # Image Display Management
    def _update_image_display(self):
        """Update the image display with current processed image."""
        try:
            processed_image = self.app_service.image_service.get_processed_image()
            if processed_image is None:
                return
            
            self.display_service.update_image_display(processed_image, self.crop_rotate_ui)
            
        except Exception as e:
            print(f"Error updating image display: {e}")
            traceback.print_exc()
    
    # Remove simple delegation methods - use services directly:
    # _update_texture -> use display_service.update_texture directly
    # _update_axis_limits_to_image -> use display_service.update_axis_limits_to_image directly

    # Segmentation Handlers
    def _handle_segmentation_mouse_down(self, app_data):
        """Handle mouse down events for segmentation mode."""
        if not self.segmentation_bbox_renderer:
            return False
            
        if not dpg.is_item_hovered(self.image_plot_tag):
            return False
            
        # Reset any pending box when starting a new one
        if app_data == 0:  # Left mouse button
            self.pending_segmentation_box = None
            
        result = self.segmentation_bbox_renderer.on_mouse_down(None, app_data)
        return result
    
    # REMOVED: Legacy box selection system
    # This conflicted with the modern BoundingBoxRenderer implementation
    # Modern segmentation uses enable_segmentation_mode() and BoundingBoxRenderer
    
    # Public Interface
    def load_image(self, image_path: str) -> bool:
        """
        Load an image using the original successful pattern.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            print(f"🔄 Loading image: {image_path}")
            
            image = self.app_service.load_image(image_path)
            if image is not None:
                print(f"✅ Image loaded successfully: {image.shape}")
                # Don't create CropRotateUI here - let main.py handle it like original
                return True
            else:
                print("❌ Failed to load image")
                return False
            
        except Exception as e:
            print(f"💥 Error loading image: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def set_crop_rotate_ui(self, crop_rotate_ui):
        """Set the crop rotate UI instance and initialize segmentation components."""
        self.crop_rotate_ui = crop_rotate_ui
        
        # Create the image series to display the image
        self._create_image_series()
        
        # Update the crop rotate UI to refresh the display
        self.crop_rotate_ui.update_image(None, None, None)
        
        # Initialize segmentation bounding box renderer
        self._initialize_segmentation_bbox_renderer()
        
        # Initialize mask overlay renderer
        self.mask_overlay_renderer = MaskOverlayRenderer(self.x_axis_tag, self.y_axis_tag)
        
        print("CropRotateUI set and segmentation components initialized")
    
    # Removed get_segmenter() - segmenter management now handled by SegmentationService
    
    def _initialize_segmentation_bbox_renderer(self):
        """Initialize the segmentation bounding box renderer"""
        if not self.crop_rotate_ui:
            return
            
        self.segmentation_bbox_renderer = BoundingBoxRenderer(
            texture_width=self.crop_rotate_ui.texture_w,
            texture_height=self.crop_rotate_ui.texture_h,
            panel_id=self.central_panel_tag,
            min_size=20,
            handle_size=20,
            handle_threshold=50
        )
        
        # Set up callbacks
        self.segmentation_bbox_renderer.set_callbacks(
            on_change=self._on_segmentation_bbox_change,
            on_start_drag=self._on_segmentation_bbox_start_drag,
            on_end_drag=self._on_segmentation_bbox_end_drag
        )
    
    def _on_segmentation_bbox_change(self, bbox: BoundingBox) -> None:
        """Called when segmentation bounding box changes during drag."""
        if self.segmentation_mode and self.segmentation_texture is not None:
            # Update the pending box during drag for a more responsive experience
            if bbox and bbox.width > 5 and bbox.height > 5:
                self.pending_segmentation_box = bbox.to_dict()
                
                # Update status text with current dimensions
                self._update_status(f"Selection: {int(bbox.width)}x{int(bbox.height)}")
            
            self._update_segmentation_overlay()
    
    def _on_segmentation_bbox_start_drag(self, bbox: BoundingBox) -> None:
        """Called when segmentation bounding box drag starts."""
        pass  # No special handling needed for start
    
    def _on_segmentation_bbox_end_drag(self, bbox: BoundingBox) -> None:
        """Called when segmentation bounding box drag ends."""
        if self.segmentation_mode and self.segmentation_texture is not None:
            # Validate bounding box
            if bbox and bbox.width > 10 and bbox.height > 10:
                self.pending_segmentation_box = bbox.to_dict()
                # Update status text
                self._update_status(f"Selection area: {int(bbox.width)}x{int(bbox.height)}. Press Confirm to segment.")
                self._update_segmentation_overlay(force_update=True)
                
                # Auto-confirm the selection after a small delay
                # This helps users who might not realize they need to press the Confirm button
                if hasattr(self, 'tool_panel') and self.tool_panel:
                    try:
                        if hasattr(self.tool_panel, 'enable_confirm_button'):
                            self.tool_panel.enable_confirm_button()
                    except Exception as e:
                        print(f"Error enabling confirm button: {e}")
            else:
                self.pending_segmentation_box = None
                self._update_status("Selection too small (must be larger than 10x10)")
        else:
            self.pending_segmentation_box = None
    
    def _update_segmentation_overlay(self, force_update=False):
        """Update the segmentation overlay with the current bounding box"""
        if not self.segmentation_bbox_renderer or self.segmentation_texture is None:
            return
            
        # Render the bounding box on the texture
        blended = self.segmentation_bbox_renderer.render_on_texture(self.segmentation_texture)
        
        # Update the texture in DearPyGUI
        if self.crop_rotate_ui and self.crop_rotate_ui.texture_tag:
            texture_data = blended.flatten().astype(np.float32) / 255.0
            dpg.set_value(self.crop_rotate_ui.texture_tag, texture_data)
    
    def enable_segmentation_mode(self):
        """Enable segmentation mode with real-time bounding box"""
        print("Enabling segmentation mode...")
        
        if not self.crop_rotate_ui or not self.segmentation_bbox_renderer:
            print("Cannot enable segmentation mode: missing components")
            self._update_status("Error: Segmentation not available")
            return False
            
        # Disable crop mode if active
        if safe_item_check("crop_mode") and dpg.get_value("crop_mode"):
            dpg.set_value("crop_mode", False)
            if self.tool_panel and hasattr(self.tool_panel, 'toggle_crop_mode'):
                self.tool_panel.toggle_crop_mode(None, None, None)
        
        # Reset any existing segmentation state
        self.pending_segmentation_box = None
        if self.segmentation_bbox_renderer:
            self.segmentation_bbox_renderer.reset()
        
        self.segmentation_mode = True
        
        # Also enable segmentation mode in the service
        if self.app_service:
            self.app_service.segmentation_service.enable_segmentation_mode()
        
        # Store the current texture for overlay
        if hasattr(self.crop_rotate_ui, 'rotated_texture') and self.crop_rotate_ui.rotated_texture is not None:
            self.segmentation_texture = self.crop_rotate_ui.rotated_texture.copy()
        else:
            # Fallback: create texture from current image
            self._create_segmentation_texture()
        
        # Set bounds for the bounding box (limit to image area)
        if self.crop_rotate_ui.original_image is not None:
            h, w = self.crop_rotate_ui.original_image.shape[:2]
            offset_x = (self.crop_rotate_ui.texture_w - w) // 2
            offset_y = (self.crop_rotate_ui.texture_h - h) // 2
            
            bounds = BoundingBox(
                x=offset_x,
                y=offset_y,
                width=w,
                height=h
            )
            self.segmentation_bbox_renderer.set_bounds(bounds)
        
        # Update the status text
        self._update_status("Segmentation mode: click and drag to select area")
            
        return True
    
    def disable_segmentation_mode(self):
        """Disable segmentation mode"""
        self.segmentation_mode = False
        self.segmentation_texture = None
        self.pending_segmentation_box = None
        
        # Also disable segmentation mode in the service
        if self.app_service:
            self.app_service.segmentation_service.disable_segmentation_mode()
        
        if self.segmentation_bbox_renderer:
            self.segmentation_bbox_renderer.reset()
        
        # Refresh the display without segmentation overlay
        if self.crop_rotate_ui:
            self.crop_rotate_ui.update_image(None, None, None)
    
    def _create_segmentation_texture(self):
        """Create segmentation texture from current image"""
        if not self.crop_rotate_ui or self.crop_rotate_ui.original_image is None:
            return
            
        image = self.crop_rotate_ui.original_image
        h, w = image.shape[:2]
        
        # Create texture with gray background
        texture = np.full((self.crop_rotate_ui.texture_h, self.crop_rotate_ui.texture_w, 4), 
                         [100, 100, 100, 0], dtype=np.uint8)
        
        # Center the image in the texture
        offset_x = (self.crop_rotate_ui.texture_w - w) // 2
        offset_y = (self.crop_rotate_ui.texture_h - h) // 2
        
        # Convert image to RGBA if needed
        if image.shape[2] == 3:
            image_rgba = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)
        else:
            image_rgba = image
        
        # Place image in texture
        texture[offset_y:offset_y + h, offset_x:offset_x + w] = image_rgba
        self.segmentation_texture = texture
    
    # MOVED: Complex segmentation business logic to SegmentationService
    # - confirm_segmentation_selection() -> SegmentationService.confirm_segmentation_selection()
    # - _validate_segmentation_requirements() -> SegmentationService.validate_segmentation_requirements()
    # - Default bounding box creation logic -> SegmentationService.create_default_bounding_box()
    # - Coordinate transformation logic -> SegmentationService.transform_texture_coordinates_to_image()
    
    # Helper method to reduce repetitive status and error handling
    def _update_status(self, message: str):
        """Update status text if it exists."""
        if dpg.does_item_exist("status_text"):
            dpg.configure_item("status_text", default_value=message)
    
    def cancel_segmentation_selection(self):
        """Cancel the current segmentation selection"""
        self.disable_segmentation_mode()
        
        # Update tool panel to reflect the change
        if self.tool_panel:
            self.tool_panel.set_segmentation_mode(False)
    
    # MOVED: segment_current_image() and segment_with_box()
    # These complex business logic methods moved to SegmentationService
    # UI components should call app_service.perform_automatic_segmentation() directly
    
    def update_mask_overlays(self, masks):
        """Update the visual mask overlays on the image"""
        if self.mask_overlay_renderer:
            self.mask_overlay_renderer.update_mask_overlays(masks, self.crop_rotate_ui)
        else:
            print("MaskOverlayRenderer not initialized")
    
    def show_selected_mask(self, selected_index):
        """Show only the selected mask and hide others"""
        if not self.mask_overlay_renderer:
            print("MaskOverlayRenderer not initialized")
            return
            
        if not self.app_service:
            return
            
        # Get masks from service
        masks = self.app_service.get_mask_service().get_masks()
        if masks:
            self.mask_overlay_renderer.show_selected_mask(selected_index, len(masks))
    
    # REMOVED: Redundant mask management wrappers
    # These methods just called app_service methods with no added value
    # UI components should call app_service directly:
    # - clear_all_masks() -> app_service.clear_all_masks()
    # - delete_mask() -> app_service.delete_mask()
    # - rename_mask() -> app_service.rename_mask()
    
    def _cleanup_mask_overlay(self, mask_index):
        """Clean up the visual overlay for a specific mask"""
        if self.mask_overlay_renderer:
            self.mask_overlay_renderer.cleanup_mask_overlay(mask_index)
        else:
            # Fallback cleanup if renderer not available
            series_tag = f"mask_series_{mask_index}"
            if dpg.does_item_exist(series_tag):
                try:
                    dpg.configure_item(series_tag, show=False)
                    dpg.delete_item(series_tag)
                except Exception as e:
                    print(f"Error cleaning up mask overlay {mask_index}: {e}")
    
    # Remove redundant wrapper methods - use services directly:
    # _update_ui_after_mask_change -> use event_coordinator.update_ui_after_mask_change directly
    
    # Removed cleanup_segmenter_memory() - memory management now handled by SegmentationService
    # Removed reset_segmenter() - instance management now handled by SegmentationService
    
    # MOVED: Loading indicator management to MasksPanel
    # - _show_segmentation_loading() -> MasksPanel manages its own loading state
    # - _hide_segmentation_loading() -> MasksPanel manages its own loading state
    
    # REMOVED: toggle_box_selection_mode()
    # This was part of the legacy box selection system
    # Modern segmentation uses enable_segmentation_mode() instead
    
    def _create_image_series(self):
        """Create the image series to display the loaded image."""
        if self.display_service.create_image_series(self.crop_rotate_ui, self.image_plot_tag, self.y_axis_tag):
            self._update_axis_limits(initial=True)

    def _update_axis_limits(self, initial=False):
        """Update axis limits to properly display the image."""
        self.display_service.update_axis_limits(self.crop_rotate_ui, self.x_axis_tag, self.y_axis_tag, self.image_plot_tag, initial)
    
    def handle_resize(self):
        """Handle window resize events."""
        try:
            self.layout_manager.handle_resize()
            
            # Update local dimensions for backward compatibility
            dims = self.layout_manager.get_layout_dimensions()
            self.viewport_width = dims['viewport_width']
            self.viewport_height = dims['viewport_height'] 
            self.available_height = dims['available_height']
            self.right_panel_width = dims['tool_panel_width']
            self.central_panel_width = dims['central_panel_width']
            
            # Update histogram width in tool panel
            if self.tool_panel and hasattr(self.tool_panel, 'handle_resize'):
                self.tool_panel.handle_resize()
            
            # Update main window size
            if dpg.does_item_exist(self.window_tag):
                dpg.configure_item(self.window_tag, width=self.viewport_width, height=self.viewport_height)
            
        except Exception as e:
            print(f"Error handling resize: {e}")
            traceback.print_exc()
    
    def cleanup(self):
        """Cleanup application resources."""
        try:
            # Segmentation cleanup now handled by SegmentationService through app_service
            if self.app_service:
                try:
                    # Use service to clean up segmentation resources
                    segmentation_service = self.app_service.get_segmentation_service()
                    if segmentation_service:
                        segmentation_service.cleanup_memory()
                except Exception as e:
                    print(f"Error cleaning up segmentation service: {e}")
            
            # Cleanup bounding box renderer
            if self.bbox_renderer:
                self.bbox_renderer = None
            
            # Cleanup tool panel
            if self.tool_panel:
                try:
                    self.tool_panel.cleanup()
                except:
                    pass
            
            # Cleanup crop rotate UI
            if self.crop_rotate_ui:
                self.crop_rotate_ui = None
            
            # Cleanup DisplayService
            if self.display_service:
                self.display_service.cleanup()
            
            # Cleanup EventCoordinator
            if self.event_coordinator:
                self.event_coordinator.cleanup()
            
            # Memory cleanup
            if self.memory_manager:
                self.memory_manager.cleanup_gpu_memory()
                
            print("✓ Production main window cleanup completed")
            
        except Exception as e:
            print(f"⚠️  Error during production main window cleanup: {e}")
    
    def _toggle_segmentation_mode(self):
        """Toggle segmentation mode for manual box selection."""
        # Check current state through the masks panel
        if self.tool_panel and hasattr(self.tool_panel, 'masks_panel'):
            masks_panel = self.tool_panel.masks_panel
            if hasattr(masks_panel, '_toggle_segmentation_mode'):
                # Delegate to masks panel which handles the UI state
                masks_panel._toggle_segmentation_mode(None, None, None)
        else:
            # Fallback - toggle mode directly
            self.segmentation_mode = not getattr(self, 'segmentation_mode', False)
            if self.segmentation_mode:
                success = self.enable_segmentation_mode()
                if not success:
                    self.segmentation_mode = False
            else:
                self.disable_segmentation_mode()
    
    def _on_parameter_change(self, sender, app_data, user_data):
        """Handle parameter changes from tool panel - delegate to event coordinator."""
        if hasattr(self, 'event_coordinator') and self.event_coordinator:
            self.event_coordinator.handle_parameter_change(sender, app_data, user_data)
        else:
            # Fallback if event coordinator not available
            print("Warning: Event coordinator not available for parameter change handling")
