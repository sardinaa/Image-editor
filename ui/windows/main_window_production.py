#!/usr/bin/env python3
import dearpygui.dearpygui as dpg
import cv2
import numpy as np
import traceback
import threading
from typing import Optional

from core.application import ApplicationService
from ui.components.tool_panel_modular import ModularToolPanel
from ui.interactions.crop_rotate import CropRotateUI
from ui.renderers.bounding_box_renderer import BoundingBoxRenderer, BoundingBox
from ui.renderers.mask_overlay_renderer import MaskOverlayRenderer
from core.services.display_service import DisplayService
from core.services.event_coordinator import EventCoordinator
from ui.interactions.event_handlers import EventHandlers
from core.services.export_service import ExportService
from ui.layout_manager import LayoutManager
from utils.ui_helpers import safe_item_check
from utils.memory_utils import MemoryManager


class ProductionMainWindow:
    
    def __init__(self, app_service: ApplicationService):
        self.app_service = app_service
        self.memory_manager = MemoryManager()
        
        # UI component references
        self.tool_panel: Optional[ModularToolPanel] = None
        self.crop_rotate_ui: Optional[CropRotateUI] = None
        self.bbox_renderer: Optional[BoundingBoxRenderer] = None
        self.segmentation_bbox_renderer: Optional[BoundingBoxRenderer] = None
        self.mask_overlay_renderer: Optional[MaskOverlayRenderer] = None
        
        # Services
        self.display_service = DisplayService()
        self.event_coordinator = EventCoordinator(app_service, self)
        self.export_service = ExportService(app_service)
        self.layout_manager = LayoutManager(self)
        self.event_handlers = EventHandlers(app_service)
        
        # Window tags and layout  
        self.window_tag = "main_window"
        self.central_panel_tag = "central_panel"
        self.right_panel_tag = "right_panel"
        self.image_plot_tag = "image_plot"
        # Unique axis tags to avoid conflicts with CropRotateUI
        self.x_axis_tag = "main_x_axis"
        self.y_axis_tag = "main_y_axis"
        
        # Segmentation state (UI-specific only)
        self.segmentation_mode = False
        self.segmentation_texture = None
        self.pending_segmentation_box = None

        self.viewport_width = 0
        self.viewport_height = 0
        self.available_height = 0
        self.right_panel_width = 0
        self.central_panel_width = 0
        self.menu_bar_height = 0
    
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
        
        if not dpg.does_item_exist("main_mouse_handlers"):
            with dpg.handler_registry(tag="main_mouse_handlers"):
                dpg.add_mouse_down_handler(callback=self.event_handlers.on_mouse_down, button=dpg.mvMouseButton_Left)
                dpg.add_mouse_drag_handler(callback=self.event_handlers.on_mouse_drag, button=dpg.mvMouseButton_Left)
                dpg.add_mouse_release_handler(callback=self.event_handlers.on_mouse_release, button=dpg.mvMouseButton_Left)
                dpg.add_mouse_wheel_handler(callback=self.event_handlers.on_mouse_wheel)

                dpg.add_key_down_handler(callback=self.event_handlers.on_key_press)
    
    def _open_image(self):
        """Open image file dialog."""
        dpg.show_item("file_open_dialog")

    def _export_image(self):
        """Export/save image file dialog."""
        dpg.show_item("export_modal")

    def _reset_all_processing(self):
        """Reset all processing parameters."""
        if self.tool_panel:
            self.tool_panel.reset_all_parameters()

    def _clear_all_masks(self):
        """Clear all masks via application service."""
        if self.app_service:
            self.app_service.clear_all_masks()
            
            if self.tool_panel and hasattr(self.tool_panel, 'masks_panel'):
                self.tool_panel.masks_panel.update_masks([], [])
            
            self.update_mask_overlays([])

    ##############################################
    # MAKE THIS IN THE SETUP
    ##############################################
    def set_crop_rotate_ui(self, crop_rotate_ui):
        """Set the crop rotate UI instance and initialize segmentation components."""
        self.crop_rotate_ui = crop_rotate_ui

        self._create_image_series()

        self.crop_rotate_ui.update_image(None, None, None)
        
        self._initialize_segmentation_bbox_renderer()
        
        self.mask_overlay_renderer = MaskOverlayRenderer(self.x_axis_tag, self.y_axis_tag)
        
        # Configure performance settings for the mask overlay renderer
        from utils.performance_config import PerformanceConfig
        perf_config = PerformanceConfig.get_optimized_settings()
        self.mask_overlay_renderer.set_performance_settings(
            max_visible=perf_config['max_visible_overlays'],
            throttle_ms=perf_config['overlay_delay'],
            progressive=perf_config['progressive_loading']
        )
    
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
        
        self.segmentation_bbox_renderer.set_callbacks(
            on_change=self._on_segmentation_bbox_change,
            on_end_drag=self._on_segmentation_bbox_end_drag
        )
    
    def _on_segmentation_bbox_change(self, bbox: BoundingBox) -> None:
        """Called when segmentation bounding box changes during drag."""
        if self.segmentation_mode and self.segmentation_texture is not None:
            # Update the pending box during drag for a more responsive experience
            if bbox and bbox.width > 5 and bbox.height > 5:
                self.pending_segmentation_box = bbox.to_dict()
                
                self._update_status(f"Selection: {int(bbox.width)}x{int(bbox.height)}")
            
            self._update_segmentation_overlay()
    
    def _on_segmentation_bbox_end_drag(self, bbox: BoundingBox) -> None:
        """Called when segmentation bounding box drag ends."""
        if self.segmentation_mode and self.segmentation_texture is not None:
            if bbox and bbox.width > 10 and bbox.height > 10:
                self.pending_segmentation_box = bbox.to_dict()

                self._update_status(f"Selection area: {int(bbox.width)}x{int(bbox.height)}. Press Confirm to segment.")
                self._update_segmentation_overlay()
                
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
    
    def _update_segmentation_overlay(self):
        """Update the segmentation overlay with the current bounding box"""
        if not self.segmentation_bbox_renderer or self.segmentation_texture is None:
            return

        blended = self.segmentation_bbox_renderer.render_on_texture(self.segmentation_texture)
        
        # Update the texture in DearPyGUI
        if self.crop_rotate_ui and self.crop_rotate_ui.texture_tag:
            texture_data = blended.flatten().astype(np.float32) / 255.0
            dpg.set_value(self.crop_rotate_ui.texture_tag, texture_data)
    
    def enable_segmentation_mode(self):
        """Enable segmentation mode with real-time bounding box"""
        if not self.crop_rotate_ui or not self.segmentation_bbox_renderer:
            self._update_status("Error: Segmentation not available")
            return False
            
        if safe_item_check("crop_mode") and dpg.get_value("crop_mode"):
            dpg.set_value("crop_mode", False)
            if self.tool_panel and hasattr(self.tool_panel, 'toggle_crop_mode'):
                self.tool_panel.toggle_crop_mode(None, None, None)
        
        self.pending_segmentation_box = None
        if self.segmentation_bbox_renderer:
            self.segmentation_bbox_renderer.reset()
        
        self.segmentation_mode = True
        
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
        
        self._update_status("Segmentation mode: click and drag to select area")
            
        return True
    
    def disable_segmentation_mode(self):
        """Disable segmentation mode"""
        self.segmentation_mode = False
        self.segmentation_texture = None
        self.pending_segmentation_box = None
        
        if self.app_service:
            self.app_service.segmentation_service.disable_segmentation_mode()
        
        if self.segmentation_bbox_renderer:
            self.segmentation_bbox_renderer.reset()
        
        if self.crop_rotate_ui:
            self.crop_rotate_ui.update_image(None, None, None)
        
        self._update_status("")
    
    def _create_segmentation_texture(self):
        """Create segmentation texture from current image"""
        if not self.crop_rotate_ui or self.crop_rotate_ui.original_image is None:
            return
            
        image = self.crop_rotate_ui.original_image
        h, w = image.shape[:2]
        texture = np.full((self.crop_rotate_ui.texture_h, self.crop_rotate_ui.texture_w, 4), 
                         [37,37,38,255], dtype=np.uint8)
        
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
    
    def _update_status(self, message: str):
        """Update status text if it exists."""
        if dpg.does_item_exist("status_text"):
            dpg.configure_item("status_text", default_value=message)

            # Clear the status text after 5 seconds
            def clear_status():
                if dpg.does_item_exist("status_text"):
                    dpg.configure_item("status_text", default_value="")

            threading.Timer(5.0, clear_status).start()
    
    def cancel_segmentation_selection(self):
        """Cancel the current segmentation selection"""
        self.disable_segmentation_mode()
        
        # Update tool panel to reflect the change
        if self.tool_panel:
            self.tool_panel.set_segmentation_mode(False)

    def show_selected_mask(self, selected_index):
        """Show only the selected mask and hide others"""
        if not self.mask_overlay_renderer:
            return
            
        if not self.app_service:
            return
            
        # Get masks from service
        masks = self.app_service.get_mask_service().get_masks()
        if masks:
            self.mask_overlay_renderer.show_selected_mask(selected_index, len(masks))
    
    def show_selected_masks(self, selected_indices):
        """Show multiple selected masks and hide others"""
        if not self.mask_overlay_renderer:
            return
            
        if not self.app_service:
            return
            
        # Get masks from service
        masks = self.app_service.get_mask_service().get_masks()
        if masks:
            self.mask_overlay_renderer.show_selected_masks(selected_indices, len(masks))
    
    def update_mask_overlays(self, masks):
        """Update the visual mask overlays on the image"""
        if self.mask_overlay_renderer:
            self.mask_overlay_renderer.update_mask_overlays(masks, self.crop_rotate_ui)
    
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
            if self.app_service:
                try:
                    # Use service to clean up segmentation resources
                    segmentation_service = self.app_service.get_segmentation_service()
                    if segmentation_service:
                        segmentation_service.cleanup_memory()
                except Exception as e:
                    print(f"Error cleaning up segmentation service: {e}")

            if self.bbox_renderer:
                self.bbox_renderer = None
            
            if self.tool_panel:
                try:
                    self.tool_panel.cleanup()
                except:
                    pass

            if self.crop_rotate_ui:
                self.crop_rotate_ui = None
            
            if self.display_service:
                self.display_service.cleanup()

            if self.event_coordinator:
                self.event_coordinator.cleanup()
            
            if self.memory_manager:
                self.memory_manager.cleanup_gpu_memory()
            
        except Exception as e:
            print(f"⚠️  Error during production main window cleanup: {e}")

    def _on_parameter_change(self, sender, app_data, user_data):
        """Handle parameter changes from tool panel - delegate to event coordinator."""
        if hasattr(self, 'event_coordinator') and self.event_coordinator:
            self.event_coordinator.handle_parameter_change(sender, app_data, user_data)
