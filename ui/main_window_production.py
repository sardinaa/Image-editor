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
from ui.segmentation import ImageSegmenter
from ui.bounding_box_renderer import BoundingBoxRenderer, BoundingBox
from utils.ui_helpers import safe_item_check
from utils.memory_utils import MemoryManager


class ProductionMainWindow:
    """
    Production-ready main window using service layer architecture.
    
    This replaces the original main_window.py with a cleaner, more maintainable implementation
    that leverages the ApplicationService for business logic separation.
    """
    
    def __init__(self, app_service: ApplicationService):
        self.app_service = app_service
        self.memory_manager = MemoryManager()
        
        # UI component references
        self.tool_panel: Optional[ModularToolPanel] = None
        self.crop_rotate_ui: Optional[CropRotateUI] = None
        self.segmenter: Optional[ImageSegmenter] = None
        self.bbox_renderer: Optional[BoundingBoxRenderer] = None
        self.segmentation_bbox_renderer: Optional[BoundingBoxRenderer] = None
        
        # Window tags and layout  
        self.window_tag = "main_window"
        self.central_panel_tag = "central_panel"
        self.right_panel_tag = "right_panel"
        self.image_plot_tag = "image_plot"
        self.draw_list_tag = "box_selection_draw_list"
        
        # Unique axis tags to avoid conflicts with CropRotateUI
        self.x_axis_tag = "main_x_axis"
        self.y_axis_tag = "main_y_axis"
        
        # Interactive state
        self.box_selection_mode = False
        self.box_selection_active = False
        self.box_start = [0, 0]
        self.box_end = [0, 0]
        
        # Segmentation state
        self.segmentation_mode = False
        self.segmentation_texture = None
        self.pending_segmentation_box = None
        self.layer_masks = []
        self.mask_names = []
        
        # Display update state
        self._updating_display = False
    
    def setup(self) -> None:
        """Setup the main window and all components."""
        try:
            self._calculate_layout()
            self._create_main_window()
            self._setup_components()
            self._setup_handlers()
            
        except Exception as e:
            print(f"Error setting up main window: {e}")
            raise
    
    def _calculate_layout(self) -> None:
        """Calculate layout dimensions based on viewport."""
        # Use fixed dimensions during initial setup
        self.viewport_width = 1400
        self.viewport_height = 900
        self.menu_bar_height = 30
        
        self.available_height = self.viewport_height - self.menu_bar_height
        self.right_panel_width = int(self.viewport_width * 0.25)  # 25% for tools
        self.central_panel_width = self.viewport_width - self.right_panel_width
    
    def _create_main_window(self) -> None:
        """Create the main window structure."""
        with dpg.window(
            label="Photo Editor - Production",
            tag=self.window_tag,
            width=self.viewport_width,
            height=self.viewport_height,
            no_scrollbar=True,
            no_move=True,
            no_resize=True,
            no_collapse=True
        ):
            self._create_menu_bar()
            
            with dpg.group(horizontal=True):
                self._create_central_panel()
                self._create_right_panel()
            
            # Add status bar at the bottom
            with dpg.group(horizontal=True, width=-1):
                dpg.add_text("Ready", tag="status_text")
    
    def _create_menu_bar(self) -> None:
        """Create the menu bar."""
        with dpg.menu_bar():
            with dpg.menu(label="File"):
                dpg.add_menu_item(label="Open Image", callback=self._open_image)
                dpg.add_menu_item(label="Export Image", callback=self._export_image)
                dpg.add_separator()
                dpg.add_menu_item(label="Exit", callback=lambda: dpg.stop_dearpygui())
            
            with dpg.menu(label="Edit"):
                dpg.add_menu_item(label="Reset All", callback=self._reset_all_processing)
                dpg.add_menu_item(label="Clear Masks", callback=self._clear_all_masks)
            
            with dpg.menu(label="Segmentation"):
                dpg.add_menu_item(label="Auto Segment", callback=self._auto_segment)
                dpg.add_menu_item(label="Toggle Box Selection", callback=self._toggle_segmentation_mode)
    
    def _create_central_panel(self) -> None:
        """Create the central image display panel."""
        with dpg.child_window(
            tag=self.central_panel_tag,
            width=self.central_panel_width,
            height=self.available_height,
            border=False
        ):
            self._create_image_plot()
    
    def _create_image_plot(self) -> None:
        """Create the main image display plot."""
        # Calculate plot dimensions
        margin = 20
        plot_width = self.central_panel_width - (2 * margin)
        plot_height = self.available_height - (2 * margin)
        
        with dpg.child_window(
            parent=self.central_panel_tag,
            width=-1,
            height=-1,
            border=False,
            no_scrollbar=True
        ):
            with dpg.plot(
                label="Image Display",
                tag=self.image_plot_tag,
                height=plot_height,
                width=plot_width,
                anti_aliased=True,
                pan_button=dpg.mvMouseButton_Left,
                fit_button=dpg.mvMouseButton_Middle,
                no_mouse_pos=False,
                track_offset=True
            ):
                dpg.add_plot_axis(dpg.mvXAxis, label="X", no_gridlines=True, tag=self.x_axis_tag)
                y_axis_tag = self.y_axis_tag
                dpg.add_plot_axis(dpg.mvYAxis, label="Y", no_gridlines=True, tag=y_axis_tag)
                
                # Add placeholder for image series
                # Will be populated when image is loaded
    
    def _create_right_panel(self) -> None:
        """Create the right tool panel."""
        with dpg.child_window(
            tag=self.right_panel_tag,
            width=self.right_panel_width,
            height=self.available_height,
            border=True
        ):
            print("🛠️  Creating tool panel...")
            
            # Create modular tool panel using service layer
            self.tool_panel = ModularToolPanel(
                update_callback=self._on_parameter_change,
                app_service=self.app_service,
                main_window=self
            )
            
            self.tool_panel.setup()
            
            # Draw the tool panel UI
            if hasattr(self.tool_panel, 'draw'):
                self.tool_panel.draw()
            else:
                # Fallback: create simple tool panel UI
                dpg.add_text("Tool Panel")
                dpg.add_separator()
                dpg.add_text("Image Editor Tools")
                dpg.add_button(label="Load Image", callback=self._open_image)
                dpg.add_button(label="Save Image", callback=self._save_image)
                
            print("✅ Tool panel created")
    
    def _setup_components(self) -> None:
        """Setup additional UI components."""
        # Initialize crop/rotate UI when image is loaded
        # Initialize segmentation components when needed
        # Setup bounding box renderer for segmentation
        pass
    
    def _setup_handlers(self) -> None:
        """Setup mouse and keyboard handlers."""
        if not dpg.does_item_exist("main_mouse_handlers"):
            with dpg.handler_registry(tag="main_mouse_handlers"):
                dpg.add_mouse_down_handler(callback=self._on_mouse_down)
                dpg.add_mouse_drag_handler(callback=self._on_mouse_drag)
                dpg.add_mouse_release_handler(callback=self._on_mouse_release)
                dpg.add_mouse_wheel_handler(callback=self._on_mouse_wheel)
    
    def _setup_crop_mouse_handlers(self):
        """Setup mouse handlers specifically for crop functionality."""
        # The main handlers will delegate to crop handlers when appropriate
        pass
    
    # Event Handlers
    def _on_mouse_down(self, sender, app_data):
        """Handle mouse down events."""
        # Check if crop mode is active and delegate to CropRotateUI
        if (safe_item_check("crop_mode") and dpg.get_value("crop_mode") and 
            self.crop_rotate_ui):
            if self.crop_rotate_ui.on_mouse_down(sender, app_data):
                return  # Event handled by crop UI
        
        if self.segmentation_mode:
            self._handle_segmentation_mouse_down(app_data)
        elif self.box_selection_mode:
            self._handle_box_selection_mouse_down(app_data)
    
    def _on_mouse_drag(self, sender, app_data):
        """Handle mouse drag events."""
        # Check if crop mode is active and delegate to CropRotateUI
        if (safe_item_check("crop_mode") and dpg.get_value("crop_mode") and 
            self.crop_rotate_ui):
            if self.crop_rotate_ui.on_mouse_drag(sender, app_data):
                return  # Event handled by crop UI
        
        # Check if segmentation mode is active
        if self.segmentation_mode and self.segmentation_bbox_renderer:
            if self.segmentation_bbox_renderer.on_mouse_drag(sender, app_data):
                return  # Event handled by segmentation bbox renderer
        
        if self.box_selection_active:
            self._handle_box_selection_drag(app_data)
    
    def _on_mouse_release(self, sender, app_data):
        """Handle mouse release events."""
        # Check if crop mode is active and delegate to CropRotateUI
        if (safe_item_check("crop_mode") and dpg.get_value("crop_mode") and 
            self.crop_rotate_ui):
            if self.crop_rotate_ui.on_mouse_release(sender, app_data):
                return  # Event handled by crop UI
        
        # Check if segmentation mode is active
        if self.segmentation_mode and self.segmentation_bbox_renderer:
            if self.segmentation_bbox_renderer.on_mouse_release(sender, app_data):
                return  # Event handled by segmentation bbox renderer
        
        if self.box_selection_active:
            self._handle_box_selection_release(app_data)
    
    def _on_mouse_wheel(self, sender, app_data):
        """Handle mouse wheel events for zooming."""
        # Only zoom when mouse is over the image plot and we have an image loaded
        if (dpg.does_item_exist(self.image_plot_tag) and 
            self.crop_rotate_ui and 
            dpg.is_item_hovered(self.image_plot_tag)):
            
            # Get current plot limits
            x_limits = dpg.get_axis_limits(self.x_axis_tag)
            y_limits = dpg.get_axis_limits(self.y_axis_tag)
            
            # Calculate zoom factor based on wheel direction
            # Positive app_data = scroll up = zoom in (smaller zoom factor)
            # Negative app_data = scroll down = zoom out (larger zoom factor)
            zoom_factor = 0.9 if app_data > 0 else 1.1
            
            # Calculate new limits centered around current view center
            x_range = x_limits[1] - x_limits[0]
            y_range = y_limits[1] - y_limits[0]
            x_center = (x_limits[0] + x_limits[1]) / 2
            y_center = (y_limits[0] + y_limits[1]) / 2
            
            new_x_range = x_range * zoom_factor
            new_y_range = y_range * zoom_factor
            
            # Set new limits
            dpg.set_axis_limits(self.x_axis_tag, x_center - new_x_range/2, x_center + new_x_range/2)
            dpg.set_axis_limits(self.y_axis_tag, y_center - new_y_range/2, y_center + new_y_range/2)
            
            return True  # Event consumed
        
        return False
    
    def _on_parameter_change(self, sender, app_data, user_data):
        """Handle parameter changes from tool panel."""
        if self._updating_display:
            return
            
        try:
            # Get current image
            current_image = self.app_service.image_service.get_current_image()
            if current_image is None:
                return
            
            # Get or create image processor
            if not self.app_service.image_service.image_processor:
                self.app_service.image_service.create_image_processor(current_image)
            
            processor = self.app_service.image_service.image_processor
            
            if processor:
                # Get parameter values from UI
                params = self._collect_current_parameters()
                
                # Check for automatic mask reset if we're in mask editing mode
                self._check_for_automatic_mask_reset(params)
                
                # Update processor parameters
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
                
                # Set curves data on processor for apply_all_edits to use
                curves_data = params.get('curves')
                if curves_data:
                    processor.curves_data = curves_data
                else:
                    processor.curves_data = None
                
                # Apply all edits and get processed image
                processed_image = processor.apply_all_edits()
                if processed_image is not None:
                    # Update CropRotateUI's original image if it exists
                    if self.crop_rotate_ui:
                        self.crop_rotate_ui.original_image = processed_image.copy()
                        # Let CropRotateUI handle the display update
                        self.crop_rotate_ui.update_image(None, None, None)
                    else:
                        # Fallback: use our own texture update
                        self._update_texture(processed_image)
                    
                    # Update histogram with the processed image
                    if self.tool_panel:
                        try:
                            self.tool_panel.update_histogram(processed_image)
                        except AttributeError:
                            pass  # Histogram update not available
            
        except Exception as e:
            print(f"Error handling parameter change: {e}")
            traceback.print_exc()
    
    def _check_for_automatic_mask_reset(self, current_params):
        """Check if current parameters are at defaults and automatically reset mask if so."""
        try:
            # Only check if we have a tool panel with masks
            if not (self.tool_panel and hasattr(self.tool_panel, 'panel_manager')):
                return
                
            masks_panel = self.tool_panel.panel_manager.get_panel("masks")
            if not masks_panel:
                return
            
            # Only do auto-reset if we're in mask editing mode and there are committed changes
            if not (masks_panel.mask_editing_enabled and 
                   masks_panel.current_mask_index >= 0 and
                   masks_panel.current_mask_index in masks_panel.mask_committed_params):
                return
            
            # Define default parameter values
            default_params = {
                'exposure': 0,
                'illumination': 0.0,
                'contrast': 1.0,
                'shadow': 0,
                'highlights': 0,
                'whites': 0,
                'blacks': 0,
                'saturation': 1.0,
                'texture': 0,
                'grain': 0,
                'temperature': 0
            }
            
            # Check if curves are at default (linear)
            curves_at_default = True
            if self.tool_panel.curves_panel:
                curves = self.tool_panel.curves_panel.get_curves()
                if curves and 'curves' in curves:
                    default_curve = [(0, 0), (128, 128), (255, 255)]
                    curve_data = curves['curves']
                    for channel in ['r', 'g', 'b']:
                        if channel in curve_data and curve_data[channel] != default_curve:
                            curves_at_default = False
                            break
            
            # Check if all parameters are at their default values
            all_at_defaults = True
            for param_name, default_value in default_params.items():
                current_value = current_params.get(param_name, default_value)
                # Handle floating point comparison with small tolerance
                if isinstance(default_value, float):
                    if abs(current_value - default_value) > 0.001:
                        all_at_defaults = False
                        break
                else:
                    if current_value != default_value:
                        all_at_defaults = False
                        break
            
            # If all parameters AND curves are at defaults, trigger automatic reset
            if all_at_defaults and curves_at_default:
                mask_index = masks_panel.current_mask_index
                print(f"🔄 All parameters at defaults - automatically resetting mask {mask_index} to original state")
                
                # Check if we have a base state to restore to
                if mask_index not in masks_panel.mask_base_image_states:
                    print(f"⚠️ No base image state saved for mask {mask_index}, cannot auto-reset")
                    return
                
                # Get the processor
                processor = self.app_service.image_service.image_processor
                if not processor:
                    return
                
                # Restore the base image to the saved state before this mask was edited
                saved_base_state = masks_panel.mask_base_image_states[mask_index]
                processor.base_image = saved_base_state.copy()
                print(f"🔄 Auto-restored base image to state before mask {mask_index} was first edited")
                
                # Clear committed parameters since we've reverted the base image
                if mask_index in masks_panel.mask_committed_params:
                    del masks_panel.mask_committed_params[mask_index]
                    print(f"🗑️ Cleared committed parameters for mask {mask_index} (auto-reset)")
                
                # Keep current UI parameters (which should be at defaults) 
                # This allows user to see they're at defaults and start fresh if they want
                current_params = masks_panel._get_current_parameters()
                curves_data = masks_panel._get_current_curves_data()
                masks_panel.mask_params[mask_index] = {
                    'parameters': current_params,
                    'curves': curves_data
                }
                
                print(f"✅ Automatically reset mask {mask_index} to original state")
                
        except Exception as e:
            print(f"Error in automatic mask reset: {e}")

    def _collect_current_parameters(self) -> dict:
        """Collect current parameter values from the tool panel."""
        params = {}
        try:
            # Collect exposure parameters
            if safe_item_check("exposure"):
                params['exposure'] = dpg.get_value("exposure")
            if safe_item_check("illumination"):
                params['illumination'] = dpg.get_value("illumination")
            if safe_item_check("contrast"):
                params['contrast'] = dpg.get_value("contrast")
            if safe_item_check("shadow"):
                params['shadow'] = dpg.get_value("shadow")
            if safe_item_check("highlights"):
                params['highlights'] = dpg.get_value("highlights")
            if safe_item_check("whites"):
                params['whites'] = dpg.get_value("whites")
            if safe_item_check("blacks"):
                params['blacks'] = dpg.get_value("blacks")
            
            # Collect color effect parameters
            if safe_item_check("saturation"):
                params['saturation'] = dpg.get_value("saturation")
            if safe_item_check("texture"):
                params['texture'] = dpg.get_value("texture")
            if safe_item_check("grain"):
                params['grain'] = dpg.get_value("grain")
            if safe_item_check("temperature"):
                params['temperature'] = dpg.get_value("temperature")
            
            # Additional parameters if they exist
            if safe_item_check("vibrance"):
                params['vibrance'] = dpg.get_value("vibrance")
            if safe_item_check("tint"):
                params['tint'] = dpg.get_value("tint")
            if safe_item_check("hue"):
                params['hue'] = dpg.get_value("hue")
            if safe_item_check("clarity"):
                params['clarity'] = dpg.get_value("clarity")
            
            # Collect curves data from tool panel
            if self.tool_panel and self.tool_panel.curves_panel:
                curves_data = self.tool_panel.curves_panel.get_curves()
                if curves_data:
                    params['curves'] = curves_data
            
        except Exception as e:
            print(f"Error collecting parameters: {e}")
        
        return params
    
    # Menu Callbacks
    def _open_image(self):
        """Open image file dialog."""
        if safe_item_check("file_open_dialog"):
            dpg.show_item("file_open_dialog")
    
    def _save_image(self):
        """Save current image using export dialog."""
        self._export_image()
    
    def _export_image(self):
        """Show the export image modal."""
        if safe_item_check("export_modal"):
            dpg.show_item("export_modal")
        else:
            self._create_export_modal()
            dpg.show_item("export_modal")
    
    def _export_processed(self):
        """Export processed image - legacy function, redirects to export."""
        self._export_image()
    
    def _reset_all_processing(self):
        """Reset all processing parameters."""
        if self.tool_panel:
            self.tool_panel.reset_all_parameters()
    
    def _clear_all_masks(self):
        """Clear all masks."""
        print("Clearing all masks from menu")
        self.clear_all_masks()
    
    def _auto_segment(self, sender=None, app_data=None, user_data=None):
        """Handle auto segmentation menu item."""
        print("Auto segment triggered from menu")
        self.segment_current_image()
    
    def _toggle_segmentation_mode(self, sender=None, app_data=None, user_data=None):
        """Handle box selection mode toggle for manual segmentation."""
        print("Box selection mode toggle triggered from menu")
        self.box_selection_mode = not self.box_selection_mode
        
        if self.box_selection_mode:
            print("Box selection mode enabled - drag to create selection box")
            # Disable segmentation mode if it's active
            if self.segmentation_mode:
                self.disable_segmentation_mode()
        else:
            print("Box selection mode disabled")
            # Hide the draw list if it exists
            if dpg.does_item_exist(self.draw_list_tag):
                dpg.configure_item(self.draw_list_tag, show=False)
    
    def _create_export_modal(self):
        """Create the comprehensive export modal dialog."""
        import os
        
        # Create directory selection dialog at top level (not inside modal)
        with dpg.file_dialog(
            directory_selector=True,
            show=False,
            callback=self._export_path_callback,
            cancel_callback=self._export_path_cancel_callback,
            tag="export_path_dialog",
            width=600,
            height=400,
            modal=True
        ):
            pass
        
        with dpg.window(
            label="Export Image",
            tag="export_modal",
            modal=True,
            show=False,
            width=500,
            height=400,
            no_resize=True
        ):
            dpg.add_text("Export Settings", color=[176, 204, 255])
            dpg.add_separator()
            dpg.add_spacer(height=10)
            
            # File name section
            with dpg.group(horizontal=True):
                dpg.add_text("File Name:")
                dpg.add_spacer(width=20)
                dpg.add_input_text(
                    tag="export_filename",
                    default_value="exported_image",
                    width=300,
                    hint="Enter filename without extension",
                    callback=self._update_export_preview
                )
            
            dpg.add_spacer(height=10)
            
            # File type section
            with dpg.group(horizontal=True):
                dpg.add_text("Format:")
                dpg.add_spacer(width=43)
                dpg.add_combo(
                    tag="export_format",
                    items=["PNG", "JPEG", "BMP", "TIFF"],
                    default_value="PNG",
                    width=150,
                    callback=self._on_format_change
                )
            
            dpg.add_spacer(height=10)
            
            # Quality section (initially hidden for PNG)
            with dpg.group(tag="quality_group", show=False):
                with dpg.group(horizontal=True):
                    dpg.add_text("Quality:")
                    dpg.add_spacer(width=34)
                    dpg.add_slider_int(
                        tag="export_quality",
                        default_value=95,
                        min_value=1,
                        max_value=100,
                        width=200,
                        format="%d%%",
                        callback=self._update_export_preview
                    )
                dpg.add_spacer(height=10)
            
            # Path selection section
            with dpg.group(horizontal=True):
                dpg.add_text("Save to:")
                dpg.add_spacer(width=30)
                dpg.add_input_text(
                    tag="export_path",
                    default_value=os.path.expanduser("~"),
                    width=250,
                    readonly=True
                )
                dpg.add_button(
                    label="Browse...",
                    width=70,
                    callback=self._browse_export_path
                )
            
            dpg.add_spacer(height=20)
            dpg.add_separator()
            dpg.add_spacer(height=10)
            
            # Preview information
            dpg.add_text("Preview:", color=[200, 200, 200])
            dpg.add_text("", tag="export_preview", wrap=400)
            
            dpg.add_spacer(height=20)
            
            # Buttons
            with dpg.group(horizontal=True):
                dpg.add_spacer(width=250)
                dpg.add_button(
                    label="Cancel",
                    width=70,
                    callback=lambda: dpg.hide_item("export_modal")
                )
                dpg.add_spacer(width=10)
                dpg.add_button(
                    label="Export",
                    width=70,
                    callback=self._perform_export
                )
        
        # Update preview initially
        self._update_export_preview()
    
    def _on_format_change(self, sender, app_data, user_data):
        """Handle format change in export dialog."""
        format_type = dpg.get_value("export_format")
        
        # Show/hide quality slider based on format
        if format_type == "JPEG":
            dpg.configure_item("quality_group", show=True)
        else:
            dpg.configure_item("quality_group", show=False)
        
        self._update_export_preview()
    
    def _browse_export_path(self):
        """Open path browser for export location."""
        print("📂 Browse export path clicked")
        
        # Temporarily hide the export modal to avoid layering issues
        if dpg.does_item_exist("export_modal"):
            dpg.hide_item("export_modal")
            print("🔄 Hid export modal")
        
        # Show the path dialog
        if dpg.does_item_exist("export_path_dialog"):
            dpg.show_item("export_path_dialog")
            print("🔄 Showed path dialog")
        else:
            print("❌ Export path dialog does not exist")
    
    def _export_path_callback(self, sender, app_data, user_data):
        """Handle export path selection."""
        print(f"📁 Export path callback triggered")
        print(f"   Sender: {sender}")
        print(f"   App data: {app_data}")
        print(f"   User data: {user_data}")
        
        # Hide the path dialog first
        if dpg.does_item_exist("export_path_dialog"):
            dpg.hide_item("export_path_dialog")
        
        # Try to extract path from different possible data structures
        selected_path = ""
        if isinstance(app_data, dict):
            # Try different keys that DearPyGUI might use
            selected_path = app_data.get("file_path_name", "")
            if not selected_path:
                selected_path = app_data.get("current_path", "")
            if not selected_path:
                selected_path = app_data.get("file_name", "")
        elif isinstance(app_data, str):
            # Direct string path
            selected_path = app_data
        
        if selected_path:
            print(f"✅ Selected path: {selected_path}")
            if dpg.does_item_exist("export_path"):
                dpg.set_value("export_path", selected_path)
            self._update_export_preview()
        else:
            print("❌ No path selected or path extraction failed")
        
        # Use a small delay to ensure proper order of operations
        def show_modal_after_delay():
            if dpg.does_item_exist("export_modal"):
                dpg.show_item("export_modal")
                print("🔄 Showed export modal again (delayed)")
            else:
                print("❌ Export modal does not exist (delayed check)")
        
        # Schedule the modal to show after a small delay
        import threading
        timer = threading.Timer(0.1, show_modal_after_delay)
        timer.start()
    
    def _export_path_cancel_callback(self, sender, app_data, user_data):
        """Handle export path dialog cancellation."""
        print("🚫 Export path dialog cancelled")
        
        # Hide the path dialog
        if dpg.does_item_exist("export_path_dialog"):
            dpg.hide_item("export_path_dialog")
        
        # Use a small delay to ensure proper order of operations  
        def show_modal_after_delay():
            if dpg.does_item_exist("export_modal"):
                dpg.show_item("export_modal")
                print("🔄 Showed export modal again after cancel (delayed)")
            else:
                print("❌ Export modal does not exist (delayed check)")
        
        # Schedule the modal to show after a small delay
        import threading
        timer = threading.Timer(0.1, show_modal_after_delay)
        timer.start()
    
    def _update_export_preview(self):
        """Update the export preview text."""
        try:
            filename = dpg.get_value("export_filename")
            format_type = dpg.get_value("export_format")
            path = dpg.get_value("export_path")
            
            if not filename:
                filename = "exported_image"
            
            extension = {
                "PNG": ".png",
                "JPEG": ".jpg", 
                "BMP": ".bmp",
                "TIFF": ".tif"
            }.get(format_type, ".png")
            
            full_path = os.path.join(path, filename + extension)
            
            preview_text = f"File: {filename}{extension}\nLocation: {path}\nFull path: {full_path}"
            
            if format_type == "JPEG":
                quality = dpg.get_value("export_quality")
                preview_text += f"\nQuality: {quality}%"
            
            dpg.set_value("export_preview", preview_text)
            
        except Exception as e:
            dpg.set_value("export_preview", f"Preview error: {e}")
    
    def _perform_export(self):
        """Perform the actual image export."""
        try:
            # Get current processed image
            current_image = self.app_service.image_service.get_processed_image()
            if current_image is None:
                dpg.set_value("export_preview", "Error: No image to export")
                return
            
            # Get export settings
            filename = dpg.get_value("export_filename").strip()
            format_type = dpg.get_value("export_format")
            path = dpg.get_value("export_path")
            
            if not filename:
                dpg.set_value("export_preview", "Error: Please enter a filename")
                return
            
            # Create file extension mapping
            extension = {
                "PNG": ".png",
                "JPEG": ".jpg",
                "BMP": ".bmp", 
                "TIFF": ".tif"
            }.get(format_type, ".png")
            
            # Build full file path
            full_path = os.path.join(path, filename + extension)
            
            # Check if directory exists
            if not os.path.exists(path):
                dpg.set_value("export_preview", f"Error: Directory does not exist: {path}")
                return
            
            # Import required modules
            import cv2
            import numpy as np
            
            # Prepare image for saving
            save_image = current_image.copy()
            
            # Convert color format for OpenCV (RGB to BGR)
            if len(save_image.shape) == 3:
                if save_image.shape[2] == 3:
                    save_image = cv2.cvtColor(save_image, cv2.COLOR_RGB2BGR)
                elif save_image.shape[2] == 4:
                    save_image = cv2.cvtColor(save_image, cv2.COLOR_RGBA2BGRA)
            
            # Set up save parameters based on format
            save_params = []
            if format_type == "JPEG":
                quality = dpg.get_value("export_quality")
                save_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
            elif format_type == "PNG":
                save_params = [cv2.IMWRITE_PNG_COMPRESSION, 6]  # Good compression
            
            # Save the image
            success = cv2.imwrite(full_path, save_image, save_params)
            
            if success:
                print(f"✓ Image exported successfully to: {full_path}")
                dpg.set_value("export_preview", f"✓ Successfully exported to:\n{full_path}")
                
                # Close modal after short delay
                def close_modal():
                    dpg.hide_item("export_modal")
                
                # Close immediately for now - could add timer if wanted
                close_modal()
                
            else:
                dpg.set_value("export_preview", f"✗ Failed to export image to:\n{full_path}")
                print(f"✗ Failed to export image to: {full_path}")
                
        except Exception as e:
            error_msg = f"Export error: {str(e)}"
            dpg.set_value("export_preview", error_msg)
            print(f"Export error: {e}")
            import traceback
            traceback.print_exc()
    
    # Image Display Management
    def _update_image_display(self):
        """Update the image display with current processed image."""
        if self._updating_display:
            return
        
        self._updating_display = True
        try:
            processed_image = self.app_service.image_service.get_processed_image()
            if processed_image is None:
                return
            
            # Update texture
            self._update_texture(processed_image)
            
        except Exception as e:
            print(f"Error updating image display: {e}")
            traceback.print_exc()
        finally:
            self._updating_display = False
    
    def _update_texture(self, image: np.ndarray):
        """Update image texture using the original working pattern from CropRotateUI."""
        if self._updating_display:
            return
        
        # Check if crop mode is active - if so, let CropRotateUI handle texture updates
        crop_mode_active = safe_item_check("crop_mode") and dpg.get_value("crop_mode")
        if crop_mode_active and self.crop_rotate_ui:
            # Update the CropRotateUI's original image first
            self.crop_rotate_ui.original_image = image.copy()
            # Let CropRotateUI handle the texture update
            self.crop_rotate_ui.update_image(None, None, None)
            return
            
        try:
            
            # Prepare image data following original CropRotateUI pattern
            height, width = image.shape[:2]
            
            # Store original image dimensions for axis calculation
            self.orig_w = width
            self.orig_h = height
            
            # Use dynamic texture dimensions based on image size (like the original CropRotateUI)
            # Scale down large images to reasonable texture size while maintaining aspect ratio
            max_texture_dimension = 1024
            scale_factor = min(max_texture_dimension / width, max_texture_dimension / height, 1.0)
            
            texture_width = max(int(width * scale_factor), 400)
            texture_height = max(int(height * scale_factor), 300)
            
            # Ensure texture can contain the scaled image
            if scale_factor < 1.0:
                # Image was scaled down, use scaled dimensions
                scaled_width = int(width * scale_factor)
                scaled_height = int(height * scale_factor)
                image = cv2.resize(image, (scaled_width, scaled_height))
                width, height = scaled_width, scaled_height
            
            # Create gray background like original implementation
            gray_background = np.full((texture_height, texture_width, 4), 
                                     [100, 100, 100, 0], dtype=np.uint8)
            
            # Calculate centering offset (like original)
            offset_x = (texture_width - width) // 2
            offset_y = (texture_height - height) // 2
            
            # Store offsets for axis calculation
            self.image_offset_x = offset_x
            self.image_offset_y = offset_y
            
            # Convert to RGBA if needed
            if len(image.shape) == 2:
                # Grayscale to RGB then RGBA
                image_rgba = cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_GRAY2RGB), cv2.COLOR_RGB2RGBA)
            elif image.shape[2] == 3:
                # RGB to RGBA
                image_rgba = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)
            else:
                # Already RGBA
                image_rgba = image
            
            # Place image on background (centered)
            end_y = min(offset_y + image_rgba.shape[0], texture_height)
            end_x = min(offset_x + image_rgba.shape[1], texture_width)
            img_end_y = end_y - offset_y
            img_end_x = end_x - offset_x
            
            if offset_y >= 0 and offset_x >= 0:
                gray_background[offset_y:end_y, offset_x:end_x] = image_rgba[:img_end_y, :img_end_x]
            
            # Convert to float32 and normalize like original
            texture_data = gray_background.flatten().astype(np.float32) / 255.0;
            
            # Use original pattern: update existing texture or create new one
            texture_tag = "main_display_texture"
            if dpg.does_item_exist(texture_tag):
                # Update existing texture data (like original implementation)
                dpg.set_value(texture_tag, texture_data)
            else:
                # Create new raw texture (like original implementation)
                with dpg.texture_registry():
                    dpg.add_raw_texture(
                        texture_width, texture_height,
                        texture_data,
                        tag=texture_tag,
                        format=dpg.mvFormat_Float_rgba
                    )
                
                # Create image series
                if dpg.does_item_exist(self.y_axis_tag):
                    dpg.add_image_series(
                        texture_tag,
                        bounds_min=[0, 0],
                        bounds_max=[texture_width, texture_height],
                        parent=self.y_axis_tag,
                        tag="main_image_series"
                    )
            
            # Store current texture info
            self.current_texture_tag = texture_tag
            self.current_texture_width = texture_width
            self.current_texture_height = texture_height
            
            # Update axis limits to center on actual image
            self._update_axis_limits_to_image()
            
        except Exception as e:
            print(f"Error updating texture: {e}")
            traceback.print_exc()
    
    def _update_axis_limits_to_image(self):
        """Update axis limits to center on the actual image like the original implementation."""
        try:
            if not hasattr(self, 'orig_w') or not hasattr(self, 'orig_h'):
                # Fallback to simple axis limits if we don't have image dimensions
                if safe_item_check(self.x_axis_tag):
                    dpg.set_axis_limits(self.x_axis_tag, 0, self.current_texture_width)
                if safe_item_check(self.y_axis_tag):
                    dpg.set_axis_limits(self.y_axis_tag, 0, self.current_texture_height)
                return
            
            # Get image and texture dimensions
            orig_w = self.orig_w
            orig_h = self.orig_h
            texture_w = self.current_texture_width
            texture_h = self.current_texture_height
            image_offset_x = getattr(self, 'image_offset_x', 0)
            image_offset_y = getattr(self, 'image_offset_y', 0)
            
            # Get current plot dimensions
            plot_width = dpg.get_item_width(self.image_plot_tag) if dpg.does_item_exist(self.image_plot_tag) else 800
            plot_height = dpg.get_item_height(self.image_plot_tag) if dpg.does_item_exist(self.image_plot_tag) else 600
            
            if plot_width <= 0 or plot_height <= 0:
                plot_width, plot_height = 800, 600  # Fallback dimensions
            
            # Calculate aspect ratios
            plot_aspect = plot_width / plot_height
            image_aspect = orig_w / orig_h
            
            # Calculate how to fit the actual image (not texture) within the plot while maintaining aspect ratio
            # We want to show the image area with some padding to see the whole image properly
            padding_factor = 1.05  # 5% padding around the image
            
            if image_aspect > plot_aspect:
                # Image is wider relative to plot - fit to plot width
                # Show the full image width with padding
                display_width = orig_w * padding_factor
                display_height = display_width / plot_aspect
                
                # Center the view on the actual image
                x_center = image_offset_x + orig_w / 2
                y_center = image_offset_y + orig_h / 2
                x_min = x_center - display_width / 2
                x_max = x_center + display_width / 2
                y_min = y_center - display_height / 2
                y_max = y_center + display_height / 2
            else:
                # Image is taller relative to plot - fit to plot height
                # Show the full image height with padding
                display_height = orig_h * padding_factor
                display_width = display_height * plot_aspect
                
                # Center the view on the actual image
                x_center = image_offset_x + orig_w / 2
                y_center = image_offset_y + orig_h / 2
                x_min = x_center - display_width / 2
                x_max = x_center + display_width / 2
                y_min = y_center - display_height / 2
                y_max = y_center + display_height / 2
            
            # Set the axis limits to center on the actual image
            if safe_item_check(self.x_axis_tag):
                dpg.set_axis_limits(self.x_axis_tag, x_min, x_max)
            
            if safe_item_check(self.y_axis_tag):
                dpg.set_axis_limits(self.y_axis_tag, y_min, y_max)
                
        except Exception as e:
            print(f"Error updating axis limits: {e}")
            traceback.print_exc()

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
    
    def _handle_box_selection_mouse_down(self, app_data):
        """Handle mouse down events for box selection mode."""
        if dpg.is_item_hovered(self.image_plot_tag):
            self.box_selection_active = True
            plot_pos = dpg.get_plot_mouse_pos()
            self.box_start = list(plot_pos)
            self.box_end = list(plot_pos)
            
            # Create/setup the drawing layer if needed (add to y_axis like original)
            if not dpg.does_item_exist(self.draw_list_tag) and dpg.does_item_exist(self.image_plot_tag):
                # Get the y_axis from the plot
                plot_children = dpg.get_item_children(self.image_plot_tag, slot=1)
                if plot_children and len(plot_children) >= 2:
                    y_axis = plot_children[1]  # Second child is the y_axis
                    # Create a drawlist with the same size as the plot
                    plot_width = dpg.get_item_width(self.image_plot_tag)
                    plot_height = dpg.get_item_height(self.image_plot_tag)
                    dpg.add_drawlist(width=plot_width, height=plot_height, tag=self.draw_list_tag, parent=y_axis, show=False)
                    dpg.draw_rectangle([0, 0], [0, 0], color=[255, 255, 0], thickness=2, 
                                       tag="box_selection_rect", parent=self.draw_list_tag)
            
            if dpg.does_item_exist(self.draw_list_tag):
                dpg.configure_item(self.draw_list_tag, show=True)
                # Update initial rectangle
                self.update_box_rectangle()
            return True
        return False
    
    def _handle_box_selection_drag(self, app_data):
        """Handle mouse drag events for box selection mode."""
        if dpg.is_item_hovered(self.image_plot_tag) or self.box_selection_active:
            plot_pos = dpg.get_plot_mouse_pos()
            self.box_end = list(plot_pos)
            
            # Update rectangle
            if dpg.does_item_exist("box_selection_rect"):
                self.update_box_rectangle()  # Update rectangle with new coordinates
            return True
        return False
    
    def _handle_box_selection_release(self, app_data):
        """Handle mouse release events for box selection mode."""
        if self.box_selection_active:
            self.box_selection_active = False
            
            # Ensure box dimensions are valid
            if abs(self.box_end[0] - self.box_start[0]) > 10 and abs(self.box_end[1] - self.box_start[1]) > 10:
                # Format the box in [x1, y1, x2, y2] format
                box = [
                    min(self.box_start[0], self.box_end[0]),
                    min(self.box_start[1], self.box_end[1]),
                    max(self.box_start[0], self.box_end[0]),
                    max(self.box_start[1], self.box_end[1])
                ]
                
                # Make sure all values are finite numbers
                if all(isinstance(coord, (int, float)) and np.isfinite(coord) for coord in box):
                    print(f"Valid box selection: {box}")
                    # Perform segmentation with the box
                    self.segment_with_box(box)
                else:
                    print(f"Invalid box coordinates: {box}")
                    # Show error message to user if possible
                    dpg.configure_item("status_text", default_value="Error: Invalid selection area")
            else:
                print("Box selection too small (must be > 10x10 pixels)")
                # Show error message to user if possible
                dpg.configure_item("status_text", default_value="Selection area too small")
            
            # Hide the drawing layer containing the rectangle
            if dpg.does_item_exist(self.draw_list_tag):
                dpg.configure_item(self.draw_list_tag, show=False)
            return True
        return False
    
    def update_box_rectangle(self):
        """Update the box selection rectangle with current coordinates (following original pattern)."""
        if dpg.does_item_exist("box_selection_rect"):
            # Delete the old rectangle and create a new one (original pattern)
            dpg.delete_item("box_selection_rect")
        
        if dpg.does_item_exist(self.draw_list_tag):
            # Create new rectangle with updated coordinates (original pattern)
            dpg.draw_rectangle(self.box_start, self.box_end, color=[255, 255, 0], thickness=2,
                               tag="box_selection_rect", parent=self.draw_list_tag)
    
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
        
        print("CropRotateUI set and segmentation components initialized")
    
    def get_segmenter(self):
        """Get or create the segmenter instance. Created once to avoid GPU memory issues."""
        if self.segmenter is None:
            try:
                print("Creating ImageSegmenter instance...")
                # Use auto device selection for better memory management
                self.segmenter = ImageSegmenter(device="auto")
                print("ImageSegmenter instance created successfully")
            except Exception as e:
                print(f"Error creating ImageSegmenter: {e}")
                # Try fallback to CPU if GPU fails
                try:
                    print("Attempting to create ImageSegmenter with CPU...")
                    self.segmenter = ImageSegmenter(device="cpu")
                    print("ImageSegmenter instance created successfully on CPU")
                except Exception as cpu_e:
                    print(f"Error creating ImageSegmenter on CPU: {cpu_e}")
                    return None
        return self.segmenter
    
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
                if dpg.does_item_exist("status_text"):
                    dpg.configure_item("status_text", default_value=f"Selection: {int(bbox.width)}x{int(bbox.height)}")
            
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
                if dpg.does_item_exist("status_text"):
                    dpg.configure_item("status_text", default_value=f"Selection area: {int(bbox.width)}x{int(bbox.height)}. Press Confirm to segment.")
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
                if dpg.does_item_exist("status_text"):
                    dpg.configure_item("status_text", default_value="Selection too small (must be larger than 10x10)")
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
            if dpg.does_item_exist("status_text"):
                dpg.configure_item("status_text", default_value="Error: Segmentation not available")
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
        if dpg.does_item_exist("status_text"):
            dpg.configure_item("status_text", default_value="Segmentation mode: click and drag to select area")
            
        return True
    
    def disable_segmentation_mode(self):
        """Disable segmentation mode"""
        self.segmentation_mode = False
        self.segmentation_texture = None
        self.pending_segmentation_box = None
        
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
    
    def confirm_segmentation_selection(self):
        """Confirm the current segmentation selection and perform segmentation"""
        print(f"Confirm selection called - pending_box: {self.pending_segmentation_box}, segmentation_mode: {self.segmentation_mode}")
        
        # Debug current state
        if hasattr(self, 'segmentation_bbox_renderer') and self.segmentation_bbox_renderer:
            if hasattr(self.segmentation_bbox_renderer, 'bounding_box') and self.segmentation_bbox_renderer.bounding_box:
                current_box = self.segmentation_bbox_renderer.bounding_box
                print(f"Current bounding box in renderer: {current_box.x}, {current_box.y}, {current_box.width}, {current_box.height}")
                
                # If we have a bounding box but no pending box, use the current box
                if not self.pending_segmentation_box and current_box.width > 10 and current_box.height > 10:
                    print("Using current bounding box since no pending box is set")
                    self.pending_segmentation_box = current_box.to_dict()
            else:
                print("No bounding box in renderer")
                
                # Create a default bounding box as a fallback if user just clicks confirm without drawing
                if not self.pending_segmentation_box and self.crop_rotate_ui and self.crop_rotate_ui.original_image is not None:
                    h, w = self.crop_rotate_ui.original_image.shape[:2]
                    offset_x = (self.crop_rotate_ui.texture_w - w) // 2
                    offset_y = (self.crop_rotate_ui.texture_h - h) // 2
                    
                    # Create a box in the center that's 80% of the image size
                    box_w = int(w * 0.8)
                    box_h = int(h * 0.8)
                    box_x = offset_x + (w - box_w) // 2
                    box_y = offset_y + (h - box_h) // 2
                    
                    self.pending_segmentation_box = {
                        "x": box_x,
                        "y": box_y,
                        "w": box_w,
                        "h": box_h
                    }
                    
                    # Update the renderer with this box
                    if self.segmentation_bbox_renderer:
                        default_box = BoundingBox(box_x, box_y, box_w, box_h)
                        self.segmentation_bbox_renderer.set_bounding_box(default_box)
                        self._update_segmentation_overlay(force_update=True)
        else:
            print("No segmentation_bbox_renderer available")
        
        if not self.segmentation_mode:
            print("Segmentation mode is not active")
            if dpg.does_item_exist("status_text"):
                dpg.configure_item("status_text", default_value="Error: Segmentation mode not active")
            return
            
        if not self.pending_segmentation_box:
            print("No segmentation area selected")
            if dpg.does_item_exist("status_text"):
                dpg.configure_item("status_text", default_value="Error: No area selected. Draw a box first.")
            return
            
        # Convert bounding box to the format expected by segment_with_box
        bbox = self.pending_segmentation_box
        
        # Validate the box dimensions
        if bbox["w"] < 10 or bbox["h"] < 10:
            print(f"Selection area too small: {bbox['w']}x{bbox['h']} (minimum 10x10)")
            if dpg.does_item_exist("status_text"):
                dpg.configure_item("status_text", default_value="Error: Selection area too small")
            return
            
        # Box format: [x1, y1, x2, y2]
        box = [bbox["x"], bbox["y"], bbox["x"] + bbox["w"], bbox["y"] + bbox["h"]]
        
        # Disable segmentation mode
        self.disable_segmentation_mode()
        
        # Show loading indicator before performing segmentation
        self._show_segmentation_loading("Processing selection...")
        
        try:
            # Perform segmentation
            self.segment_with_box(box)
        finally:
            # Hide loading indicator (segment_with_box has its own loading management, 
            # but this ensures it's hidden if there are any early returns)
            self._hide_segmentation_loading()
        
        # Update tool panel to reflect the change
        if self.tool_panel:
            self.tool_panel.set_segmentation_mode(False)
    
    def cancel_segmentation_selection(self):
        """Cancel the current segmentation selection"""
        self.disable_segmentation_mode()
        
        # Update tool panel to reflect the change
        if self.tool_panel:
            self.tool_panel.set_segmentation_mode(False)
    
    def segment_current_image(self):
        """Segment the current image using SAM for automatic segmentation"""
        if self.crop_rotate_ui and hasattr(self.crop_rotate_ui, "original_image"):
            # Show loading indicator
            self._show_segmentation_loading("Segmenting image...")
            
            # Get the reusable segmenter instance
            segmenter = self.get_segmenter()
            if segmenter is None:
                print("Failed to create segmenter instance")
                self._hide_segmentation_loading()
                if dpg.does_item_exist("status_text"):
                    dpg.configure_item("status_text", default_value="Error: Failed to initialize segmenter")
                return
                
            image = self.crop_rotate_ui.original_image
            try:
                # Clean up memory before segmentation
                self.cleanup_segmenter_memory()
                
                masks = segmenter.segment(image)
                
                if not masks or len(masks) == 0:
                    print("No masks generated during automatic segmentation")
                    if dpg.does_item_exist("status_text"):
                        dpg.configure_item("status_text", default_value="No objects detected in the image")
                    return
                    
                # For automatic segmentation, replace all existing masks
                self.layer_masks = masks  # Store masks for further editing
                
                # Reset mask names for automatic segmentation
                self.mask_names = [f"Mask {idx + 1}" for idx in range(len(masks))]
                
                self.tool_panel.update_masks(masks, self.mask_names)
                self.update_mask_overlays(masks)
                
                # Update status text with success message
                if dpg.does_item_exist("status_text"):
                    dpg.configure_item("status_text", default_value=f"Created {len(masks)} masks")
                
                print(f"Automatic segmentation completed with {len(masks)} masks (replaced all previous masks)")
            except Exception as e:
                print(f"Error during automatic segmentation: {e}")
                # Update status text with error message
                if dpg.does_item_exist("status_text"):
                    dpg.configure_item("status_text", default_value=f"Error: {str(e)}")
                # On error, try to clean up memory and potentially reset segmenter
                try:
                    self.cleanup_segmenter_memory()
                    # If it's a memory error, suggest resetting the segmenter
                    if "memory" in str(e).lower() or "cuda" in str(e).lower():
                        print("Memory-related error detected. Consider resetting segmenter for next use.")
                        self.reset_segmenter()
                except Exception as cleanup_e:
                    print(f"Error during cleanup: {cleanup_e}")
            finally:
                # Always hide loading indicator
                self._hide_segmentation_loading()
        else:
            print("No image available for segmentation.")
            if dpg.does_item_exist("status_text"):
                dpg.configure_item("status_text", default_value="Error: No image available")
    
    def segment_with_box(self, box):
        """
        Segment the current image using a bounding box as input.
        
        Args:
            box: Bounding box coordinates [x1, y1, x2, y2] in texture coordinate system
                 (Y=0 at top, Y increases downward) from BoundingBoxRenderer
        """
        if not all(isinstance(coord, (int, float)) and np.isfinite(coord) for coord in box):
            print(f"Error: Invalid box coordinates: {box}")
            if dpg.does_item_exist("status_text"):
                dpg.configure_item("status_text", default_value="Error: Invalid selection area")
            return
            
        # Validate box dimensions
        if box[2] - box[0] <= 0 or box[3] - box[1] <= 0:
            print(f"Error: Box has zero or negative dimensions: {box}")
            if dpg.does_item_exist("status_text"):
                dpg.configure_item("status_text", default_value="Error: Invalid selection dimensions")
            return
            
        if self.crop_rotate_ui and hasattr(self.crop_rotate_ui, "original_image"):
            # Get the reusable segmenter instance
            segmenter = self.get_segmenter()
            if segmenter is None:
                print("Failed to create segmenter instance")
                if dpg.does_item_exist("status_text"):
                    dpg.configure_item("status_text", default_value="Error: Failed to initialize segmenter")
                return
                
            image = self.crop_rotate_ui.original_image
            
            # Get the original image dimensions
            h, w = image.shape[:2]
            
            # Calculate the texture offsets where the image is centered within the texture
            texture_w = self.crop_rotate_ui.texture_w
            texture_h = self.crop_rotate_ui.texture_h
            offset_x = (texture_w - w) // 2
            offset_y = (texture_h - h) // 2
            
            # The box coordinates are already in texture coordinate system (from BoundingBoxRenderer)
            # We just need to adjust for the texture offset to get image coordinates
            # Note: No Y-axis flipping needed since both are in texture/image coordinate system
            scaled_box = [
                max(0, min(w-1, int(box[0] - offset_x))),     # X1: Adjust X for texture offset
                max(0, min(h-1, int(box[1] - offset_y))),     # Y1: Adjust Y for texture offset
                max(0, min(w-1, int(box[2] - offset_x))),     # X2: Adjust X for texture offset
                max(0, min(h-1, int(box[3] - offset_y)))      # Y2: Adjust Y for texture offset
            ]
            
            # Check if the box is too small after scaling
            if scaled_box[2] - scaled_box[0] < 10 or scaled_box[3] - scaled_box[1] < 10:
                print(f"Error: Scaled box is too small: {scaled_box}")
                if dpg.does_item_exist("status_text"):
                    dpg.configure_item("status_text", default_value="Error: Selection too small relative to image")
                return
                
            print(f"Original box (texture coords): {box}")
            print(f"Texture dimensions: {texture_w}x{texture_h}")
            print(f"Image dimensions: {w}x{h}")
            print(f"Image offset in texture: ({offset_x}, {offset_y})")
            print(f"Segmenting with box (image coords): {scaled_box}")
            
            try:
                # Show loading indicator
                self._show_segmentation_loading("Segmenting selected area...")
                
                # Clean up memory before segmentation
                self.cleanup_segmenter_memory()
                
                new_masks = segmenter.segment_with_box(image, scaled_box)
                
                # Check if we got any masks
                if not new_masks or len(new_masks) == 0:
                    print("No masks generated from the selected area")
                    if dpg.does_item_exist("status_text"):
                        dpg.configure_item("status_text", default_value="No objects found in selection")
                    return
                
                # Initialize layer_masks if it doesn't exist
                if not hasattr(self, 'layer_masks') or self.layer_masks is None:
                    self.layer_masks = []
                
                # Initialize mask_names if it doesn't exist
                if not hasattr(self, 'mask_names') or self.mask_names is None:
                    self.mask_names = []
                
                # Always accumulate new masks with existing ones
                start_index = len(self.layer_masks)
                self.layer_masks.extend(new_masks)
                
                # Ensure we have names for all existing masks first
                while len(self.mask_names) < start_index:
                    self.mask_names.append(f"Mask {len(self.mask_names) + 1}")
                    
                # Add names for new masks
                for idx in range(len(new_masks)):
                    self.mask_names.append(f"Mask {start_index + idx + 1}")
                
                print(f"Generated {len(new_masks)} new masks, total masks: {len(self.layer_masks)}, total names: {len(self.mask_names)}")
                
                # Update UI with all masks
                if hasattr(self, 'tool_panel') and self.tool_panel:
                    try:
                        print(f"Attempting to update masks table with {len(self.layer_masks)} masks and {len(self.mask_names)} names")
                        # Use a copy of the data to prevent modification during iteration
                        masks_copy = self.layer_masks.copy()
                        names_copy = self.mask_names.copy()
                        self.tool_panel.update_masks(masks_copy, names_copy)
                        print("Masks table updated successfully")
                    except Exception as e:
                        print(f"Error updating masks table: {e}")
                        import traceback
                        traceback.print_exc()
                        
                self.update_mask_overlays(self.layer_masks)
                
                # Update status text
                if dpg.does_item_exist("status_text"):
                    dpg.configure_item("status_text", default_value=f"Created {len(new_masks)} new masks")
                
            except Exception as e:
                print(f"Error during box segmentation: {e}")
                # Update status text
                if dpg.does_item_exist("status_text"):
                    dpg.configure_item("status_text", default_value=f"Error: {str(e)}")
                # On error, try to clean up memory and potentially reset segmenter
                try:
                    self.cleanup_segmenter_memory()
                    # If it's a memory error, suggest resetting the segmenter
                    if "memory" in str(e).lower() or "cuda" in str(e).lower():
                        print("Memory-related error detected. Consider resetting segmenter for next use.")
                        self.reset_segmenter()
                except Exception as cleanup_e:
                    print(f"Error during cleanup: {cleanup_e}")
            finally:
                # Always hide loading indicator
                self._hide_segmentation_loading()
        else:
            print("No image available for segmentation.")
            if dpg.does_item_exist("status_text"):
                dpg.configure_item("status_text", default_value="Error: No image available")
    
    def _rotate_mask(self, mask, angle, orig_w, orig_h):
        """
        Rotate a binary mask by the given angle.
        
        Args:
            mask: Binary mask array (2D numpy array)
            angle: Rotation angle in degrees
            orig_w: Original image width
            orig_h: Original image height
            
        Returns:
            Rotated mask array
        """
        try:
            # Convert boolean mask to uint8 if needed
            if mask.dtype == bool:
                mask_uint8 = mask.astype(np.uint8) * 255
            else:
                mask_uint8 = mask.astype(np.uint8)
            
            # Get rotation matrix
            center = (orig_w / 2, orig_h / 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            
            # Calculate new dimensions after rotation
            cos = abs(M[0, 0])
            sin = abs(M[0, 1])
            new_w = int((orig_h * sin + orig_w * cos))
            new_h = int((orig_h * cos + orig_w * sin))
            
            # Adjust the rotation matrix for the new image center
            M[0, 2] += (new_w / 2) - center[0]
            M[1, 2] += (new_h / 2) - center[1]
            
            # Apply rotation to mask
            rotated_mask = cv2.warpAffine(mask_uint8, M, (new_w, new_h), 
                                        flags=cv2.INTER_NEAREST,
                                        borderMode=cv2.BORDER_CONSTANT, 
                                        borderValue=0)
            
            # Convert back to boolean
            return (rotated_mask > 127).astype(bool)
            
        except Exception as e:
            print(f"Error rotating mask: {e}")
            # Return original mask if rotation fails
            return mask

    def update_mask_overlays(self, masks):
        """Update the visual mask overlays on the image"""
        if not masks or not self.crop_rotate_ui:
            return
        
        # Get texture dimensions
        h = self.crop_rotate_ui.texture_h
        w = self.crop_rotate_ui.texture_w
        orig_h = self.crop_rotate_ui.orig_h
        orig_w = self.crop_rotate_ui.orig_w
        
        # Get current rotation angle if available
        current_angle = 0
        if dpg.does_item_exist("rotation_slider"):
            current_angle = dpg.get_value("rotation_slider")
        
        # Color palette for different masks (RGBA format)
        colors = [
            [255, 0, 0, 100],     # Red
            [0, 255, 0, 100],     # Green  
            [0, 0, 255, 100],     # Blue
            [255, 255, 0, 100],   # Yellow
            [255, 0, 255, 100],   # Magenta
            [0, 255, 255, 100],   # Cyan
            [255, 128, 0, 100],   # Orange
            [128, 0, 255, 100],   # Purple
            [255, 192, 203, 100], # Pink
            [0, 128, 128, 100],   # Teal
        ]
        
        max_masks = len(masks)  # Support unlimited masks
        
        # Calculate offset based on rotation
        if current_angle != 0 and hasattr(self.crop_rotate_ui, 'rotated_image') and self.crop_rotate_ui.rotated_image is not None:
            # Use rotated image dimensions and offset
            if hasattr(self.crop_rotate_ui, 'rot_h') and hasattr(self.crop_rotate_ui, 'rot_w'):
                rotated_h = self.crop_rotate_ui.rot_h
                rotated_w = self.crop_rotate_ui.rot_w
                offset_x = (w - rotated_w) // 2
                offset_y = (h - rotated_h) // 2
            else:
                # Fallback to original calculation
                offset_x = (w - orig_w) // 2
                offset_y = (h - orig_h) // 2
        else:
            # No rotation, use original calculation
            offset_x = (w - orig_w) // 2
            offset_y = (h - orig_h) // 2
        
        # Create a unique timestamp for this batch of masks to avoid tag conflicts
        import time
        timestamp = int(time.time() * 1000000)  # Microsecond timestamp for uniqueness
        
        # Remove existing series (but not textures) to avoid conflicts
        # Clean up old series - support up to 100 masks for large accumulations
        for idx in range(100):
            old_series_tag = f"mask_series_{idx}"
            if dpg.does_item_exist(old_series_tag):
                try:
                    dpg.delete_item(old_series_tag)
                except Exception as e:
                    print(f"Error deleting old series {old_series_tag}: {e}")
        
        # Create mask overlays with unique texture tags to avoid conflicts
        successful_masks = 0
        for idx in range(max_masks):
            try:
                mask = masks[idx]
                binary_mask = mask.get("segmentation")
                if binary_mask is None:
                    continue
                
                # Rotate the mask if there's a rotation angle
                if current_angle != 0:
                    rotated_mask = self._rotate_mask(binary_mask, current_angle, orig_w, orig_h)
                else:
                    rotated_mask = binary_mask
                
                # Create overlay texture
                overlay = np.zeros((h, w, 4), dtype=np.uint8)
                color = colors[idx % len(colors)]
                
                # Determine the correct dimensions to use for mask application
                if current_angle != 0 and hasattr(self.crop_rotate_ui, 'rot_h') and hasattr(self.crop_rotate_ui, 'rot_w'):
                    # Use rotated image dimensions
                    mask_h = min(rotated_mask.shape[0], self.crop_rotate_ui.rot_h)
                    mask_w = min(rotated_mask.shape[1], self.crop_rotate_ui.rot_w)
                    
                    # Apply rotated mask with color
                    for channel in range(4):
                        overlay[offset_y:offset_y + mask_h, offset_x:offset_x + mask_w, channel] = np.where(
                            rotated_mask[:mask_h, :mask_w] == 1, color[channel], 0)
                else:
                    # Apply original mask with color (no rotation)
                    for channel in range(4):
                        overlay[offset_y:offset_y + orig_h, offset_x:offset_x + orig_w, channel] = np.where(
                            rotated_mask == 1, color[channel], 0)
                
                # Convert to texture format
                texture_data = overlay.flatten().astype(np.float32) / 255.0
                
                # Use unique texture tag with timestamp to avoid conflicts
                texture_tag = f"mask_overlay_{idx}_{timestamp}"
                series_tag = f"mask_series_{idx}"  # Keep series tag simple for easier management
                
                # Create new texture with unique tag
                with dpg.texture_registry():
                    dpg.add_raw_texture(w, h, texture_data, tag=texture_tag, format=dpg.mvFormat_Float_rgba)
                
                # Create new image series
                dpg.add_image_series(texture_tag,
                                    bounds_min=[0, 0],
                                    bounds_max=[w, h],
                                    parent=self.y_axis_tag,
                                    tag=series_tag,
                                    show=False)  # Start hidden
                
                successful_masks += 1
                
            except Exception as e:
                print(f"Error creating mask overlay {idx}: {e}")
                traceback.print_exc()
                continue
        
        # Hide any unused existing overlays (beyond current mask count)
        # Check up to 100 potential existing overlays to handle large numbers of accumulated masks
        for idx in range(max_masks, 100):
            series_tag = f"mask_series_{idx}"
            if dpg.does_item_exist(series_tag):
                try:
                    dpg.configure_item(series_tag, show=False)
                except Exception as e:
                    print(f"Error hiding overlay {idx}: {e}")
        
        # Show first mask if any were created successfully and masks should be visible
        if successful_masks > 0:
            try:
                # Only show masks if:
                # 1. Masks are enabled
                # 2. Crop mode is not active
                # 3. Show mask overlay is enabled
                masks_enabled = dpg.does_item_exist("mask_section_toggle") and dpg.get_value("mask_section_toggle")
                crop_mode_active = dpg.does_item_exist("crop_mode") and dpg.get_value("crop_mode")
                show_overlay = dpg.does_item_exist("show_mask_overlay") and dpg.get_value("show_mask_overlay")
                
                if masks_enabled and not crop_mode_active and show_overlay:
                    self.show_selected_mask(0)
                    print(f"Successfully processed {successful_masks} mask overlays and showing first mask")
                else:
                    print(f"Successfully processed {successful_masks} mask overlays but not showing (masks_enabled={masks_enabled}, crop_mode_active={crop_mode_active}, show_overlay={show_overlay})")
            except Exception as e:
                print(f"Error showing selected mask: {e}")
    
    def show_selected_mask(self, selected_index):
        """Show only the selected mask and hide others"""
        if not hasattr(self, 'layer_masks') or not self.layer_masks:
            return
        
        for idx in range(len(self.layer_masks)):
            series_tag = f"mask_series_{idx}"
            if dpg.does_item_exist(series_tag):
                try:
                    # Show only the selected mask
                    dpg.configure_item(series_tag, show=(idx == selected_index))
                except Exception as e:
                    print(f"Error configuring mask {idx}: {e}")
        
        # Make sure the axis is properly fit
        if selected_index < len(self.layer_masks):
            try:
                dpg.fit_axis_data(self.x_axis_tag)
                dpg.fit_axis_data(self.y_axis_tag)
            except Exception as e:
                print(f"Error fitting axis data: {e}")
        else:
            print(f"Selected mask {selected_index} doesn't exist")
    
    def clear_all_masks(self):
        """Clear all accumulated masks and reset the segmentation state"""
        print("Clearing all accumulated masks")
        
        # Clear the stored masks
        self.layer_masks = []
        self.mask_names = []
        
        # Hide and clean up all mask overlays
        # Clean up all possible mask series - support up to 100 masks
        for idx in range(100):
            series_tag = f"mask_series_{idx}"
            if dpg.does_item_exist(series_tag):
                try:
                    dpg.configure_item(series_tag, show=False)
                    dpg.delete_item(series_tag)
                    print(f"Deleted mask overlay {idx}")
                except Exception as e:
                    print(f"Error deleting mask overlay {idx}: {e}")
        
        # Clear the mask list in the tool panel
        if self.tool_panel:
            self.tool_panel.update_masks([], [])
        
        print("All masks cleared successfully")
    
    def delete_mask(self, mask_index):
        """Delete a specific mask by index"""
        if not hasattr(self, 'layer_masks') or mask_index >= len(self.layer_masks):
            return
        
        # Remove mask from storage
        del self.layer_masks[mask_index]
        del self.mask_names[mask_index]
        
        # Clean up visual overlay
        self._cleanup_mask_overlay(mask_index)
        
        # Update UI
        self._update_ui_after_mask_change()
    
    def rename_mask(self, mask_index, new_name):
        """Rename a specific mask by index"""
        if not hasattr(self, 'mask_names') or mask_index >= len(self.mask_names):
            return
        
        self.mask_names[mask_index] = new_name
        
        # Update tool panel
        if self.tool_panel:
            self.tool_panel.update_masks(self.layer_masks, self.mask_names)
    
    def _cleanup_mask_overlay(self, mask_index):
        """Clean up the visual overlay for a specific mask"""
        series_tag = f"mask_series_{mask_index}"
        if dpg.does_item_exist(series_tag):
            try:
                dpg.configure_item(series_tag, show=False)
                dpg.delete_item(series_tag)
            except Exception as e:
                print(f"Error cleaning up mask overlay {mask_index}: {e}")
    
    def _update_ui_after_mask_change(self):
        """Update UI components after mask changes (delete/rename)"""
        # Update tool panel with current masks
        if self.tool_panel:
            self.tool_panel.update_masks(self.layer_masks, self.mask_names)
        
        # Re-create all overlays to fix indexing
        self.update_mask_overlays(self.layer_masks)
    
    def cleanup_segmenter_memory(self):
        """Clean up segmenter memory when needed"""
        if hasattr(self, 'segmenter') and self.segmenter:
            try:
                self.segmenter.cleanup_memory()
                print("Segmenter memory cleaned up")
            except Exception as e:
                print(f"Error cleaning up segmenter memory: {e}")
    
    def reset_segmenter(self):
        """Reset the segmenter instance to free memory"""
        if hasattr(self, 'segmenter') and self.segmenter:
            try:
                # Clean up memory first
                self.cleanup_segmenter_memory()
                # Reset the segmenter
                self.segmenter = None
                print("Segmenter instance reset")
            except Exception as e:
                print(f"Error resetting segmenter: {e}")
    
    def _show_segmentation_loading(self, message="Processing..."):
        """Show loading indicator for segmentation process"""
        if dpg.does_item_exist("segmentation_loading_group"):
            # Update the message and show the indicator
            if dpg.does_item_exist("segmentation_loading_text"):
                dpg.set_value("segmentation_loading_text", message)
            dpg.configure_item("segmentation_loading_group", show=True)
        
        # Update status text with the loading message
        if dpg.does_item_exist("status_text"):
            dpg.configure_item("status_text", default_value=f"{message}")
    
    def _hide_segmentation_loading(self):
        """Hide loading indicator for segmentation process"""
        if dpg.does_item_exist("segmentation_loading_group"):
            dpg.configure_item("segmentation_loading_group", show=False)
            
        # Reset status text when loading is complete
        if dpg.does_item_exist("status_text"):
            dpg.configure_item("status_text", default_value="Ready")
    
    def toggle_box_selection_mode(self, sender, app_data):
        """Toggle the box selection mode on/off"""
        # Get the current state - could be from either the tool panel or sender
        if sender and dpg.does_item_exist(sender):
            self.box_selection_mode = dpg.get_value(sender)
        elif dpg.does_item_exist("segmentation_mode"):
            # If called from tool panel, get value from tool panel checkbox
            self.box_selection_mode = dpg.get_value("segmentation_mode")
        else:
            # Fallback - toggle current state
            self.box_selection_mode = not self.box_selection_mode
        
        # Sync both checkboxes to the same state
        if dpg.does_item_exist("segmentation_mode"):
            dpg.set_value("segmentation_mode", self.box_selection_mode)
        
        print(f"Box selection mode: {self.box_selection_mode} (always accumulates masks)")
        
        # Hide the drawing layer when turning off the mode
        if dpg.does_item_exist(self.draw_list_tag) and not self.box_selection_mode:
            dpg.configure_item(self.draw_list_tag, show=False)
        
        # Reset box selection state
        self.box_selection_active = False
    
    def _create_image_series(self):
        """Create the image series to display the loaded image."""
        if not self.crop_rotate_ui or dpg.does_item_exist("central_image"):
            return
            
        try:
            # Get the y_axis from the plot to add the image series
            plot_children = dpg.get_item_children(self.image_plot_tag, slot=1)
            if not plot_children or len(plot_children) < 2:
                print("Error: Could not find y_axis in plot")
                return
                
            y_axis = plot_children[1]  # Second child is the y_axis
            
            # Create image series that shows the full texture
            dpg.add_image_series(
                self.crop_rotate_ui.texture_tag,
                bounds_min=[0, 0],
                bounds_max=[self.crop_rotate_ui.texture_w, self.crop_rotate_ui.texture_h],
                parent=y_axis,
                tag="central_image"
            )
            
            # Update axis limits to show the image properly
            self._update_axis_limits(initial=True)
            
        except Exception as e:
            print(f"Error creating image series: {e}")
            traceback.print_exc()

    def _update_axis_limits(self, initial=False):
        """Update axis limits to properly display the image."""
        if not self.crop_rotate_ui:
            return
            
        try:
            # Don't override user's pan/zoom unless this is initial setup
            if not initial and dpg.does_item_exist(self.image_plot_tag):
                return
                
            # Get image dimensions
            orig_w = self.crop_rotate_ui.orig_w
            orig_h = self.crop_rotate_ui.orig_h
            texture_w = self.crop_rotate_ui.texture_w
            texture_h = self.crop_rotate_ui.texture_h
            
            # Calculate where the image is positioned within the texture (centered)
            image_offset_x = (texture_w - orig_w) // 2
            image_offset_y = (texture_h - orig_h) // 2
            
            # Get plot dimensions for aspect ratio calculation
            plot_width = dpg.get_item_width(self.image_plot_tag) if dpg.does_item_exist(self.image_plot_tag) else 800
            plot_height = dpg.get_item_height(self.image_plot_tag) if dpg.does_item_exist(self.image_plot_tag) else 600
            
            if plot_width <= 0 or plot_height <= 0:
                plot_width, plot_height = 800, 600  # Fallback dimensions
            
            # Calculate aspect ratios
            plot_aspect = plot_width / plot_height
            image_aspect = orig_w / orig_h
            
            # Calculate display dimensions with padding
            padding_factor = 1.05  # 5% padding around the image
            
            if image_aspect > plot_aspect:
                # Image is wider - fit to plot width
                display_width = orig_w * padding_factor
                display_height = display_width / plot_aspect
            else:
                # Image is taller - fit to plot height
                display_height = orig_h * padding_factor
                display_width = display_height * plot_aspect
            
            # Center the view on the actual image
            x_center = image_offset_x + orig_w / 2
            y_center = image_offset_y + orig_h / 2
            x_min = x_center - display_width / 2
            x_max = x_center + display_width / 2
            y_min = y_center - display_height / 2
            y_max = y_center + display_height / 2
            
            # Set axis limits
            dpg.set_axis_limits(self.x_axis_tag, x_min, x_max)
            dpg.set_axis_limits(self.y_axis_tag, y_min, y_max)
            
        except Exception as e:
            print(f"Error updating axis limits: {e}")
    
    def handle_resize(self):
        """Handle window resize events."""
        try:
            # Get new viewport dimensions
            viewport_width = dpg.get_viewport_client_width()
            viewport_height = dpg.get_viewport_client_height()
            
            # Update layout dimensions
            self.viewport_width = viewport_width
            self.viewport_height = viewport_height
            self.available_height = viewport_height - self.menu_bar_height
            self.right_panel_width = int(viewport_width * 0.25)  # 25% for tools
            self.central_panel_width = viewport_width - self.right_panel_width
            
            # Update main window size
            if dpg.does_item_exist(self.window_tag):
                dpg.configure_item(self.window_tag, width=viewport_width, height=viewport_height)
            
            # Update panel sizes
            if dpg.does_item_exist(self.central_panel_tag):
                dpg.configure_item(self.central_panel_tag, 
                                 width=self.central_panel_width, 
                                 height=self.available_height)
            
            if dpg.does_item_exist(self.right_panel_tag):
                dpg.configure_item(self.right_panel_tag,
                                 width=self.right_panel_width,
                                 height=self.available_height)
            
            # Update image plot size
            if dpg.does_item_exist(self.image_plot_tag):
                margin = 20
                plot_width = max(self.central_panel_width - (2 * margin), 200)
                plot_height = max(self.available_height - (2 * margin), 200)
                
                dpg.configure_item(self.image_plot_tag, width=plot_width, height=plot_height)
                
                # Update drawlist size if it exists
                if dpg.does_item_exist(self.draw_list_tag):
                    dpg.configure_item(self.draw_list_tag, width=plot_width, height=plot_height)
            
            # Update tool panel curves if available
            if (self.tool_panel and 
                hasattr(self.tool_panel, 'curves_panel') and 
                self.tool_panel.curves_panel and
                hasattr(self.tool_panel.curves_panel, 'resize_plot')):
                self.tool_panel.curves_panel.resize_plot()
            
            # Update crop rotate UI if available
            if self.crop_rotate_ui:
                # Update the crop UI's panel reference and recalculate axis limits
                self.crop_rotate_ui.panel_id = self.central_panel_tag
                if hasattr(self.crop_rotate_ui, 'update_axis_limits'):
                    self.crop_rotate_ui.update_axis_limits()
                
                # Force update the image display
                if hasattr(self.crop_rotate_ui, 'update_image'):
                    self.crop_rotate_ui.update_image(None, None, None)
            
            # Update main window axis limits
            if hasattr(self, '_update_axis_limits'):
                self._update_axis_limits(initial=True)
            
        except Exception as e:
            print(f"Error handling resize: {e}")
            traceback.print_exc()
    
    def cleanup(self):
        """Cleanup application resources."""
        try:
            # Cleanup segmentation resources
            if self.segmenter:
                try:
                    # Clear GPU memory if available
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except:
                    pass
                self.segmenter = None
            
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
            
            # Memory cleanup
            if self.memory_manager:
                self.memory_manager.cleanup_gpu_memory()
                
            print("✓ Production main window cleanup completed")
            
        except Exception as e:
            print(f"⚠️  Error during production main window cleanup: {e}")
