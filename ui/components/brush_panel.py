"""
Brush Tool Panel Component

Provides controls for manual mask drawing with adjustable brush size and eraser mode.
"""
import dearpygui.dearpygui as dpg
from typing import Dict, Any
from .base_panel import BasePanel
from utils.ui_helpers import UIStateManager


class BrushPanel(BasePanel):
    """Panel for brush tool controls."""
    
    def __init__(self, callback=None, main_window=None):
        super().__init__(callback, main_window)
        self.panel_tag = "brush_panel_container"
        
    def setup(self) -> None:
        """Setup the brush panel."""
        self.parameters = {
            'brush_mode': False,
            'brush_size': 20,
            'eraser_mode': False,
            'brush_opacity': 1.0,
            'brush_hardness': 0.8
        }
    
    def draw(self) -> None:
        """Draw the brush panel UI."""
        # Brush section
        self._create_checkbox(
            label="Brush Tool",
            tag="brush_mode",
            default=False
        )
        
        # Set callback after creating the checkbox
        if UIStateManager.safe_item_exists("brush_mode"):
            dpg.set_item_callback("brush_mode", self.toggle_brush_mode)
        
        # Brush controls panel (initially hidden)
        with dpg.child_window(
            tag="brush_panel",
            height=120,
            autosize_x=True,
            show=False,
            border=False
        ):
            # Brush size control
            self._create_slider_int(
                label="Brush Size",
                tag="brush_size",
                default=20,
                min_val=1,
                max_val=100
            )
            
            # Set callback after creating the slider
            if UIStateManager.safe_item_exists("brush_size"):
                dpg.set_item_callback("brush_size", self._param_changed)
            
            # Brush opacity control
            self._create_slider_float(
                label="Opacity",
                tag="brush_opacity",
                default=1.0,
                min_val=0.1,
                max_val=1.0
            )
            
            # Set callback after creating the slider
            if UIStateManager.safe_item_exists("brush_opacity"):
                dpg.set_item_callback("brush_opacity", self._param_changed)
            
            # Brush hardness control
            self._create_slider_float(
                label="Hardness",
                tag="brush_hardness",
                default=0.8,
                min_val=0.0,
                max_val=1.0
            )
            
            # Set callback after creating the slider
            if UIStateManager.safe_item_exists("brush_hardness"):
                dpg.set_item_callback("brush_hardness", self._param_changed)
            
            dpg.add_spacer(height=5)
            
            # Eraser mode toggle
            self._create_checkbox(
                label="Eraser Mode",
                tag="eraser_mode",
                default=False
            )
            
            # Set callback after creating the checkbox
            if UIStateManager.safe_item_exists("eraser_mode"):
                dpg.set_item_callback("eraser_mode", self._param_changed)
            
            dpg.add_spacer(height=5)
            
            # Control buttons
            with dpg.group(horizontal=True):
                self._create_button(
                    label="Clear Mask",
                    callback=self._clear_brush_mask,
                    width=82,
                    height=20
                )
                
                self._create_button(
                    label="Add to Masks",
                    callback=self._add_brush_mask_to_collection,
                    width=82,
                    height=20
                )
    
    def toggle_brush_mode(self, sender, app_data, user_data):
        """Toggle brush mode on/off."""
        current = UIStateManager.safe_get_value("brush_mode", False)
        
        # If brush mode is being enabled, disable conflicting modes
        if current:
            # Disable crop mode if active
            if (UIStateManager.safe_item_exists("crop_mode") and 
                UIStateManager.safe_get_value("crop_mode", False)):
                UIStateManager.safe_set_value("crop_mode", False)
                # Trigger the crop mode toggle to handle all the crop disabling logic
                if self.main_window and hasattr(self.main_window, 'tool_panel'):
                    tool_panel = self.main_window.tool_panel
                    crop_panel = tool_panel.panel_manager.get_panel("crop")
                    if crop_panel:
                        crop_panel.toggle_crop_mode(None, None, None)
            
            # Disable segmentation mode if active
            if (UIStateManager.safe_item_exists("segmentation_mode") and 
                UIStateManager.safe_get_value("segmentation_mode", False)):
                UIStateManager.safe_set_value("segmentation_mode", False)
                if self.main_window and hasattr(self.main_window, 'tool_panel'):
                    tool_panel = self.main_window.tool_panel
                    masks_panel = tool_panel.panel_manager.get_panel("masks")
                    if masks_panel:
                        masks_panel._toggle_segmentation_mode(None, None, None)
        
        # Show/hide brush panel
        UIStateManager.safe_configure_item("brush_panel", show=current)
        
        # Enable/disable brush mode in main window
        if self.main_window and hasattr(self.main_window, 'set_brush_mode'):
            self.main_window.set_brush_mode(current)
        
        # Trigger main callback
        self._param_changed(sender, app_data, user_data)
    
    def _param_changed(self, sender, app_data, user_data):
        """Handle parameter changes and update brush renderer."""
        # Update brush parameters in main window
        if self.main_window and hasattr(self.main_window, '_update_brush_parameters'):
            self.main_window._update_brush_parameters()
        
        # Call parent parameter changed
        super()._param_changed(sender, app_data, user_data)
    
    def _clear_brush_mask(self, sender, app_data, user_data):
        """Clear the current brush mask."""
        if self.main_window and hasattr(self.main_window, 'clear_brush_mask'):
            self.main_window.clear_brush_mask()
        self._param_changed(sender, app_data, user_data)
    
    def _add_brush_mask_to_collection(self, sender, app_data, user_data):
        """Add the current brush mask to the mask collection."""
        if self.main_window and hasattr(self.main_window, 'add_brush_mask_to_collection'):
            success = self.main_window.add_brush_mask_to_collection()
            if success:
                self.main_window._update_status("✓ Brush mask added to collection")
            else:
                self.main_window._update_status("✗ No brush mask to add")
        self._param_changed(sender, app_data, user_data)
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get current brush parameters."""
        return {
            'brush_mode': UIStateManager.safe_get_value("brush_mode", False),
            'brush_size': UIStateManager.safe_get_value("brush_size", 20),
            'eraser_mode': UIStateManager.safe_get_value("eraser_mode", False),
            'brush_opacity': UIStateManager.safe_get_value("brush_opacity", 1.0),
            'brush_hardness': UIStateManager.safe_get_value("brush_hardness", 0.8)
        }
    
    def set_parameters(self, params: Dict[str, Any]) -> None:
        """Set brush parameters."""
        for param_name, value in params.items():
            if UIStateManager.safe_item_exists(param_name):
                UIStateManager.safe_set_value(param_name, value)
