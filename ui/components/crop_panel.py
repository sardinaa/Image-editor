"""
Crop and rotate control panel.
Handles crop mode toggle, rotation slider, and crop operations.
"""
import dearpygui.dearpygui as dpg
from typing import Dict, Any, Optional
from ui.components.base_panel import BasePanel
from utils.ui_helpers import UIStateManager


class CropPanel(BasePanel):
    """Panel for crop and rotate controls."""
    
    def __init__(self, callback=None, main_window=None, crop_and_rotate_ref=None):
        super().__init__(callback, main_window)
        self.panel_tag = "crop_panel_container"
        self.crop_and_rotate_ref = crop_and_rotate_ref
    
    def setup(self) -> None:
        """Setup the crop panel."""
        self.parameters = {
            'crop_mode': False,
            'rotate_angle': 0
        }
    
    def draw(self) -> None:
        """Draw the crop panel UI."""
        # Crop section
        self._create_checkbox(
            label="Crop & Rotate",
            tag="crop_mode",
            default=False
        )
        
        # Set callback after creating the checkbox
        if UIStateManager.safe_item_exists("crop_mode"):
            dpg.set_item_callback("crop_mode", self.toggle_crop_mode)
        
        # Crop controls panel (initially hidden)
        with dpg.child_window(
            tag="crop_panel",
            height=65,
            autosize_x=True,
            show=False,
            border=True
        ):
            self._create_slider_float(
                label="Rotation",
                tag="rotation_slider",
                default=0,
                min_val=0,
                max_val=360
            )
            
            # Set specific callback for rotation
            if UIStateManager.safe_item_exists("rotation_slider"):
                dpg.set_item_callback("rotation_slider", self._update_crop_rotate)
            
            self._create_button(
                label="Max Area",
                callback=self._set_max_rect,
                width=-1,
                height=18
            )
    
    def toggle_crop_mode(self, sender, app_data, user_data):
        """Toggle crop mode on/off."""
        current = UIStateManager.safe_get_value("crop_mode", False)
        
        # If crop mode is being enabled, disable masks
        if current:
            # Check if masks are currently enabled
            if (UIStateManager.safe_item_exists("mask_section_toggle") and 
                UIStateManager.safe_get_value("mask_section_toggle", False)):
                print("Disabling masks to enable crop & rotate")
                UIStateManager.safe_set_value("mask_section_toggle", False)
                # Trigger the mask section toggle to handle all the mask disabling logic
                if self.main_window and hasattr(self.main_window, 'tool_panel'):
                    tool_panel = self.main_window.tool_panel
                    if hasattr(tool_panel, 'toggle_mask_section'):
                        tool_panel.toggle_mask_section(None, None, None)
        
        # Show/hide crop panel
        UIStateManager.safe_configure_item("crop_panel", show=current)
        
        # Handle mask overlay visibility when crop mode changes
        if self.main_window and hasattr(self.main_window, 'layer_masks') and self.main_window.layer_masks:
            try:
                if current:
                    # Crop mode enabled - hide all mask overlays
                    from utils.ui_helpers import MaskOverlayManager
                    MaskOverlayManager.hide_all_overlays(len(self.main_window.layer_masks))
                    print("Hidden all mask overlays (crop mode enabled)")
                else:
                    # Crop mode disabled - update overlays to apply rotation
                    # The masks panel will control visibility based on mask state
                    self.main_window.update_mask_overlays(self.main_window.layer_masks)
                    print("Updated mask overlays (crop mode disabled, rotation applied)")
            except Exception as e:
                print(f"Error handling mask overlays during crop mode toggle: {e}")
        
        # Apply crop when checkbox is unchecked
        if not current:
            self._crop_image(sender, app_data, user_data)
        
        # Trigger main callback
        self._param_changed(sender, app_data, user_data)
    
    def _update_crop_rotate(self, sender, app_data, user_data):
        """Update crop/rotate visualization."""
        if self.crop_and_rotate_ref:
            crop_rotate_ui = self.crop_and_rotate_ref()
            if crop_rotate_ui:
                crop_rotate_ui.update_image(None, None, None)
        
        # Note: Mask overlays are NOT updated here during rotation to avoid inefficiency.
        # Masks are only visible when both mask mode is active AND crop mode is disabled.
        # Mask rotation is applied only once when crop mode is turned off in toggle_crop_mode.
        
        self._param_changed(sender, app_data, user_data)
    
    def _set_max_rect(self, sender, app_data, user_data):
        """Set crop rectangle to maximum area."""
        if self.crop_and_rotate_ref:
            crop_rotate_ui = self.crop_and_rotate_ref()
            if crop_rotate_ui:
                crop_rotate_ui.set_max_rect()
    
    def _crop_image(self, sender, app_data, user_data):
        """Apply crop to the image."""
        if self.crop_and_rotate_ref:
            crop_rotate_ui = self.crop_and_rotate_ref()
            if crop_rotate_ui:
                crop_rotate_ui.crop_image()
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get current crop parameters."""
        return {
            'crop_mode': UIStateManager.safe_get_value("crop_mode", False),
            'rotate_angle': UIStateManager.safe_get_value("rotation_slider", 0)
        }
