import dearpygui.dearpygui as dpg
from typing import Dict, Any
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
            'rotate_angle': 0,
            'flip_horizontal': False,
            'flip_vertical': False
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
            border=False
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
            
            # Flip buttons row
            with dpg.group(horizontal=True):
                self._create_button(
                    label="Flip H",
                    callback=self._flip_horizontal,
                    width=80,
                    height=18
                )
                
                self._create_button(
                    label="Flip V", 
                    callback=self._flip_vertical,
                    width=80,
                    height=18
                )
            
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
            if (UIStateManager.safe_item_exists("mask_section_toggle") and 
                UIStateManager.safe_get_value("mask_section_toggle", False)):
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
                    from utils.ui_helpers import MaskOverlayManager
                    MaskOverlayManager.hide_all_overlays(len(self.main_window.layer_masks))
                else:
                    self.main_window.update_mask_overlays(self.main_window.layer_masks)
            except Exception as e:
                print(f"Error handling mask overlays during crop mode toggle: {e}")
        
        # Apply crop when checkbox is unchecked
        if not current:
            self._crop_image(sender, app_data, user_data)
        
        # Trigger main callback
        self._param_changed(sender, app_data, user_data)
    
    def _update_crop_rotate(self, sender, app_data, user_data):
        """Update crop/rotate visualization."""
        crop_rotate_ui = None
        if self.crop_and_rotate_ref:
            crop_rotate_ui = self.crop_and_rotate_ref()
        elif self.main_window and hasattr(self.main_window, 'crop_rotate_ui'):
            crop_rotate_ui = self.main_window.crop_rotate_ui
            
        if crop_rotate_ui:
            crop_rotate_ui.update_image(None, None, None)
        
        # Update mask overlays if masks are enabled and visible, and crop mode is NOT active
        # This ensures masks rotate consistently with the image rotation
        if (self.main_window and 
            hasattr(self.main_window, 'mask_overlay_renderer') and 
            self.main_window.mask_overlay_renderer and
            self.main_window.app_service and 
            self.main_window.crop_rotate_ui):
            
            # Check if masks should be updated (not in crop mode, masks enabled)
            crop_mode_active = UIStateManager.safe_get_value("crop_mode", False)
            masks_enabled = UIStateManager.safe_get_value("mask_section_toggle", False)
            show_overlay = UIStateManager.safe_get_value("show_mask_overlay", True)
            
            if not crop_mode_active and masks_enabled and show_overlay:
                masks = self.main_window.app_service.get_mask_service().get_masks()
                if masks:
                    # Update mask overlays to apply current rotation
                    self.main_window.mask_overlay_renderer.update_mask_overlays(
                        masks, self.main_window.crop_rotate_ui
                    )
        
        self._param_changed(sender, app_data, user_data)
    
    def _set_max_rect(self, sender, app_data, user_data):
        """Set crop rectangle to maximum area."""
        crop_rotate_ui = None
        if self.crop_and_rotate_ref:
            crop_rotate_ui = self.crop_and_rotate_ref()
        elif self.main_window and hasattr(self.main_window, 'crop_rotate_ui'):
            crop_rotate_ui = self.main_window.crop_rotate_ui
            
        if crop_rotate_ui:
            crop_rotate_ui.set_to_max_rect(sender, app_data, user_data)
    
    def _crop_image(self, sender, app_data, user_data):
        """Apply crop to the image."""
        crop_rotate_ui = None
        if self.crop_and_rotate_ref:
            crop_rotate_ui = self.crop_and_rotate_ref()
        elif self.main_window and hasattr(self.main_window, 'crop_rotate_ui'):
            crop_rotate_ui = self.main_window.crop_rotate_ui
            
        if crop_rotate_ui:
            crop_rotate_ui.crop_image(sender, app_data, user_data)
    
    def _flip_horizontal(self, sender, app_data, user_data):
        """Toggle horizontal flip."""
        current = self.parameters.get('flip_horizontal', False)
        self.parameters['flip_horizontal'] = not current
        
        crop_rotate_ui = None
        if self.crop_and_rotate_ref:
            crop_rotate_ui = self.crop_and_rotate_ref()
        elif self.main_window and hasattr(self.main_window, 'crop_rotate_ui'):
            crop_rotate_ui = self.main_window.crop_rotate_ui
            
        if crop_rotate_ui:
            crop_rotate_ui.toggle_flip_horizontal()
        
        self._param_changed(sender, app_data, user_data)
    
    def _flip_vertical(self, sender, app_data, user_data):
        """Toggle vertical flip."""
        current = self.parameters.get('flip_vertical', False)
        self.parameters['flip_vertical'] = not current

        crop_rotate_ui = None
        if self.crop_and_rotate_ref:
            crop_rotate_ui = self.crop_and_rotate_ref()
        elif self.main_window and hasattr(self.main_window, 'crop_rotate_ui'):
            crop_rotate_ui = self.main_window.crop_rotate_ui
            
        if crop_rotate_ui:
            crop_rotate_ui.toggle_flip_vertical()
        
        self._param_changed(sender, app_data, user_data)
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get current crop parameters."""
        params = {
            'crop_mode': UIStateManager.safe_get_value("crop_mode", False),
            'rotate_angle': UIStateManager.safe_get_value("rotation_slider", 0)
        }

        crop_rotate_ui = None
        if self.crop_and_rotate_ref:
            crop_rotate_ui = self.crop_and_rotate_ref()
        elif self.main_window and hasattr(self.main_window, 'crop_rotate_ui'):
            crop_rotate_ui = self.main_window.crop_rotate_ui
            
        if crop_rotate_ui:
            flip_states = crop_rotate_ui.get_flip_states()
            params.update(flip_states)
        else:
            params.update({
                'flip_horizontal': self.parameters.get('flip_horizontal', False),
                'flip_vertical': self.parameters.get('flip_vertical', False)
            })
        
        return params
