"""
Color and effects control panel.
Handles saturation, texture, grain, and temperature adjustments.
"""
import dearpygui.dearpygui as dpg
from typing import Dict, Any
from ui.components.base_panel import BasePanel


class ColorEffectsPanel(BasePanel):
    """Panel for color and effects controls."""
    
    def __init__(self, callback=None, main_window=None):
        super().__init__(callback, main_window)
        self.panel_tag = "color_effects_panel"
    
    def setup(self) -> None:
        """Setup the color effects panel."""
        self.parameters = {
            'saturation': 1.0,
            'texture': 0,
            'grain': 0,
            'temperature': 0
        }
    
    def draw(self) -> None:
        """Draw the color effects panel UI."""
        # Color & Effects section
        self._create_section_header("Color & Effects", [200, 200, 200])
        
        self._create_slider_float(
            label="Saturation",
            tag="saturation",
            default=1.0,
            min_val=0.0,
            max_val=3.0
        )
        
        self._create_slider_int(
            label="Texture",
            tag="texture",
            default=0,
            min_val=-100,
            max_val=100
        )
        
        self._create_slider_int(
            label="Grain",
            tag="grain",
            default=0,
            min_val=0,
            max_val=100
        )
        
        self._create_slider_int(
            label="Temperature",
            tag="temperature",
            default=0,
            min_val=-100,
            max_val=100
        )
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get current color effects parameters."""
        from utils.ui_helpers import UIStateManager
        
        return {
            'saturation': UIStateManager.safe_get_value("saturation", 1.0),
            'texture': UIStateManager.safe_get_value("texture", 0),
            'grain': UIStateManager.safe_get_value("grain", 0),
            'temperature': UIStateManager.safe_get_value("temperature", 0)
        }
