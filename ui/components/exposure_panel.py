"""
Exposure and lighting control panel.
Handles exposure, illumination, contrast, and tone adjustments.
"""
import dearpygui.dearpygui as dpg
from typing import Dict, Any
from ui.components.base_panel import BasePanel


class ExposurePanel(BasePanel):
    """Panel for exposure and lighting controls."""
    
    def __init__(self, callback=None, main_window=None):
        super().__init__(callback, main_window)
        self.panel_tag = "exposure_panel"
    
    def setup(self) -> None:
        """Setup the exposure panel."""
        self.parameters = {
            'exposure': 0,
            'illumination': 0.0,
            'contrast': 1.0,
            'shadow': 0,
            'highlights': 0,
            'whites': 0,
            'blacks': 0
        }
    
    def draw(self) -> None:
        """Draw the exposure panel UI."""
        # Exposure & Lighting section
        self._create_section_header("Exposure & Lighting", [200, 200, 200])
        
        self._create_slider_int(
            label="Exposure",
            tag="exposure",
            default=0,
            min_val=-100,
            max_val=100
        )
        
        self._create_slider_float(
            label="Illumination",
            tag="illumination",
            default=0.0,
            min_val=-100.0,
            max_val=100.0
        )
        
        self._create_slider_float(
            label="Contrast",
            tag="contrast",
            default=1.0,
            min_val=0.5,
            max_val=3.0
        )
        
        dpg.add_spacer(height=1)
        
        # Tone Adjustments section
        self._create_section_header("Tone Adjustments", [200, 200, 200])
        
        self._create_slider_int(
            label="Shadow",
            tag="shadow",
            default=0,
            min_val=-100,
            max_val=100
        )
        
        self._create_slider_int(
            label="Highlights",
            tag="highlights",
            default=0,
            min_val=-100,
            max_val=100
        )
        
        self._create_slider_int(
            label="Whites",
            tag="whites",
            default=0,
            min_val=-100,
            max_val=100
        )
        
        self._create_slider_int(
            label="Blacks",
            tag="blacks",
            default=0,
            min_val=-100,
            max_val=100
        )
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get current exposure parameters."""
        from utils.ui_helpers import UIStateManager
        
        return {
            'exposure': UIStateManager.safe_get_value("exposure", 0),
            'illumination': UIStateManager.safe_get_value("illumination", 0.0),
            'contrast': UIStateManager.safe_get_value("contrast", 1.0),
            'shadow': UIStateManager.safe_get_value("shadow", 0),
            'highlights': UIStateManager.safe_get_value("highlights", 0),
            'whites': UIStateManager.safe_get_value("whites", 0),
            'blacks': UIStateManager.safe_get_value("blacks", 0)
        }
