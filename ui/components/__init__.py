"""
UI Components Package

This package contains all UI panel components for the image editor application.
Provides a centralized location for all panel-related functionality.
"""

# Import all panel components for convenient access
from .base_panel import PanelManager
from .exposure_panel import ExposurePanel
from .color_effects_panel import ColorEffectsPanel
from .crop_panel import CropPanel
from .masks_panel import MasksPanel
from .brush_panel import BrushPanel
from .curves_panel import CurvesPanel
from .histogram_panel import HistogramPanel
from .tool_panel_modular import ModularToolPanel, ToolPanel

# Export all components
__all__ = [
    'PanelManager',
    'ExposurePanel', 
    'ColorEffectsPanel',
    'CropPanel',
    'MasksPanel',
    'BrushPanel',
    'CurvesPanel',
    'HistogramPanel',
    'ModularToolPanel',
    'ToolPanel'
]

# Version info
__version__ = '1.0.0'
