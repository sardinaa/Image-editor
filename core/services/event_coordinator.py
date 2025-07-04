"""
EventCoordinator - Parameter Management and Business Logic Coordination Service
Extracts parameter handling and business logic coordination from ProductionMainWindow.
This service focuses on parameter changes, image processing coordination, and UI synchronization.
Note: Direct input handling (mouse/keyboard) is handled by ui/event_handlers.py
"""

import dearpygui.dearpygui as dpg
import traceback
from typing import Dict, Any, Callable

from utils.ui_helpers import safe_item_check


class EventCoordinator:
    """
    Parameter management and business logic coordination service.
    
    Handles parameter changes, image processing coordination, and UI synchronization.
    This service extracts business logic from the UI layer, providing clean separation
    between UI events (handled by EventHandlers) and business logic coordination.
    
    Responsibilities:
    - Parameter change coordination between UI and services
    - Image processing pipeline management  
    - Automatic mask reset logic
    - UI component synchronization after data changes
    - Service layer delegation for business operations
    """
    
    def __init__(self, app_service, main_window):
        """Initialize the EventCoordinator with service dependencies."""
        self.app_service = app_service
        self.main_window = main_window
        
        # Event handling state
        self._updating_display = False
        self._resetting_parameter = False  # Flag to prevent interference during resets
        self._reset_parameter_override = {}  # Store override values during reset
        self._parameter_callbacks = {}
        
    def handle_parameter_change(self, sender, app_data, user_data):
        """
        Handle parameter changes from tool panel.
        
        This method centralizes all parameter change logic that was previously
        scattered throughout the main window class.
        """
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
                # Special handling during parameter reset
                if self._resetting_parameter:
                    # During reset, we trust the processor values that were set directly
                    # and skip the normal parameter collection to avoid interference
                    processed_image = processor.apply_all_edits()
                    if processed_image is not None:
                        self._update_image_display(processed_image)
                        self._update_histogram(processed_image)
                    return
                
                # Get parameter values from UI
                params = self._collect_current_parameters()
                
                # Check for automatic mask reset if we're in mask editing mode
                self._check_for_automatic_mask_reset(params)
                
                # Update processor parameters
                self._update_processor_parameters(processor, params)
                
                # Apply all edits and get processed image
                processed_image = processor.apply_all_edits()
                if processed_image is not None:
                    self._update_image_display(processed_image)
                    self._update_histogram(processed_image)
            
        except Exception as e:
            print(f"Error handling parameter change: {e}")
            traceback.print_exc()
    
    def _update_processor_parameters(self, processor, params: Dict[str, Any]):
        """Update image processor with current parameters."""
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
    
    def _update_image_display(self, processed_image):
        """Update the image display with processed image."""
        # Update CropRotateUI's original image if it exists
        if self.main_window.crop_rotate_ui:
            self.main_window.crop_rotate_ui.original_image = processed_image.copy()
            # Let CropRotateUI handle the display update
            self.main_window.crop_rotate_ui.update_image(None, None, None)
        else:
            # Fallback: use texture update
            self.main_window._update_texture(processed_image)
    
    def _update_histogram(self, processed_image):
        """Update histogram with the processed image."""
        if self.main_window.tool_panel:
            try:
                self.main_window.tool_panel.update_histogram(processed_image)
            except AttributeError:
                pass  # Histogram update not available
    
    def _collect_current_parameters(self) -> Dict[str, Any]:
        """Collect current parameter values from the tool panel."""
        params = {}
        try:
            # Collect exposure parameters
            if safe_item_check("exposure"):
                value = dpg.get_value("exposure")
                # Check for override value during reset
                override = self.get_parameter_override("exposure")
                params['exposure'] = override if override is not None else value
                
            if safe_item_check("illumination"):
                value = dpg.get_value("illumination")
                override = self.get_parameter_override("illumination")
                params['illumination'] = override if override is not None else value
                
            if safe_item_check("contrast"):
                value = dpg.get_value("contrast")
                override = self.get_parameter_override("contrast")
                params['contrast'] = override if override is not None else value
                
            if safe_item_check("shadow"):
                value = dpg.get_value("shadow")
                override = self.get_parameter_override("shadow")
                params['shadow'] = override if override is not None else value
                
            if safe_item_check("highlights"):
                value = dpg.get_value("highlights")
                override = self.get_parameter_override("highlights")
                params['highlights'] = override if override is not None else value
                
            if safe_item_check("whites"):
                value = dpg.get_value("whites")
                override = self.get_parameter_override("whites")
                params['whites'] = override if override is not None else value
                
            if safe_item_check("blacks"):
                value = dpg.get_value("blacks")
                override = self.get_parameter_override("blacks")
                params['blacks'] = override if override is not None else value
            
            # Collect color effect parameters
            if safe_item_check("saturation"):
                value = dpg.get_value("saturation")
                override = self.get_parameter_override("saturation")
                params['saturation'] = override if override is not None else value
                
            if safe_item_check("texture"):
                value = dpg.get_value("texture")
                override = self.get_parameter_override("texture")
                params['texture'] = override if override is not None else value
                
            if safe_item_check("grain"):
                value = dpg.get_value("grain")
                override = self.get_parameter_override("grain")
                params['grain'] = override if override is not None else value
                
            if safe_item_check("temperature"):
                value = dpg.get_value("temperature")
                override = self.get_parameter_override("temperature")
                params['temperature'] = override if override is not None else value
            
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
            if self.main_window.tool_panel and self.main_window.tool_panel.curves_panel:
                curves_data = self.main_window.tool_panel.curves_panel.get_curves()
                if curves_data:
                    params['curves'] = curves_data
            
        except Exception as e:
            print(f"Error collecting parameters: {e}")
        
        return params
    
    def _check_for_automatic_mask_reset(self, current_params):
        """Check if current parameters are at defaults and automatically reset mask if so."""
        try:
            # Only check if we have a tool panel with masks
            if not (self.main_window.tool_panel and hasattr(self.main_window.tool_panel, 'panel_manager')):
                return
                
            masks_panel = self.main_window.tool_panel.panel_manager.get_panel("masks")
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
            if self.main_window.tool_panel.curves_panel:
                curves = self.main_window.tool_panel.curves_panel.get_curves()
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
                self._perform_automatic_mask_reset(masks_panel, current_params)
                
        except Exception as e:
            print(f"Error in automatic mask reset: {e}")
    
    def _perform_automatic_mask_reset(self, masks_panel, current_params):
        """Perform the automatic mask reset operation."""
        mask_index = masks_panel.current_mask_index
        
        # Check if we have a mask being edited
        if mask_index not in masks_panel.mask_base_image_states:
            return
        
        # Get the processor
        processor = self.app_service.image_service.image_processor
        if not processor:
            return
        
        # Get the current mask ID for this mask index
        mask_id = f"mask_{mask_index}"
        
        # Clear committed edits for this mask from the processor
        if mask_id in processor.committed_mask_edits:
            del processor.committed_mask_edits[mask_id]
        
        # Clear committed parameters from the masks panel
        if mask_index in masks_panel.mask_committed_params:
            del masks_panel.mask_committed_params[mask_index]
        
        # Reset current parameters to defaults (they should already be at defaults)
        processor.reset_current_parameters()
        
        # Keep current UI parameters (which should be at defaults) 
        # This allows user to see they're at defaults and start fresh if they want
        current_params = masks_panel._get_current_parameters()
        curves_data = masks_panel._get_current_curves_data()
        masks_panel.mask_params[mask_index] = {
            'parameters': current_params,
            'curves': curves_data
        }
    
    def set_resetting_parameter(self, resetting: bool):
        """Set the parameter resetting flag to prevent interference during resets."""
        self._resetting_parameter = resetting
        
        if not resetting:
            # Clear override values when reset is complete
            if self._reset_parameter_override:
                self._reset_parameter_override.clear()
    
    def set_parameter_override(self, param_name: str, value):
        """Set a parameter override value during reset operations."""
        self._reset_parameter_override[param_name] = value
    
    def get_parameter_override(self, param_name: str):
        """Get a parameter override value if it exists."""
        return self._reset_parameter_override.get(param_name)
    
    def force_processor_parameter(self, param_name: str, value):
        """Force a parameter value directly on the processor, bypassing UI collection."""
        try:
            processor = self.app_service.image_service.image_processor
            if processor and hasattr(processor, param_name):
                setattr(processor, param_name, value)
                return True
        except Exception as e:
            pass
        return False
    
    def cleanup(self):
        """Cleanup EventCoordinator resources."""
        self._parameter_callbacks.clear()
        self.app_service = None
        self.main_window = None
