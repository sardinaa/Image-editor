import dearpygui.dearpygui as dpg
from typing import Dict, Any, Optional, Callable
import traceback

# Import panel components
from .base_panel import PanelManager
from .exposure_panel import ExposurePanel
from .color_effects_panel import ColorEffectsPanel
from .crop_panel import CropPanel
from .masks_panel import MasksPanel

# Import existing specialized panels
from .curves_panel import CurvesPanel
from .histogram_panel import HistogramPanel

from utils.ui_helpers import UIStateManager


class ModularToolPanel:
    """Modular tool panel using component-based architecture."""
    
    def __init__(self, update_callback: Optional[Callable] = None, 
                 mask_service=None, 
                 app_service=None,
                 crop_and_rotate_ref: Optional[Callable] = None, 
                 main_window=None):
        self.callback = update_callback
        self.crop_and_rotate_ref = crop_and_rotate_ref
        self.main_window = main_window
        self.mask_service = mask_service
        self.app_service = app_service
        
        # Panel manager for coordinating components
        self.panel_manager = PanelManager()
        
        # Legacy curves support
        self.curves = {
            "r": [(0, 0), (128, 128), (255, 255)],
            "g": [(0, 0), (128, 128), (255, 255)],
            "b": [(0, 0), (128, 128), (255, 255)]
        }
        
        # Specialized panels
        self.curves_panel = None
        self.histogram_panel = None
        
        # Initialize all panel components
        self._initialize_panels()
    
    def _initialize_panels(self):
        """Initialize all panel components."""
        # Create and register panel components
        self.panel_manager.register_panel(
            "exposure", 
            ExposurePanel(self.callback, self.main_window)
        )
        
        self.panel_manager.register_panel(
            "color_effects", 
            ColorEffectsPanel(self.callback, self.main_window)
        )
        
        self.panel_manager.register_panel(
            "crop", 
            CropPanel(self.callback, self.main_window, self.crop_and_rotate_ref)
        )
        
        self.panel_manager.register_panel(
            "masks", 
            MasksPanel(self.callback, self.main_window)
        )
        
        # Setup all panels
        for panel in self.panel_manager.panels.values():
            panel.setup()
    
    def setup(self):
        """Setup the modular tool panel and all its components."""
        # Calculate available width for histogram
        available_width = self._get_available_width()
        
        # Initialize specialized panels
        self.curves_panel = CurvesPanel(self.callback)
        self.histogram_panel = HistogramPanel(width=available_width)
        
        # Setup specialized panels
        if hasattr(self.curves_panel, 'setup'):
            self.curves_panel.setup()
        if hasattr(self.histogram_panel, 'setup'):
            self.histogram_panel.setup()
        
        # Update mask service reference in masks panel
        if self.mask_service:
            masks_panel = self.panel_manager.get_panel("masks")
            if masks_panel and hasattr(masks_panel, 'set_mask_service'):
                masks_panel.set_mask_service(self.mask_service)
    
    def _get_available_width(self):
        """Calculate available width for the histogram panel."""
        try:
            # Get viewport width
            viewport_width = dpg.get_viewport_client_width()
            # Tool panel is 25% of viewport width
            tool_panel_width = int(viewport_width * 0.25)
            # Account for padding and borders - be more conservative
            available_width = tool_panel_width - 25  # Account for left padding + border
            # Ensure minimum width
            return max(available_width, 200)
        except (SystemError, RuntimeError):
            # Viewport not created yet, use default
            return 220
    
    def draw(self):
        """Draw the complete tool panel."""
        with dpg.group(horizontal=False):
            # Histogram panel at the top
            self._draw_histogram_panel()
            
            dpg.add_separator()
            dpg.add_spacer(height=1)  # Reduced from 2
            
            # Main tools header
            dpg.add_text("Basic Editing Tools", color=[176, 204, 255])
            dpg.add_separator()
            dpg.add_spacer(height=1)  # Reduced from 2
            
            # Crop panel
            crop_panel = self.panel_manager.get_panel("crop")
            if crop_panel:
                crop_panel.draw()
            
            dpg.add_separator()
            
            # Exposure panel
            exposure_panel = self.panel_manager.get_panel("exposure")
            if exposure_panel:
                exposure_panel.draw()
            
            dpg.add_spacer(height=1)
            
            # Color effects panel
            color_panel = self.panel_manager.get_panel("color_effects")
            if color_panel:
                color_panel.draw()
            
            # RGB Curves section
            self._draw_curves_panel()
            
            # Masks panel
            masks_panel = self.panel_manager.get_panel("masks")
            if masks_panel:
                masks_panel.draw()
        
        # Register deferred callbacks after all UI elements are created
        self._register_all_deferred_callbacks()
        
        # Setup global double-click handler for slider resets
        self._setup_global_double_click_handler()
    
    def _draw_histogram_panel(self):
        """Draw the histogram panel."""
        if not self.histogram_panel:
            self.histogram_panel = HistogramPanel()
        self.histogram_panel.show()
    
    def _draw_curves_panel(self):
        """Draw the curves panel."""
        dpg.add_separator()
        dpg.add_spacer(height=1)  # Reduced from 2
        dpg.add_text("RGB Curves", color=[176, 204, 255])
        
        # Embed the curves panel directly
        if not self.curves_panel:
            self.curves_panel = CurvesPanel(callback=self.callback)
            self.curves_panel.curves = self.curves
        
        self.curves_panel.show()
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get parameters from all panels."""
        # Get parameters from all modular panels
        all_params = self.panel_manager.get_all_parameters()
        
        # Add curves parameters
        if self.curves_panel:
            self.curves = self.curves_panel.get_curves()
            all_params['curves'] = self.curves
        
        # Add any additional legacy parameters
        additional_params = {
            'rotate_angle': UIStateManager.safe_get_value("rotation_slider", 0),
        }
        all_params.update(additional_params)
        
        return all_params
    
    def set_parameters(self, params: Dict[str, Any]):
        """Set parameters for all panels."""
        # Set parameters for all modular panels
        self.panel_manager.set_all_parameters(params)
        
        # Handle curves separately
        if 'curves' in params and self.curves_panel:
            self.curves_panel.set_curves(params['curves'])
            self.curves = params['curves']
    
    def reset_all_parameters(self):
        """Reset all processing parameters to their default values."""
        try:
            # Temporarily disable callbacks to prevent triggering while resetting
            self.disable_parameter_callbacks()
            
            # Get default parameter values from all panels using their own definitions
            default_params = self.panel_manager.get_all_default_parameters()
            
            # Add any additional default parameters not handled by panels
            additional_defaults = {
                'rotation_slider': 0
            }
            default_params.update(additional_defaults)
            
            # Reset UI controls to default values
            for param_name, default_value in default_params.items():
                if UIStateManager.safe_item_exists(param_name):
                    UIStateManager.safe_set_value(param_name, default_value)
            
            # Reset curves to default (linear)
            if self.curves_panel:
                self.curves_panel.curves = {
                    "r": [(0, 0), (128, 128), (255, 255)],
                    "g": [(0, 0), (128, 128), (255, 255)],
                    "b": [(0, 0), (128, 128), (255, 255)]
                }
                if hasattr(self.curves_panel, 'update_plot'):
                    self.curves_panel.update_plot()
            
            # Reset self.curves to match
            self.curves = {
                "r": [(0, 0), (128, 128), (255, 255)],
                "g": [(0, 0), (128, 128), (255, 255)],
                "b": [(0, 0), (128, 128), (255, 255)]
            }
            
            # Reset mask editing state and disable mask mode if active
            masks_panel = self.panel_manager.get_panel("masks")
            if masks_panel:
                # Disable mask editing if currently active
                if masks_panel.mask_editing_enabled:
                    masks_panel._disable_mask_editing()
                
                # Ensure mask section is toggled off
                if UIStateManager.safe_item_exists("mask_section_toggle"):
                    UIStateManager.safe_set_value("mask_section_toggle", False)
                    UIStateManager.safe_configure_item("mask_panel", show=False)
            
            # Reset the image processor to original state if available
            if (self.main_window and 
                hasattr(self.main_window, 'app_service') and 
                self.main_window.app_service and
                hasattr(self.main_window.app_service, 'image_service') and
                self.main_window.app_service.image_service and
                hasattr(self.main_window.app_service.image_service, 'image_processor') and
                self.main_window.app_service.image_service.image_processor):
                
                processor = self.main_window.app_service.image_service.image_processor
                if hasattr(processor, 'reset'):
                    processor.reset()
            
            # Re-enable callbacks
            self.enable_parameter_callbacks()
            
            # Trigger a parameter change to update the image
            if self.callback:
                self.callback(None, None, None)
            
        except Exception as e:
            print(f"Error resetting parameters: {e}")
            traceback.print_exc()
            # Make sure to re-enable callbacks even if there's an error
            self.enable_parameter_callbacks()

    def get_all_parameters(self) -> Dict[str, Any]:
        """Get all tool parameters from all panels."""
        params = {}
        
        # Get parameters from each panel
        for panel_name, panel in self.panel_manager.panels.items():
            if hasattr(panel, 'get_parameters'):
                panel_params = panel.get_parameters()
                params.update(panel_params)
        
        # Get curves data if available
        if self.curves_panel:
            params['curves'] = {
                'curves': self.curves,
                'interpolation_mode': getattr(self.curves_panel, 'interpolation_mode', 'Linear')
            }
        
        return params
    
    def handle_resize(self):
        """Handle window resize events and update histogram width."""
        if self.histogram_panel:
            new_width = self._get_available_width()
            self.histogram_panel.set_width(new_width)
    
    def update_histogram(self, image):
        """Update histogram with new image data."""
        if self.histogram_panel:
            self.histogram_panel.update_histogram(image)
    
    def update_masks(self, masks, mask_names=None):
        """Update masks in the masks panel."""
        try:
            masks_panel = self.panel_manager.get_panel("masks")
            if masks_panel:
                masks_panel.update_masks(masks, mask_names)
        except Exception as e:
            print(f"Error in ToolPanel.update_masks: {e}")
            traceback.print_exc()
    
    def toggle_crop_mode(self, sender, app_data, user_data):
        """Toggle crop mode (delegated to crop panel)."""
        crop_panel = self.panel_manager.get_panel("crop")
        if crop_panel:
            crop_panel.toggle_crop_mode(sender, app_data, user_data)
    
    def toggle_mask_section(self, sender, app_data, user_data):
        """Toggle mask section (delegated to masks panel)."""
        masks_panel = self.panel_manager.get_panel("masks")
        if masks_panel:
            masks_panel.toggle_mask_section(sender, app_data, user_data)
    
    def get_selected_mask_indices(self):
        """Get currently selected mask indices."""
        masks_panel = self.panel_manager.get_panel("masks")
        if masks_panel:
            return list(masks_panel.selected_mask_indices)
        return []
    
    def set_segmentation_mode(self, enabled: bool):
        """Set segmentation mode state."""
        masks_panel = self.panel_manager.get_panel("masks")
        if masks_panel:
            masks_panel.set_segmentation_mode(enabled)
    
    def enable_parameter_callbacks(self):
        """Enable parameter change callbacks on all panels."""
        # Enable callbacks on component panels
        for panel in self.panel_manager.panels.values():
            if hasattr(panel, 'enable_callbacks'):
                panel.enable_callbacks()
                
        # Enable callbacks on specialized panels
        if self.curves_panel and hasattr(self.curves_panel, 'enable_callbacks'):
            self.curves_panel.enable_callbacks()
    
    def disable_parameter_callbacks(self):
        """Disable parameter change callbacks on all panels."""
        # Disable callbacks on component panels
        for panel in self.panel_manager.panels.values():
            if hasattr(panel, 'disable_callbacks'):
                panel.disable_callbacks()
                
        # Disable callbacks on specialized panels  
        if self.curves_panel and hasattr(self.curves_panel, 'disable_callbacks'):
            self.curves_panel.disable_callbacks()
    
    def cleanup(self):
        """Cleanup all panel components and resources."""
        try:
            # Cleanup specialized panels
            if self.histogram_panel and hasattr(self.histogram_panel, 'cleanup'):
                self.histogram_panel.cleanup()
            
            if self.curves_panel and hasattr(self.curves_panel, 'cleanup'):
                self.curves_panel.cleanup()
            
            # Cleanup all registered panels
            for panel_name, panel in self.panel_manager.panels.items():
                if hasattr(panel, 'cleanup'):
                    panel.cleanup()
            
        except Exception as e:
            print(f"Error during ModularToolPanel cleanup: {e}")

    # Legacy compatibility methods
    def _param_changed(self, sender, app_data, user_data):
        """Handle parameter changes (legacy compatibility)."""
        if self.callback:
            self.callback(sender, app_data, user_data)
    
    def _update_crop_rotate(self, sender, app_data, user_data):
        """Update crop/rotate (legacy compatibility)."""
        crop_panel = self.panel_manager.get_panel("crop")
        if crop_panel:
            crop_panel._update_crop_rotate(sender, app_data, user_data)
    
    def _set_max_rect(self, sender, app_data, user_data):
        """Set max rectangle (legacy compatibility)."""
        crop_panel = self.panel_manager.get_panel("crop")
        if crop_panel:
            crop_panel._set_max_rect(sender, app_data, user_data)
    
    def _crop_image(self, sender, app_data, user_data):
        """Crop image (legacy compatibility)."""
        crop_panel = self.panel_manager.get_panel("crop")
        if crop_panel:
            crop_panel._crop_image(sender, app_data, user_data)
    
    def _register_all_deferred_callbacks(self):
        """Register all deferred callbacks and double-click handlers for all panels."""
        # Register deferred callbacks for all component panels
        for panel_name, panel in self.panel_manager.panels.items():
            if hasattr(panel, 'register_deferred_callbacks'):
                panel.register_deferred_callbacks()
    
    def _setup_global_double_click_handler(self):
        """Setup global double-click handler for slider reset functionality."""
        
        def global_double_click_handler(sender, app_data, user_data):
            """Handle global double-click events for slider resets."""
            # Collect all slider defaults from all panels
            all_slider_defaults = {}
            for panel_name, panel in self.panel_manager.panels.items():
                if hasattr(panel, 'get_slider_defaults'):
                    slider_defaults = panel.get_slider_defaults()
                    all_slider_defaults.update(slider_defaults)
            
            # Check if any slider is being hovered and reset it
            for tag, default_value in all_slider_defaults.items():
                if dpg.does_item_exist(tag) and dpg.is_item_hovered(tag):
                    # Set reset flag in EventCoordinator to prevent interference
                    if (hasattr(self, 'main_window') and self.main_window and 
                        hasattr(self.main_window, 'event_coordinator') and 
                        self.main_window.event_coordinator):
                        self.main_window.event_coordinator.set_resetting_parameter(True)
                    
                    try:
                        # Set override value in EventCoordinator BEFORE setting UI value
                        if (hasattr(self, 'main_window') and self.main_window and 
                            hasattr(self.main_window, 'event_coordinator') and 
                            self.main_window.event_coordinator):
                            self.main_window.event_coordinator.set_parameter_override(tag, default_value)
                        
                        # Reset the slider value
                        dpg.set_value(tag, default_value)
                        
                        # DIRECT PROCESSOR UPDATE - bypass normal parameter collection
                        if (hasattr(self, 'main_window') and self.main_window and 
                            hasattr(self.main_window, 'app_service') and self.main_window.app_service and
                            hasattr(self.main_window.app_service, 'image_service') and 
                            self.main_window.app_service.image_service and
                            self.main_window.app_service.image_service.image_processor):
                            
                            processor = self.main_window.app_service.image_service.image_processor
                            
                            # Use the EventCoordinator's force method for better reliability
                            if (hasattr(self.main_window, 'event_coordinator') and 
                                self.main_window.event_coordinator):
                                success = self.main_window.event_coordinator.force_processor_parameter(tag, default_value)
                                if not success:
                                    # Fallback to direct attribute setting
                                    if hasattr(processor, tag):
                                        setattr(processor, tag, default_value)
                            
                            # Force image processing with current parameters
                            processed_image = processor.apply_all_edits()
                            if processed_image is not None:
                                # Update the display directly
                                if (hasattr(self.main_window, 'event_coordinator') and 
                                    self.main_window.event_coordinator):
                                    self.main_window.event_coordinator._update_image_display(processed_image)
                                    self.main_window.event_coordinator._update_histogram(processed_image)
                        
                        # Give time for UI to settle
                        import time
                        time.sleep(0.05)
                        
                    except Exception as e:
                        traceback.print_exc()
                    finally:
                        # Clear reset flag in EventCoordinator
                        if (hasattr(self, 'main_window') and self.main_window and 
                            hasattr(self.main_window, 'event_coordinator') and 
                            self.main_window.event_coordinator):
                            self.main_window.event_coordinator.set_resetting_parameter(False)
                        
                        # Final verification and force reset if needed
                        import time
                        time.sleep(0.1)
                        final_ui_value = dpg.get_value(tag)
                        
                        if final_ui_value != default_value:
                            # Disable the slider callback temporarily
                            if dpg.does_item_exist(tag):
                                dpg.configure_item(tag, callback=None)
                                dpg.set_value(tag, default_value)
                                time.sleep(0.05)
                                # Re-enable callback
                                dpg.configure_item(tag, callback=self._param_changed)
                            
                            # Also force processor value one more time
                            if (hasattr(self, 'main_window') and self.main_window and 
                                hasattr(self.main_window, 'event_coordinator') and 
                                self.main_window.event_coordinator):
                                self.main_window.event_coordinator.force_processor_parameter(tag, default_value)
                    
                    return  # Only reset one slider per double-click
        
        try:
            # Add global double-click handler
            with dpg.handler_registry():
                dpg.add_mouse_double_click_handler(callback=global_double_click_handler)
        except Exception as e:
            pass

# Legacy compatibility - alias the new class to the old name
# This allows existing code to use the new modular implementation
ToolPanel = ModularToolPanel
