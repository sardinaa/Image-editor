"""
Base panel class for tool panel components.
Provides common functionality for all tool panel sections.
"""
import dearpygui.dearpygui as dpg
from typing import Dict, Any, Callable, Optional
from utils.ui_helpers import UIStateManager


class BasePanel:
    """Base class for all tool panel components."""
    
    def __init__(self, callback: Optional[Callable] = None, main_window=None):
        self.callback = callback
        self.main_window = main_window
        self.panel_tag = None
        self.is_visible = True
        self.parameters = {}
        self._slider_defaults = {}
        self._deferred_callbacks = []  # Store slider tags for deferred callback registration
        self._callbacks_enabled = True  # Track whether callbacks are enabled
    
    def draw(self) -> None:
        """Draw the panel UI. Override in subclasses."""
        pass
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get current parameter values. Override in subclasses."""
        return self.parameters
    
    def set_parameters(self, params: Dict[str, Any]) -> None:
        """Set parameter values. Override in subclasses."""
        for param_name, value in params.items():
            if UIStateManager.safe_item_exists(param_name):
                UIStateManager.safe_set_value(param_name, value)
    
    def _param_changed(self, sender, app_data, user_data):
        """Handle parameter change. Can be overridden."""
        # Check if callbacks are enabled on this panel
        if hasattr(self, '_callbacks_enabled') and not self._callbacks_enabled:
            return
            
        # Check if callbacks are enabled in main window
        if (self.main_window and 
            hasattr(self.main_window, '_callbacks_enabled') and 
            not self.main_window._callbacks_enabled):
            return
            
        if self.callback:
            self.callback(sender, app_data, user_data)
    
    def show(self) -> None:
        """Show the panel."""
        if self.panel_tag:
            UIStateManager.safe_configure_item(self.panel_tag, show=True)
        self.is_visible = True
    
    def hide(self) -> None:
        """Hide the panel."""
        if self.panel_tag:
            UIStateManager.safe_configure_item(self.panel_tag, show=False)
        self.is_visible = False
    
    def toggle(self) -> None:
        """Toggle panel visibility."""
        if self.is_visible:
            self.hide()
        else:
            self.show()
    
    def _create_section_header(self, title: str, color: list = None) -> None:
        """Create a section header with consistent styling."""
        if color is None:
            color = [200, 200, 200]
        
        dpg.add_spacer(height=2)
        dpg.add_text(title, color=color)
        dpg.add_spacer(height=1)
    
    def _create_slider_int(self, label: str, tag: str, default: int, 
                          min_val: int, max_val: int, height: int = 18) -> None:
        """Create a standardized integer slider."""
        dpg.add_slider_int(
            label=label,
            tag=tag,
            default_value=default,
            min_value=min_val,
            max_value=max_val,
            height=height,
            callback=self._param_changed
        )
        # Add double-click handler to reset to default value
        self._add_double_click_reset_handler(tag, default)
    
    def _create_slider_float(self, label: str, tag: str, default: float,
                           min_val: float, max_val: float, height: int = 18) -> None:
        """Create a standardized float slider."""
        dpg.add_slider_float(
            label=label,
            tag=tag,
            default_value=default,
            min_value=min_val,
            max_value=max_val,
            height=height,
            callback=self._param_changed
        )
        # Add double-click handler to reset to default value
        self._add_double_click_reset_handler(tag, default)
    
    def _create_checkbox(self, label: str, tag: str, default: bool = False) -> None:
        """Create a standardized checkbox."""
        dpg.add_checkbox(
            label=label,
            tag=tag,
            default_value=default,
            callback=self._param_changed
        )
    
    def _create_button(self, label: str, callback: Callable, 
                      width: int = -1, height: int = 20, tag: str = None) -> None:
        """Create a standardized button."""
        button_kwargs = {
            'label': label,
            'callback': callback,
            'width': width,
            'height': height
        }
        
        # Only add tag if it's not None
        if tag is not None:
            button_kwargs['tag'] = tag
            
        dpg.add_button(**button_kwargs)
    
    def _add_double_click_reset_handler(self, tag: str, default_value) -> None:
        """Add a double-click handler to reset a slider to its default value."""
        # Store the slider info for global double-click handling
        if not hasattr(self, '_slider_defaults'):
            self._slider_defaults = {}
        self._slider_defaults[tag] = default_value
    
    def enable_callbacks(self) -> None:
        """Enable parameter change callbacks."""
        self._callbacks_enabled = True
            
    def disable_callbacks(self) -> None:
        """Disable parameter change callbacks."""  
        self._callbacks_enabled = False
            
    def register_deferred_callbacks(self) -> None:
        """Register callbacks for sliders that were created without callbacks."""
        for tag in self._deferred_callbacks:
            if UIStateManager.safe_item_exists(tag):
                UIStateManager.safe_configure_item(tag, callback=self._param_changed)
        self._deferred_callbacks.clear()
    
    def get_slider_defaults(self) -> Dict[str, Any]:
        """Get all slider default values for double-click reset."""
        return getattr(self, '_slider_defaults', {})
    
    def get_default_parameters(self) -> Dict[str, Any]:
        """Get the default parameter values for this panel."""
        return self.parameters.copy() if self.parameters else {}


class PanelManager:
    """Manages multiple tool panels."""
    
    def __init__(self):
        self.panels = {}
        self.panel_order = []
    
    def register_panel(self, name: str, panel: BasePanel) -> None:
        """Register a panel with the manager."""
        self.panels[name] = panel
        if name not in self.panel_order:
            self.panel_order.append(name)
    
    def get_panel(self, name: str) -> Optional[BasePanel]:
        """Get a panel by name."""
        return self.panels.get(name)
    
    def draw_all_panels(self) -> None:
        """Draw all registered panels in order."""
        for panel_name in self.panel_order:
            if panel_name in self.panels:
                panel = self.panels[panel_name]
                if panel.is_visible:
                    panel.draw()
    
    def get_all_parameters(self) -> Dict[str, Any]:
        """Get parameters from all panels."""
        all_params = {}
        for panel in self.panels.values():
            params = panel.get_parameters()
            all_params.update(params)
        return all_params
    
    def get_all_default_parameters(self) -> Dict[str, Any]:
        """Get default parameters from all panels."""
        all_defaults = {}
        for panel in self.panels.values():
            if hasattr(panel, 'get_default_parameters'):
                defaults = panel.get_default_parameters()
                all_defaults.update(defaults)
        return all_defaults
    
    def set_all_parameters(self, params: Dict[str, Any]) -> None:
        """Set parameters for all panels."""
        for panel in self.panels.values():
            panel.set_parameters(params)
    
    def show_panel(self, name: str) -> None:
        """Show a specific panel."""
        if name in self.panels:
            self.panels[name].show()
    
    def hide_panel(self, name: str) -> None:
        """Hide a specific panel."""
        if name in self.panels:
            self.panels[name].hide()
    
    def toggle_panel(self, name: str) -> None:
        """Toggle a specific panel."""
        if name in self.panels:
            self.panels[name].toggle()
