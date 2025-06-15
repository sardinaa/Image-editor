"""
Event handling system for the main window.
Separates input handling from core window logic.
"""
import dearpygui.dearpygui as dpg
from typing import Tuple, Optional, Callable


class EventHandlers:
    """Central event handling system that coordinates with the application service."""
    
    def __init__(self, app_service):
        self.app_service = app_service
        self.main_window = None  # Will be set when app_service.setup_ui() is called
    
    @property
    def _main_window(self):
        """Get main window from app service."""
        if hasattr(self.app_service, 'main_window'):
            return self.app_service.main_window
        return None
    
    def on_mouse_down(self, sender, app_data) -> bool:
        """Handle mouse down events."""
        main_window = self._main_window
        if not main_window:
            return False
        
        # Box selection mode handling
        if (hasattr(main_window, 'box_selection_mode') and 
            main_window.box_selection_mode and 
            dpg.is_item_hovered("image_plot")):
            
            main_window.box_selection_active = True
            plot_pos = dpg.get_plot_mouse_pos()
            main_window.box_start = list(plot_pos)
            main_window.box_end = list(plot_pos)
            
            if dpg.does_item_exist(main_window.box_rect_tag):
                dpg.configure_item(main_window.draw_list_tag, show=True)
                main_window.update_box_rectangle()
            return True
        
        # Delegate to main window if available
        if hasattr(main_window, 'on_mouse_down'):
            return main_window.on_mouse_down(sender, app_data)
        
        return False
    
    def on_mouse_drag(self, sender, app_data) -> bool:
        """Handle mouse drag events."""
        main_window = self._main_window
        if not main_window:
            return False
        
        if (hasattr(main_window, 'box_selection_active') and 
            main_window.box_selection_active and 
            dpg.is_item_hovered("image_plot")):
            
            plot_pos = dpg.get_plot_mouse_pos()
            main_window.box_end = list(plot_pos)
            main_window.update_box_rectangle()
            return True
        
        # Delegate to main window if available
        if hasattr(main_window, 'on_mouse_drag'):
            return main_window.on_mouse_drag(sender, app_data)
        
        return False
    
    def on_mouse_release(self, sender, app_data) -> bool:
        """Handle mouse release events."""
        main_window = self._main_window
        if not main_window:
            return False
        
        if (hasattr(main_window, 'box_selection_active') and 
            main_window.box_selection_active):
            
            main_window.box_selection_active = False
            
            # Calculate box dimensions
            x1, y1 = main_window.box_start
            x2, y2 = main_window.box_end
            
            box_width = abs(x2 - x1)
            box_height = abs(y2 - y1)
            
            if box_width > 10 and box_height > 10:  # Minimum box size
                # Convert to standard box format [x1, y1, x2, y2]
                box = [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
                print(f"Box selection completed: {box}")
                
                # Trigger segmentation through app service
                if self.app_service:
                    self.app_service.perform_box_segmentation(box)
            
            # Hide the selection rectangle
            if hasattr(main_window, 'draw_list_tag') and dpg.does_item_exist(main_window.draw_list_tag):
                dpg.configure_item(main_window.draw_list_tag, show=False)
            return True
        
        # Delegate to main window if available
        if hasattr(main_window, 'on_mouse_release'):
            return main_window.on_mouse_release(sender, app_data)
        
        return False
    
    def on_mouse_wheel(self, sender, app_data) -> bool:
        """Handle mouse wheel events for zooming."""
        main_window = self._main_window
        if not main_window:
            return False
        
        if (dpg.does_item_exist("image_plot") and 
            hasattr(main_window, 'crop_rotate_ui') and 
            main_window.crop_rotate_ui and 
            dpg.is_item_hovered("image_plot")):
            
            # Get current plot limits
            x_limits = dpg.get_axis_limits("x_axis")
            y_limits = dpg.get_axis_limits("y_axis")
            
            # Calculate zoom factor
            zoom_factor = 0.9 if app_data > 0 else 1.1
            
            # Calculate new limits
            x_range = x_limits[1] - x_limits[0]
            y_range = y_limits[1] - y_limits[0]
            x_center = (x_limits[0] + x_limits[1]) / 2
            y_center = (y_limits[0] + y_limits[1]) / 2
            
            new_x_range = x_range * zoom_factor
            new_y_range = y_range * zoom_factor
            
            # Set new limits
            dpg.set_axis_limits("x_axis", x_center - new_x_range/2, x_center + new_x_range/2)
            dpg.set_axis_limits("y_axis", y_center - new_y_range/2, y_center + new_y_range/2)
            
            return True
        
        # Delegate to main window if available
        if hasattr(main_window, 'on_mouse_wheel'):
            return main_window.on_mouse_wheel(sender, app_data)
        
        return False
    
    def on_key_press(self, sender, app_data) -> bool:
        """Handle keyboard events."""
        main_window = self._main_window
        if not main_window:
            return False
        
        key = app_data
        
        # Default key handling
        if key == dpg.mvKey_Escape:
            return self._handle_escape()
        elif key == dpg.mvKey_Delete:
            return self._handle_delete()
        
        # Delegate to main window if available
        if hasattr(main_window, 'on_key_press'):
            return main_window.on_key_press(sender, app_data)
        
        return False
    
    def _handle_escape(self) -> bool:
        """Handle escape key press."""
        main_window = self._main_window
        if not main_window:
            return False
        
        # Cancel any active operations
        if hasattr(main_window, 'box_selection_active') and main_window.box_selection_active:
            main_window.box_selection_active = False
            if hasattr(main_window, 'draw_list_tag') and dpg.does_item_exist(main_window.draw_list_tag):
                dpg.configure_item(main_window.draw_list_tag, show=False)
            return True
        
        if hasattr(main_window, 'segmentation_mode') and main_window.segmentation_mode:
            if hasattr(main_window, 'cancel_segmentation_selection'):
                main_window.cancel_segmentation_selection()
                return True
        
        return False
    
    def _handle_delete(self) -> bool:
        """Handle delete key press."""
        main_window = self._main_window
        if not main_window:
            return False
        
        # Delete selected masks through app service
        if (hasattr(main_window, 'tool_panel') and 
            main_window.tool_panel and 
            hasattr(main_window.tool_panel, 'get_selected_mask_indices')):
            
            selected_indices = main_window.tool_panel.get_selected_mask_indices()
            for mask_index in sorted(selected_indices, reverse=True):
                if self.app_service:
                    self.app_service.delete_mask(mask_index)
            return True
        
        return False


# Legacy compatibility classes (kept for backward compatibility)
class MouseEventHandler:
    """Legacy mouse event handler (deprecated - use EventHandlers instead)."""
    
    def __init__(self, main_window):
        self.main_window = main_window
        self.mouse_handlers = {}


class KeyboardEventHandler:
    """Legacy keyboard event handler (deprecated - use EventHandlers instead)."""
    
    def __init__(self, main_window):
        self.main_window = main_window
        self.key_bindings = {}


class SegmentationEventHandler:
    """Legacy segmentation event handler (deprecated - use EventHandlers instead)."""
    
    def __init__(self, main_window):
        self.main_window = main_window


class EventManager:
    """Legacy event manager (deprecated - use EventHandlers instead)."""
    
    def __init__(self, main_window):
        self.main_window = main_window
        """Get the segmentation event handler."""
        return self.segmentation_handler
