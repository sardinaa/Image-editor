"""
EventHandlers - Low-Level Input Event Handling System
Handles direct DearPyGUI input events (mouse, keyboard) and UI interactions.
This system focuses on input processing, coordinate transformations, and UI element manipulation.
Note: Business logic coordination is handled by ui/services/event_coordinator.py
"""
import dearpygui.dearpygui as dpg
from typing import Tuple, Optional, Callable


class EventHandlers:
    """
    Low-level input event handling system for DearPyGUI events.
    
    Handles direct user input events and coordinates with the main window for 
    UI interactions. This class focuses on input processing, coordinate 
    transformations, and direct UI element manipulation.
    
    Responsibilities:
    - Direct DearPyGUI callback handling (mouse, keyboard events)
    - Mouse position calculations and coordinate transformations
    - Bounding box interaction logic (resize, move, create)
    - UI mode detection and state management
    - Direct manipulation of UI elements and visual feedback
    
    Note: Business logic coordination is handled by EventCoordinator service.
    """
    
    def __init__(self, app_service):
        self.app_service = app_service
        self.main_window = None  # Will be set when app_service.setup_ui() is called
    
    @property
    def _main_window(self):
        """Get main window from app service."""
        if hasattr(self.app_service, 'main_window'):
            return self.app_service.main_window
        return None
    
    def _safe_item_check(self, item_name):
        """Safely check if DPG item exists."""
        try:
            return dpg.does_item_exist(item_name)
        except:
            return False
    
    def _check_crop_mode_active(self):
        """Check if crop mode is currently active."""
        return (self._safe_item_check("crop_mode") and 
                dpg.get_value("crop_mode"))
    
    def _check_image_plot_hovered(self):
        """Check if mouse is hovering over the image plot."""
        main_window = self._main_window
        if not main_window:
            return False
        
        plot_tag = getattr(main_window, 'image_plot_tag', "image_plot")
        return (dpg.does_item_exist(plot_tag) and 
                dpg.is_item_hovered(plot_tag))
    
    def on_mouse_down(self, sender, app_data) -> bool:
        """Handle mouse down events with centralized logic."""
        main_window = self._main_window
        if not main_window:
            return False
        
        print(f"=== EventHandlers: Mouse down event ===")
        print(f"Crop mode active: {self._check_crop_mode_active()}")
        print(f"Image plot hovered: {self._check_image_plot_hovered()}")
        
        # Priority 1: Crop mode handling - centralized logic
        if (self._check_crop_mode_active() and 
            hasattr(main_window, 'crop_rotate_ui') and 
            main_window.crop_rotate_ui and
            self._check_image_plot_hovered()):
            
            print("EventHandlers: Handling crop mode mouse down")
            return self._handle_crop_mouse_down(main_window.crop_rotate_ui.bbox_renderer, sender, app_data)
        
        # Priority 2: Segmentation mode handling
        if (hasattr(main_window, 'segmentation_mode') and 
            main_window.segmentation_mode and
            hasattr(main_window, 'segmentation_bbox_renderer') and
            main_window.segmentation_bbox_renderer and
            self._check_image_plot_hovered()):
            
            print("EventHandlers: Handling segmentation mode")
            return self._handle_crop_mouse_down(main_window.segmentation_bbox_renderer, sender, app_data)
        
        # Priority 3: Box selection mode handling  
        if (hasattr(main_window, 'box_selection_mode') and 
            main_window.box_selection_mode and 
            self._check_image_plot_hovered()):
            
            print("EventHandlers: Handling box selection mode")
            main_window.box_selection_active = True
            plot_pos = dpg.get_plot_mouse_pos()
            main_window.box_start = list(plot_pos)
            main_window.box_end = list(plot_pos)
            
            if (hasattr(main_window, 'box_rect_tag') and 
                dpg.does_item_exist(main_window.box_rect_tag) and
                hasattr(main_window, 'draw_list_tag')):
                dpg.configure_item(main_window.draw_list_tag, show=True)
                if hasattr(main_window, 'update_box_rectangle'):
                    main_window.update_box_rectangle()
            return True
        
        return False
    
    def on_mouse_drag(self, sender, app_data) -> bool:
        """Handle mouse drag events with centralized logic."""
        main_window = self._main_window
        if not main_window:
            return False
        
        # Priority 1: Crop mode handling
        if (self._check_crop_mode_active() and 
            hasattr(main_window, 'crop_rotate_ui') and 
            main_window.crop_rotate_ui and
            self._check_image_plot_hovered()):
            
            # Delegate to centralized crop handling
            if (hasattr(main_window.crop_rotate_ui, 'bbox_renderer') and 
                main_window.crop_rotate_ui.bbox_renderer):
                result = self._handle_crop_mouse_drag(main_window.crop_rotate_ui.bbox_renderer, sender, app_data)
                if result:
                    print("EventHandlers: BBox drag handled")
                return result
        
        # Priority 2: Segmentation mode handling
        if (hasattr(main_window, 'segmentation_mode') and 
            main_window.segmentation_mode and
            hasattr(main_window, 'segmentation_bbox_renderer') and
            main_window.segmentation_bbox_renderer):
            
            return self._handle_crop_mouse_drag(main_window.segmentation_bbox_renderer, sender, app_data)
        
        # Priority 3: Box selection mode handling
        if (hasattr(main_window, 'box_selection_active') and 
            main_window.box_selection_active and 
            self._check_image_plot_hovered()):
            
            plot_pos = dpg.get_plot_mouse_pos()
            main_window.box_end = list(plot_pos)
            if hasattr(main_window, 'update_box_rectangle'):
                main_window.update_box_rectangle()
            return True
        
        return False
    
    def on_mouse_release(self, sender, app_data) -> bool:
        """Handle mouse release events with centralized logic."""
        main_window = self._main_window
        if not main_window:
            return False
        
        # Priority 1: Crop mode handling
        if (self._check_crop_mode_active() and 
            hasattr(main_window, 'crop_rotate_ui') and 
            main_window.crop_rotate_ui):
            
            # Delegate to centralized crop handling
            if (hasattr(main_window.crop_rotate_ui, 'bbox_renderer') and 
                main_window.crop_rotate_ui.bbox_renderer):
                result = self._handle_crop_mouse_release(main_window.crop_rotate_ui.bbox_renderer, sender, app_data)
                
                # Ensure crop UI drag state is cleared
                if (hasattr(main_window.crop_rotate_ui, 'drag_active') and
                    main_window.crop_rotate_ui.drag_active and 
                    hasattr(main_window.crop_rotate_ui.bbox_renderer, 'is_dragging') and
                    not main_window.crop_rotate_ui.bbox_renderer.is_dragging):
                    main_window.crop_rotate_ui.drag_active = False
                
                return result
        
        # Priority 2: Segmentation mode handling
        if (hasattr(main_window, 'segmentation_mode') and 
            main_window.segmentation_mode and
            hasattr(main_window, 'segmentation_bbox_renderer') and
            main_window.segmentation_bbox_renderer):
            
            if self._handle_crop_mouse_release(main_window.segmentation_bbox_renderer, sender, app_data):
                return True
        
        # Priority 3: Box selection mode handling
        if (hasattr(main_window, 'box_selection_active') and 
            main_window.box_selection_active):
            
            main_window.box_selection_active = False
            
            # Calculate box dimensions
            if (hasattr(main_window, 'box_start') and 
                hasattr(main_window, 'box_end')):
                
                x1, y1 = main_window.box_start
                x2, y2 = main_window.box_end
                
                box_width = abs(x2 - x1)
                box_height = abs(y2 - y1)
                
                if box_width > 10 and box_height > 10:  # Minimum box size
                    # Convert to standard box format [x1, y1, x2, y2]
                    box = [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
                    print(f"Box selection completed: {box}")
                    
                    # Trigger segmentation through app service
                    if self.app_service and hasattr(self.app_service, 'perform_box_segmentation'):
                        self.app_service.perform_box_segmentation(box)
            
            # Hide the selection rectangle
            if (hasattr(main_window, 'draw_list_tag') and 
                dpg.does_item_exist(main_window.draw_list_tag)):
                dpg.configure_item(main_window.draw_list_tag, show=False)
            return True
        
        return False
    
    def on_mouse_wheel(self, sender, app_data) -> bool:
        """Handle mouse wheel events for zooming with centralized logic."""
        main_window = self._main_window
        if not main_window:
            return False
        
        # Only zoom when mouse is over the image plot and we have an image loaded
        if (self._check_image_plot_hovered() and 
            hasattr(main_window, 'crop_rotate_ui') and 
            main_window.crop_rotate_ui):
            
            # Get axis tags - try multiple possible names
            x_axis_tag = getattr(main_window, 'x_axis_tag', 'x_axis')
            y_axis_tag = getattr(main_window, 'y_axis_tag', 'y_axis')
            
            if (dpg.does_item_exist(x_axis_tag) and 
                dpg.does_item_exist(y_axis_tag)):
                
                # Get current plot limits
                x_limits = dpg.get_axis_limits(x_axis_tag)
                y_limits = dpg.get_axis_limits(y_axis_tag)
                
                # Calculate zoom factor based on wheel direction
                # Positive app_data = scroll up = zoom in (smaller zoom factor)
                # Negative app_data = scroll down = zoom out (larger zoom factor)
                zoom_factor = 0.9 if app_data > 0 else 1.1
                
                # Calculate new limits centered around current view center
                x_range = x_limits[1] - x_limits[0]
                y_range = y_limits[1] - y_limits[0]
                x_center = (x_limits[0] + x_limits[1]) / 2
                y_center = (y_limits[0] + y_limits[1]) / 2
                
                new_x_range = x_range * zoom_factor
                new_y_range = y_range * zoom_factor
                
                # Set new limits
                dpg.set_axis_limits(x_axis_tag, x_center - new_x_range/2, x_center + new_x_range/2)
                dpg.set_axis_limits(y_axis_tag, y_center - new_y_range/2, y_center + new_y_range/2)
                
                return True
        
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


# Centralized mouse handling methods for bounding box interactions
    def _handle_crop_mouse_down(self, bbox_renderer, sender, app_data) -> bool:
        """Centralized mouse down handling for bounding box renderers."""
        if not bbox_renderer:
            return False
            
        mouse_pos = dpg.get_mouse_pos()
        texture_x, texture_y = bbox_renderer.screen_to_texture_coords(mouse_pos[0], mouse_pos[1])
        
        print(f"=== EventHandlers: Centralized mouse down ===")
        print(f"Screen pos: {mouse_pos}")
        print(f"Texture coords: ({texture_x:.1f}, {texture_y:.1f})")
        
        # If no bounding box exists, start creating a new one
        if not bbox_renderer.bounding_box:
            from .bounding_box_renderer import BoundingBox, DragMode, HandleType
            
            # Create a new bounding box starting from this point
            bbox_renderer.bounding_box = BoundingBox(texture_x, texture_y, 10, 10)
            bbox_renderer.is_dragging = True
            bbox_renderer.drag_mode = DragMode.RESIZE
            bbox_renderer.drag_handle = HandleType.BOTTOM_RIGHT
            bbox_renderer.drag_start_mouse = (texture_x, texture_y)
            bbox_renderer.drag_start_box = bbox_renderer.bounding_box.copy()
            
            print(f"EventHandlers: Created new bounding box at {texture_x}, {texture_y}")
            
            if bbox_renderer.on_start_drag_callback:
                bbox_renderer.on_start_drag_callback(bbox_renderer.bounding_box.copy())
            return True
        
        # Check if Control key is pressed for move mode
        is_ctrl_pressed = dpg.is_key_down(dpg.mvKey_LControl) or dpg.is_key_down(dpg.mvKey_RControl)
        
        # Check for handle hits first (only for resize mode)
        if not is_ctrl_pressed:
            hit_handle = bbox_renderer.hit_test_handles(texture_x, texture_y)
            if hit_handle:
                from .bounding_box_renderer import DragMode
                
                bbox_renderer.is_dragging = True
                bbox_renderer.drag_mode = DragMode.RESIZE
                bbox_renderer.drag_handle = hit_handle
                bbox_renderer.drag_start_mouse = (texture_x, texture_y)
                bbox_renderer.drag_start_box = bbox_renderer.bounding_box.copy()
                
                if bbox_renderer.on_start_drag_callback:
                    bbox_renderer.on_start_drag_callback(bbox_renderer.bounding_box.copy())
                return True
        
        # Check if clicking inside the box
        if bbox_renderer.bounding_box.contains_point(texture_x, texture_y):
            from .bounding_box_renderer import DragMode
            
            bbox_renderer.is_dragging = True
            
            if is_ctrl_pressed:
                # Control + click = move mode
                bbox_renderer.drag_mode = DragMode.MOVE
                bbox_renderer.drag_offset = (texture_x - bbox_renderer.bounding_box.x, texture_y - bbox_renderer.bounding_box.y)
            else:
                # Regular click = resize mode (drag the nearest edge/corner)
                bbox_renderer.drag_mode = DragMode.RESIZE
                bbox_renderer.drag_handle = bbox_renderer._get_nearest_handle(texture_x, texture_y)
                bbox_renderer.drag_start_mouse = (texture_x, texture_y)
            
            bbox_renderer.drag_start_box = bbox_renderer.bounding_box.copy()
            
            if bbox_renderer.on_start_drag_callback:
                bbox_renderer.on_start_drag_callback(bbox_renderer.bounding_box.copy())
            return True
        
        return False
    
    def _handle_crop_mouse_drag(self, bbox_renderer, sender, app_data) -> bool:
        """Centralized mouse drag handling for bounding box renderers."""
        if not bbox_renderer:
            return False
            
        if not bbox_renderer.is_dragging or not bbox_renderer.bounding_box or not bbox_renderer.drag_start_box:
            return False
        
        # Additional safety check: ensure left mouse button is still pressed
        if not dpg.is_mouse_button_down(dpg.mvMouseButton_Left):
            print("EventHandlers: Mouse button released during drag - ending drag")
            self._end_bbox_drag(bbox_renderer)
            return False
        
        mouse_pos = dpg.get_mouse_pos()
        texture_x, texture_y = bbox_renderer.screen_to_texture_coords(mouse_pos[0], mouse_pos[1])
        
        from .bounding_box_renderer import DragMode
        
        if bbox_renderer.drag_mode == DragMode.MOVE:
            # Move the entire box
            new_x = texture_x - bbox_renderer.drag_offset[0]
            new_y = texture_y - bbox_renderer.drag_offset[1]
            
            bbox_renderer.bounding_box.x = new_x
            bbox_renderer.bounding_box.y = new_y
            
            # Clamp to bounds if set
            if bbox_renderer.bounds:
                bbox_renderer.bounding_box.clamp_to_bounds(bbox_renderer.bounds)
        
        elif bbox_renderer.drag_mode == DragMode.RESIZE:
            # Resize based on the handle being dragged
            bbox_renderer._resize_box(bbox_renderer.drag_handle, texture_x, texture_y)
        
        # Call change callback during drag for real-time visual feedback
        if bbox_renderer.on_change_callback:
            bbox_renderer.on_change_callback(bbox_renderer.bounding_box.copy())
        
        return True
    
    def _handle_crop_mouse_release(self, bbox_renderer, sender, app_data) -> bool:
        """Centralized mouse release handling for bounding box renderers."""
        if not bbox_renderer:
            return False
            
        if not bbox_renderer.is_dragging:
            return False
        
        was_dragging = bbox_renderer.is_dragging
        
        # End the drag operation
        self._end_bbox_drag(bbox_renderer)
        
        if was_dragging and bbox_renderer.bounding_box:
            # Ensure the bounding box has valid dimensions
            self._normalize_bbox_dimensions(bbox_renderer)
            
            # Call end drag callback only if dimensions are meaningful
            if (bbox_renderer.on_end_drag_callback and 
                bbox_renderer.bounding_box.width > 0 and 
                bbox_renderer.bounding_box.height > 0):
                print(f"EventHandlers: End drag with box {bbox_renderer.bounding_box.x},{bbox_renderer.bounding_box.y},{bbox_renderer.bounding_box.width}x{bbox_renderer.bounding_box.height}")
                bbox_renderer.on_end_drag_callback(bbox_renderer.bounding_box.copy())
        
        return was_dragging
    
    def _end_bbox_drag(self, bbox_renderer):
        """Helper method to end bounding box drag operation."""
        from .bounding_box_renderer import DragMode
        
        bbox_renderer.is_dragging = False
        bbox_renderer.drag_mode = DragMode.NONE
        bbox_renderer.drag_handle = None
        bbox_renderer.drag_start_box = None
    
    def _normalize_bbox_dimensions(self, bbox_renderer):
        """Helper method to normalize bounding box dimensions."""
        if not bbox_renderer.bounding_box:
            return
            
        # Normalize negative width/height
        if bbox_renderer.bounding_box.width < 0:
            bbox_renderer.bounding_box.x += bbox_renderer.bounding_box.width
            bbox_renderer.bounding_box.width = abs(bbox_renderer.bounding_box.width)
        
        if bbox_renderer.bounding_box.height < 0:
            bbox_renderer.bounding_box.y += bbox_renderer.bounding_box.height
            bbox_renderer.bounding_box.height = abs(bbox_renderer.bounding_box.height)
        
        # Enforce minimum dimensions
        if bbox_renderer.bounding_box.width < 5:
            bbox_renderer.bounding_box.width = 5
        
        if bbox_renderer.bounding_box.height < 5:
            bbox_renderer.bounding_box.height = 5


