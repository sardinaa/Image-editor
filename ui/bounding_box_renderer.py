"""
Reusable bounding box component for various image editing features.
Provides functionality for creating, resizing, and moving bounding boxes with handles.
"""

import cv2
import numpy as np
import dearpygui.dearpygui as dpg
from typing import Dict, Tuple, Optional, Callable
from enum import Enum


class HandleType(Enum):
    """Enumeration for different handle types on the bounding box."""
    TOP_LEFT = "tl"
    TOP_RIGHT = "tr"
    BOTTOM_LEFT = "bl"
    BOTTOM_RIGHT = "br"
    TOP = "t"
    BOTTOM = "b"
    LEFT = "l"
    RIGHT = "r"


class DragMode(Enum):
    """Enumeration for different drag modes."""
    NONE = "none"
    MOVE = "move"
    RESIZE = "resize"


class BoundingBox:
    """
    Represents a bounding box with position and dimensions.
    """
    def __init__(self, x: float, y: float, width: float, height: float):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
    
    def copy(self) -> 'BoundingBox':
        """Create a copy of this bounding box."""
        return BoundingBox(self.x, self.y, self.width, self.height)
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary format for compatibility."""
        return {"x": self.x, "y": self.y, "w": self.width, "h": self.height}
    
    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> 'BoundingBox':
        """Create from dictionary format."""
        return cls(data["x"], data["y"], data["w"], data["h"])
    
    def contains_point(self, x: float, y: float, margin: float = 5) -> bool:
        """Check if a point is inside the bounding box with optional margin."""
        return (self.x - margin <= x <= self.x + self.width + margin and
                self.y - margin <= y <= self.y + self.height + margin)
    
    def clamp_to_bounds(self, bounds: 'BoundingBox') -> None:
        """Clamp this bounding box to stay within the given bounds."""
        # Ensure position is within bounds
        self.x = max(bounds.x, min(self.x, bounds.x + bounds.width - self.width))
        self.y = max(bounds.y, min(self.y, bounds.y + bounds.height - self.height))
        
        # Ensure size doesn't exceed bounds
        if self.x + self.width > bounds.x + bounds.width:
            self.width = bounds.x + bounds.width - self.x
        if self.y + self.height > bounds.y + bounds.height:
            self.height = bounds.y + bounds.height - self.y
    
    def get_center(self) -> Tuple[float, float]:
        """Get the center point of the bounding box."""
        return (self.x + self.width / 2, self.y + self.height / 2)
    
    def set_center(self, cx: float, cy: float) -> None:
        """Set the center point of the bounding box."""
        self.x = cx - self.width / 2
        self.y = cy - self.height / 2


class BoundingBoxRenderer:
    """
    A reusable component for rendering and interacting with bounding boxes.
    Can be used for crop rectangles, selection boxes, or any other rectangular selection needs.
    """
    
    def __init__(self, 
                 texture_width: int, 
                 texture_height: int,
                 panel_id: str = "Central Panel",
                 min_size: float = 20,
                 handle_size: float = 8,
                 handle_threshold: float = 30):
        """
        Initialize the bounding box renderer.
        
        Args:
            texture_width: Width of the texture/canvas
            texture_height: Height of the texture/canvas
            panel_id: ID of the DearPyGUI panel containing the plot
            min_size: Minimum size for width and height
            handle_size: Size of corner handles
            handle_threshold: Threshold distance for handle detection
        """
        self.texture_width = texture_width
        self.texture_height = texture_height
        self.panel_id = panel_id
        self.min_size = min_size
        self.handle_size = handle_size
        self.handle_threshold = handle_threshold
        
        # Current state
        self.bounding_box: Optional[BoundingBox] = None
        self.bounds: Optional[BoundingBox] = None
        self.is_dragging = False
        self.drag_mode = DragMode.NONE
        self.drag_handle: Optional[HandleType] = None
        self.drag_start_mouse: Tuple[float, float] = (0, 0)
        self.drag_start_box: Optional[BoundingBox] = None
        self.drag_offset: Tuple[float, float] = (0, 0)
        
        # Visual settings
        self.box_color = (0, 255, 0, 255)  # Green
        self.handle_color = (255, 0, 0, 255)  # Red
        self.box_thickness = 2
        
        # Callbacks
        self.on_change_callback: Optional[Callable[[BoundingBox], None]] = None
        self.on_start_drag_callback: Optional[Callable[[BoundingBox], None]] = None
        self.on_end_drag_callback: Optional[Callable[[BoundingBox], None]] = None
    
    def set_bounding_box(self, box: BoundingBox) -> None:
        """Set the current bounding box."""
        self.bounding_box = box.copy()
        if self.bounds:
            self.bounding_box.clamp_to_bounds(self.bounds)
    
    def set_bounds(self, bounds: BoundingBox) -> None:
        """Set the bounds within which the bounding box must stay."""
        self.bounds = bounds.copy()
        if self.bounding_box:
            self.bounding_box.clamp_to_bounds(self.bounds)
    
    def set_callbacks(self, 
                     on_change: Optional[Callable[[BoundingBox], None]] = None,
                     on_start_drag: Optional[Callable[[BoundingBox], None]] = None,
                     on_end_drag: Optional[Callable[[BoundingBox], None]] = None) -> None:
        """Set callback functions for bounding box events."""
        if on_change:
            self.on_change_callback = on_change
        if on_start_drag:
            self.on_start_drag_callback = on_start_drag
        if on_end_drag:
            self.on_end_drag_callback = on_end_drag
    
    def screen_to_texture_coords(self, screen_x: float, screen_y: float) -> Tuple[float, float]:
        """Convert screen coordinates to texture coordinates."""
        # First try to get plot mouse position (if we're in a plot)
        try:
            if dpg.does_item_exist("image_plot"):
                plot_pos = dpg.get_plot_mouse_pos()
                if plot_pos is not None:
                    # Plot coordinates are in plot coordinate system (Y increases upward)
                    # We need to convert to image coordinate system (Y increases downward)
                    texture_x = plot_pos[0]
                    texture_y = self.texture_height - plot_pos[1]  # Invert Y coordinate
                    return texture_x, texture_y
        except Exception as e:
            pass  # Fall back to panel-based conversion
        
        # Fallback to panel-based conversion if plot coordinates aren't available
        if not dpg.does_item_exist(self.panel_id):
            return screen_x, screen_y
        
        panel_pos = dpg.get_item_pos(self.panel_id)
        panel_size = dpg.get_item_rect_size(self.panel_id)
        
        if panel_size[0] <= 0 or panel_size[1] <= 0:
            return screen_x, screen_y
        
        # Convert to normalized coordinates (0-1)
        rel_x = (screen_x - panel_pos[0]) / panel_size[0]
        rel_y = (screen_y - panel_pos[1]) / panel_size[1]
        
        # Convert to texture coordinates
        texture_x = rel_x * self.texture_width
        texture_y = rel_y * self.texture_height
        
        return texture_x, texture_y
    
    def get_handle_positions(self) -> Dict[HandleType, Tuple[float, float]]:
        """Get the positions of all handles in plot coordinate system."""
        if not self.bounding_box:
            return {}
        
        box = self.bounding_box
        # Convert from image coordinates (Y increases downward) to plot coordinates (Y increases upward)
        # In image coords: (0,0) is top-left, Y increases downward
        # In plot coords: (0,0) is bottom-left, Y increases upward
        return {
            HandleType.TOP_LEFT: (box.x, self.texture_height - box.y),
            HandleType.TOP_RIGHT: (box.x + box.width, self.texture_height - box.y),
            HandleType.BOTTOM_LEFT: (box.x, self.texture_height - (box.y + box.height)),
            HandleType.BOTTOM_RIGHT: (box.x + box.width, self.texture_height - (box.y + box.height)),
            HandleType.TOP: (box.x + box.width / 2, self.texture_height - box.y),
            HandleType.BOTTOM: (box.x + box.width / 2, self.texture_height - (box.y + box.height)),
            HandleType.LEFT: (box.x, self.texture_height - (box.y + box.height / 2)),
            HandleType.RIGHT: (box.x + box.width, self.texture_height - (box.y + box.height / 2))
        }
    
    def hit_test_handles(self, x: float, y: float) -> Optional[HandleType]:
        """Test if coordinates hit any handle. Returns the handle type or None.
        
        Args:
            x, y: Coordinates in texture/image coordinate system (Y increases downward)
        """
        if not self.bounding_box:
            return None
        
        # Convert input coordinates from texture/image coords to plot coords for comparison
        plot_x = x
        plot_y = self.texture_height - y  # Convert Y from texture coords to plot coords
        
        handle_positions = self.get_handle_positions()
        
        for handle_type, (hx, hy) in handle_positions.items():
            distance = abs(plot_x - hx) + abs(plot_y - hy)  # Manhattan distance
            if distance <= self.handle_threshold:
                return handle_type
        
        return None
    
    def on_mouse_down(self, sender, app_data) -> bool:
        """Handle mouse down events. Returns True if event was handled."""
        if not self.bounding_box:
            return False
        
        mouse_pos = dpg.get_mouse_pos()
        texture_x, texture_y = self.screen_to_texture_coords(mouse_pos[0], mouse_pos[1])
        
        # Check for handle hits first
        hit_handle = self.hit_test_handles(texture_x, texture_y)
        if hit_handle:
            self.is_dragging = True
            self.drag_mode = DragMode.RESIZE
            self.drag_handle = hit_handle
            self.drag_start_mouse = (texture_x, texture_y)
            self.drag_start_box = self.bounding_box.copy()
            
            if self.on_start_drag_callback:
                self.on_start_drag_callback(self.bounding_box.copy())
            return True
        
        # Check if clicking inside the box for move
        if self.bounding_box.contains_point(texture_x, texture_y):
            self.is_dragging = True
            self.drag_mode = DragMode.MOVE
            self.drag_offset = (texture_x - self.bounding_box.x, texture_y - self.bounding_box.y)
            self.drag_start_box = self.bounding_box.copy()
            
            if self.on_start_drag_callback:
                self.on_start_drag_callback(self.bounding_box.copy())
            return True
        
        return False
    
    def on_mouse_drag(self, sender, app_data) -> bool:
        """Handle mouse drag events. Returns True if event was handled."""
        if not self.is_dragging or not self.bounding_box or not self.drag_start_box:
            return False
        
        mouse_pos = dpg.get_mouse_pos()
        texture_x, texture_y = self.screen_to_texture_coords(mouse_pos[0], mouse_pos[1])
        
        if self.drag_mode == DragMode.MOVE:
            # Move the entire box
            new_x = texture_x - self.drag_offset[0]
            new_y = texture_y - self.drag_offset[1]
            
            self.bounding_box.x = new_x
            self.bounding_box.y = new_y
            
            # Clamp to bounds if set
            if self.bounds:
                self.bounding_box.clamp_to_bounds(self.bounds)
        
        elif self.drag_mode == DragMode.RESIZE:
            # Resize based on the handle being dragged
            dx = texture_x - self.drag_start_mouse[0]
            dy = texture_y - self.drag_start_mouse[1]
            
            self._resize_box(self.drag_handle, dx, dy)
        
        # Call change callback
        if self.on_change_callback:
            self.on_change_callback(self.bounding_box.copy())
        
        return True
    
    def on_mouse_release(self, sender, app_data) -> bool:
        """Handle mouse release events. Returns True if event was handled."""
        if not self.is_dragging:
            return False
        
        was_dragging = self.is_dragging
        self.is_dragging = False
        self.drag_mode = DragMode.NONE
        self.drag_handle = None
        self.drag_start_box = None
        
        if was_dragging and self.on_end_drag_callback:
            self.on_end_drag_callback(self.bounding_box.copy())
        
        return was_dragging
    
    def _resize_box(self, handle: HandleType, dx: float, dy: float) -> None:
        """Resize the bounding box based on handle and deltas."""
        if not self.drag_start_box:
            return
        
        # Start with the original box
        start_box = self.drag_start_box
        new_box = start_box.copy()
        
        # Apply resize based on handle
        if handle == HandleType.TOP_LEFT:
            new_box.x += dx
            new_box.y += dy
            new_box.width -= dx
            new_box.height -= dy
        elif handle == HandleType.TOP_RIGHT:
            new_box.y += dy
            new_box.width += dx
            new_box.height -= dy
        elif handle == HandleType.BOTTOM_LEFT:
            new_box.x += dx
            new_box.width -= dx
            new_box.height += dy
        elif handle == HandleType.BOTTOM_RIGHT:
            new_box.width += dx
            new_box.height += dy
        elif handle == HandleType.TOP:
            new_box.y += dy
            new_box.height -= dy
        elif handle == HandleType.BOTTOM:
            new_box.height += dy
        elif handle == HandleType.LEFT:
            new_box.x += dx
            new_box.width -= dx
        elif handle == HandleType.RIGHT:
            new_box.width += dx
        
        # Ensure minimum size
        if new_box.width < self.min_size:
            if handle in [HandleType.TOP_LEFT, HandleType.BOTTOM_LEFT, HandleType.LEFT]:
                new_box.x = start_box.x + start_box.width - self.min_size
            new_box.width = self.min_size
        
        if new_box.height < self.min_size:
            if handle in [HandleType.TOP_LEFT, HandleType.TOP_RIGHT, HandleType.TOP]:
                new_box.y = start_box.y + start_box.height - self.min_size
            new_box.height = self.min_size
        
        # Apply bounds if set
        if self.bounds:
            new_box.clamp_to_bounds(self.bounds)
        
        # Update the bounding box
        self.bounding_box = new_box
    
    def render_on_texture(self, texture: np.ndarray) -> np.ndarray:
        """
        Render the bounding box and handles on the given texture.
        Returns a copy of the texture with the bounding box drawn.
        """
        if not self.bounding_box:
            return texture
        
        result = texture.copy()
        box = self.bounding_box
        
        # Draw the main rectangle
        x1, y1 = int(box.x), int(box.y)
        x2, y2 = int(box.x + box.width), int(box.y + box.height)
        
        # Clamp coordinates to texture bounds
        x1 = max(0, min(x1, self.texture_width - 1))
        y1 = max(0, min(y1, self.texture_height - 1))
        x2 = max(0, min(x2, self.texture_width - 1))
        y2 = max(0, min(y2, self.texture_height - 1))
        
        # Draw rectangle border
        cv2.rectangle(result, (x1, y1), (x2, y2), self.box_color, self.box_thickness)
        
        # Draw corner handles using image coordinates directly (not plot coordinates)
        # In image coordinates: (0,0) is top-left, Y increases downward
        corner_positions = {
            HandleType.TOP_LEFT: (box.x, box.y),
            HandleType.TOP_RIGHT: (box.x + box.width, box.y),
            HandleType.BOTTOM_LEFT: (box.x, box.y + box.height),
            HandleType.BOTTOM_RIGHT: (box.x + box.width, box.y + box.height)
        }
        
        for handle_type, (hx, hy) in corner_positions.items():
            hx_int = int(max(0, min(hx, self.texture_width - 1)))
            hy_int = int(max(0, min(hy, self.texture_height - 1)))
            cv2.circle(result, (hx_int, hy_int), int(self.handle_size), 
                      self.handle_color, -1)
        
        return result
    
    def set_visual_style(self, 
                        box_color: Tuple[int, int, int, int] = None,
                        handle_color: Tuple[int, int, int, int] = None,
                        box_thickness: int = None,
                        handle_size: float = None) -> None:
        """Set the visual style of the bounding box."""
        if box_color:
            self.box_color = box_color
        if handle_color:
            self.handle_color = handle_color
        if box_thickness is not None:
            self.box_thickness = box_thickness
        if handle_size is not None:
            self.handle_size = handle_size
    
    def reset(self) -> None:
        """Reset the bounding box renderer to initial state."""
        self.bounding_box = None
        self.bounds = None
        self.is_dragging = False
        self.drag_mode = DragMode.NONE
        self.drag_handle = None
        self.drag_start_box = None


class BoundingBoxManager:
    """
    Manager class for handling multiple bounding box renderers.
    Useful for applications that need multiple types of selections.
    """
    
    def __init__(self):
        self.renderers: Dict[str, BoundingBoxRenderer] = {}
        self.active_renderer: Optional[str] = None
    
    def add_renderer(self, name: str, renderer: BoundingBoxRenderer) -> None:
        """Add a named bounding box renderer."""
        self.renderers[name] = renderer
    
    def set_active(self, name: str) -> None:
        """Set the active bounding box renderer."""
        if name in self.renderers:
            self.active_renderer = name
    
    def get_active_renderer(self) -> Optional[BoundingBoxRenderer]:
        """Get the currently active renderer."""
        if self.active_renderer and self.active_renderer in self.renderers:
            return self.renderers[self.active_renderer]
        return None
    
    def on_mouse_down(self, sender, app_data) -> bool:
        """Forward mouse down to active renderer."""
        renderer = self.get_active_renderer()
        if renderer:
            return renderer.on_mouse_down(sender, app_data)
        return False
    
    def on_mouse_drag(self, sender, app_data) -> bool:
        """Forward mouse drag to active renderer."""
        renderer = self.get_active_renderer()
        if renderer:
            return renderer.on_mouse_drag(sender, app_data)
        return False
    
    def on_mouse_release(self, sender, app_data) -> bool:
        """Forward mouse release to active renderer."""
        renderer = self.get_active_renderer()
        if renderer:
            return renderer.on_mouse_release(sender, app_data)
        return False