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
                 handle_size: float = 30,
                 handle_threshold: float = 60):
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
        self.box_color = (64, 64, 64, 255)  # Green
        self.handle_color = (13, 115, 184, 255)  # Red
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
            print(f"COORD CONVERSION: Plot conversion failed: {e}")
        
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
    
    def hit_test_handles(self, x: float, y: float) -> Optional[HandleType]:
        """Test if coordinates hit any handle. Returns the handle type or None.
        
        Args:
            x, y: Coordinates in texture/image coordinate system (Y increases downward)
        """
        if not self.bounding_box:
            return None
        
        box = self.bounding_box
        
        # Test handles in texture/image coordinate system directly
        handle_positions = {
            HandleType.TOP_LEFT: (box.x, box.y),
            HandleType.TOP_RIGHT: (box.x + box.width, box.y),
            HandleType.BOTTOM_LEFT: (box.x, box.y + box.height),
            HandleType.BOTTOM_RIGHT: (box.x + box.width, box.y + box.height),
            HandleType.TOP: (box.x + box.width / 2, box.y),
            HandleType.BOTTOM: (box.x + box.width / 2, box.y + box.height),
            HandleType.LEFT: (box.x, box.y + box.height / 2),
            HandleType.RIGHT: (box.x + box.width, box.y + box.height / 2)
        }
        
        closest_handle = None
        closest_distance = float('inf')
        
        for handle_type, (hx, hy) in handle_positions.items():
            # Use Euclidean distance for better circular hit detection
            distance = ((x - hx) ** 2 + (y - hy) ** 2) ** 0.5            
            if distance <= self.handle_threshold:
                if distance < closest_distance:
                    closest_distance = distance
                    closest_handle = handle_type
        
        return closest_handle
    
    def _resize_box(self, handle: HandleType, current_x: float, current_y: float) -> None:
        """Resize the bounding box based on handle and current mouse position."""
        if not self.drag_start_box:
            return
        
        # Start with the original box
        start_box = self.drag_start_box
        new_box = start_box.copy()
        
        # Calculate new boundaries based on current mouse position and fixed opposite point
        if handle == HandleType.TOP_LEFT:
            # Fixed point: bottom right
            fixed_x = start_box.x + start_box.width
            fixed_y = start_box.y + start_box.height
            new_box.x = min(current_x, fixed_x)
            new_box.y = min(current_y, fixed_y)
            new_box.width = abs(fixed_x - current_x)
            new_box.height = abs(fixed_y - current_y)
            
        elif handle == HandleType.TOP_RIGHT:
            # Fixed point: bottom left
            fixed_x = start_box.x
            fixed_y = start_box.y + start_box.height
            new_box.x = min(current_x, fixed_x)
            new_box.y = min(current_y, fixed_y)
            new_box.width = abs(current_x - fixed_x)
            new_box.height = abs(fixed_y - current_y)
            
        elif handle == HandleType.BOTTOM_LEFT:
            # Fixed point: top right
            fixed_x = start_box.x + start_box.width
            fixed_y = start_box.y
            new_box.x = min(current_x, fixed_x)
            new_box.y = min(current_y, fixed_y)
            new_box.width = abs(fixed_x - current_x)
            new_box.height = abs(current_y - fixed_y)
            
        elif handle == HandleType.BOTTOM_RIGHT:
            # Fixed point: top left
            fixed_x = start_box.x
            fixed_y = start_box.y
            new_box.x = fixed_x
            new_box.y = fixed_y
            new_box.width = abs(current_x - fixed_x)
            new_box.height = abs(current_y - fixed_y)
            
        elif handle == HandleType.TOP:
            # Fixed edge: bottom
            fixed_y = start_box.y + start_box.height
            new_box.y = min(current_y, fixed_y)
            new_box.height = abs(fixed_y - current_y)
            
        elif handle == HandleType.BOTTOM:
            # Fixed edge: top
            fixed_y = start_box.y
            new_box.y = fixed_y
            new_box.height = abs(current_y - fixed_y)
            
        elif handle == HandleType.LEFT:
            # Fixed edge: right
            fixed_x = start_box.x + start_box.width
            new_box.x = min(current_x, fixed_x)
            new_box.width = abs(fixed_x - current_x)
            
        elif handle == HandleType.RIGHT:
            # Fixed edge: left
            fixed_x = start_box.x
            new_box.x = fixed_x
            new_box.width = abs(current_x - fixed_x)
        
        # Ensure minimum size
        if new_box.width < self.min_size:
            # Maintain center when correcting width
            center_x = new_box.x + new_box.width / 2
            new_box.width = self.min_size
            new_box.x = center_x - self.min_size / 2
        
        if new_box.height < self.min_size:
            # Maintain center when correcting height
            center_y = new_box.y + new_box.height / 2
            new_box.height = self.min_size
            new_box.y = center_y - self.min_size / 2
        
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
        handle_positions = {
            HandleType.TOP_LEFT: (box.x, box.y),
            HandleType.TOP_RIGHT: (box.x + box.width, box.y),
            HandleType.BOTTOM_LEFT: (box.x, box.y + box.height),
            HandleType.BOTTOM_RIGHT: (box.x + box.width, box.y + box.height),
            HandleType.TOP: (box.x + box.width / 2, box.y),
            HandleType.BOTTOM: (box.x + box.width / 2, box.y + box.height),
            HandleType.LEFT: (box.x, box.y + box.height / 2),
            HandleType.RIGHT: (box.x + box.width, box.y + box.height / 2)
        }
        
        for handle_type, (hx, hy) in handle_positions.items():
            hx_int = int(max(0, min(hx, self.texture_width - 1)))
            hy_int = int(max(0, min(hy, self.texture_height - 1)))
            # Draw handle with white border for better visibility
            cv2.circle(result, (hx_int, hy_int), 10, 
                      self.handle_color, -1)
        
        return result
    
    def reset(self) -> None:
        """Reset the bounding box renderer to initial state."""
        self.bounding_box = None
        self.bounds = None
        self.is_dragging = False
        self.drag_mode = DragMode.NONE
        self.drag_handle = None
        self.drag_start_box = None