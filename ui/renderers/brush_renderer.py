"""
Brush Renderer for Manual Mask Drawing

Handles drawing brush strokes on the image to create masks manually.
Optimized for performance with caching and throttling.
"""
import cv2
import numpy as np
import dearpygui.dearpygui as dpg
from typing import Optional, Tuple, List, Dict
import time


class BrushRenderer:
    """Renderer for drawing brush strokes to create masks."""
    
    def __init__(self, texture_width: int, texture_height: int, panel_id: str):
        self.texture_width = texture_width
        self.texture_height = texture_height
        self.panel_id = panel_id
        
        # Brush state
        self.is_painting = False
        self.brush_size = 20
        self.brush_opacity = 1.0
        self.brush_hardness = 0.8
        self.eraser_mode = False
        
        # Current mask being drawn
        self.current_mask = np.zeros((texture_height, texture_width), dtype=np.uint8)
        
        # Drawing state
        self.last_mouse_pos = None
        self.stroke_points = []
        
        # Performance settings - more aggressive throttling
        self.update_throttle_ms = 8  # ~120 FPS max, reduced from 16ms
        self.last_update_time = 0
        self.display_update_throttle_ms = 25  # Separate throttle for display updates (~40 FPS)
        self.last_display_update = 0
        
        # Brush cursor
        self.cursor_visible = False
        self.cursor_pos = (0, 0)
        
        # Performance optimizations - brush mask caching
        self._brush_cache: Dict[Tuple[int, float], np.ndarray] = {}
        self._max_cache_size = 20  # Cache common brush sizes
        
        # Coordinate caching
        self._last_screen_coords = None
        self._last_texture_coords = None
        
        # Stroke optimization
        self._min_stroke_distance = 2.0  # Minimum distance between stroke points
        
        # Display update batching
        self._pending_display_update = False
        
    def set_brush_parameters(self, size: int, opacity: float, hardness: float, eraser_mode: bool):
        """Update brush parameters."""
        # Clear cache if significant parameter changes
        old_size = self.brush_size
        old_hardness = self.brush_hardness
        
        self.brush_size = max(1, min(100, size))
        self.brush_opacity = max(0.1, min(1.0, opacity))
        self.brush_hardness = max(0.0, min(1.0, hardness))
        self.eraser_mode = eraser_mode
        
        # Clear brush cache if size or hardness changed significantly
        if abs(old_size - self.brush_size) > 2 or abs(old_hardness - self.brush_hardness) > 0.1:
            self._brush_cache.clear()

    def screen_to_texture_coords(self, screen_x: float, screen_y: float) -> Tuple[float, float]:
        """Convert screen coordinates to texture coordinates with caching."""
        # Check if we can reuse cached coordinates
        current_screen = (screen_x, screen_y)
        if (self._last_screen_coords is not None and 
            self._last_texture_coords is not None and
            abs(current_screen[0] - self._last_screen_coords[0]) < 1.0 and
            abs(current_screen[1] - self._last_screen_coords[1]) < 1.0):
            return self._last_texture_coords
            
        try:
            import dearpygui.dearpygui as dpg
            
            # Use the same approach as bounding box renderer
            # First try to get plot mouse position (if we're in a plot)
            if dpg.does_item_exist("image_plot"):
                plot_pos = dpg.get_plot_mouse_pos()
                if plot_pos is not None:
                    # Plot coordinates are in plot coordinate system (Y increases upward)
                    # We need to convert to image coordinate system (Y increases downward)
                    texture_x = plot_pos[0]
                    texture_y = self.texture_height - plot_pos[1]  # Invert Y coordinate
                    
                    # Clamp to texture bounds
                    texture_x = max(0, min(texture_x, self.texture_width - 1))
                    texture_y = max(0, min(texture_y, self.texture_height - 1))
                    
                    # Cache the result
                    self._last_screen_coords = current_screen
                    self._last_texture_coords = (texture_x, texture_y)
                    
                    return texture_x, texture_y
            
            return 0, 0
        except Exception as e:
            print(f"Brush coordinate conversion failed: {e}")
            return 0, 0

    def start_stroke(self, x: float, y: float) -> bool:
        """Start a new brush stroke."""
        if not self._is_valid_position(x, y):
            return False
    
        self.is_painting = True
        self.last_mouse_pos = (x, y)
        self.stroke_points = [(x, y)]
        
        # Draw initial point
        self._draw_brush_point(x, y)
        return True

    def continue_stroke(self, x: float, y: float) -> bool:
        """Continue the current brush stroke with optimized throttling."""
        if not self.is_painting or not self._is_valid_position(x, y):
            return False
        
        # More aggressive throttling for stroke continuation
        current_time = time.time() * 1000
        if current_time - self.last_update_time < self.update_throttle_ms:
            return False
        
        # Distance-based filtering to reduce redundant points
        if self.last_mouse_pos:
            distance = np.sqrt((x - self.last_mouse_pos[0])**2 + (y - self.last_mouse_pos[1])**2)
            if distance < self._min_stroke_distance:
                return False

        self.last_update_time = current_time
        
        # Draw line from last position to current position with optimized interpolation
        if self.last_mouse_pos:
            self._draw_brush_line_optimized(self.last_mouse_pos[0], self.last_mouse_pos[1], x, y)
        
        self.last_mouse_pos = (x, y)
        self.stroke_points.append((x, y))
        return True

    def end_stroke(self) -> bool:
        """End the current brush stroke."""
        if not self.is_painting:
            return False
        
        self.is_painting = False
        self.last_mouse_pos = None
        self.stroke_points = []
        return True

    def _is_valid_position(self, x: float, y: float) -> bool:
        """Check if the position is within valid bounds."""
        return 0 <= x < self.texture_width and 0 <= y < self.texture_height

    def _draw_brush_point(self, x: float, y: float):
        """Draw a single brush point with caching."""
        center_x, center_y = int(x), int(y)
        radius = self.brush_size // 2
        
        # Get cached brush mask
        brush_mask = self._get_cached_brush_mask(radius)
        
        # Apply brush to mask
        self._apply_brush_to_mask(center_x, center_y, brush_mask)

    def _draw_brush_line_optimized(self, x1: float, y1: float, x2: float, y2: float):
        """Draw a line between two points with optimized step calculation."""
        # Calculate distance and number of steps - more efficient for performance
        distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        
        # Adaptive step size based on brush size and distance
        step_size = max(self.brush_size * 0.3, 1.0)  # Increased from 0.25 to 0.3
        steps = max(1, int(distance / step_size))
        
        # Limit maximum steps for performance
        steps = min(steps, 50)  # Prevent excessive interpolation
        
        # Interpolate points along the line
        for i in range(steps + 1):
            t = i / max(1, steps)
            x = x1 + t * (x2 - x1)
            y = y1 + t * (y2 - y1)
            self._draw_brush_point(x, y)

    def _get_cached_brush_mask(self, radius: int) -> np.ndarray:
        """Get a cached brush mask or create and cache a new one."""
        cache_key = (radius, self.brush_hardness)
        
        if cache_key in self._brush_cache:
            return self._brush_cache[cache_key]
        
        # Create new brush mask
        brush_mask = self._create_brush_mask(radius)
        
        # Cache with size limit
        if len(self._brush_cache) >= self._max_cache_size:
            # Remove oldest entry (simple FIFO)
            oldest_key = next(iter(self._brush_cache))
            del self._brush_cache[oldest_key]
        
        self._brush_cache[cache_key] = brush_mask
        return brush_mask

    def _create_brush_mask(self, radius: int) -> np.ndarray:
        """Create a brush mask with the specified radius and hardness - optimized."""
        size = radius * 2 + 1
        brush_mask = np.zeros((size, size), dtype=np.float32)
        
        center = radius
        
        # Pre-calculate hardness parameters
        hard_radius = radius * self.brush_hardness
        falloff_range = radius * (1.0 - self.brush_hardness) if self.brush_hardness < 1.0 else 1.0
        
        # Vectorized approach for better performance
        y_indices, x_indices = np.mgrid[0:size, 0:size]
        distances = np.sqrt((x_indices - center)**2 + (y_indices - center)**2)
        
        # Create mask based on distance
        mask_values = np.zeros_like(distances)
        
        # Core area (full opacity)
        core_mask = distances <= hard_radius
        mask_values[core_mask] = 1.0
        
        # Falloff area
        if self.brush_hardness < 1.0:
            falloff_mask = (distances > hard_radius) & (distances <= radius)
            falloff_distances = distances[falloff_mask] - hard_radius
            alpha_values = 1.0 - (falloff_distances / falloff_range)
            mask_values[falloff_mask] = np.maximum(0.0, alpha_values)
        
        # Apply opacity
        brush_mask = mask_values * self.brush_opacity
        return brush_mask
    
    def _apply_brush_to_mask(self, center_x: int, center_y: int, brush_mask: np.ndarray):
        """Apply the brush mask to the current mask - optimized version."""
        radius = brush_mask.shape[0] // 2
        
        # Calculate bounds with early bounds checking
        y_start = max(0, center_y - radius)
        y_end = min(self.texture_height, center_y + radius + 1)
        x_start = max(0, center_x - radius)
        x_end = min(self.texture_width, center_x + radius + 1)
        
        # Early exit if no overlap
        if y_start >= y_end or x_start >= x_end:
            return
        
        # Calculate brush mask bounds
        brush_y_start = max(0, radius - center_y)
        brush_y_end = brush_y_start + (y_end - y_start)
        brush_x_start = max(0, radius - center_x)
        brush_x_end = brush_x_start + (x_end - x_start)
        
        # Get the relevant sections - use views instead of copies when possible
        mask_section = self.current_mask[y_start:y_end, x_start:x_end]
        brush_section = brush_mask[brush_y_start:brush_y_end, brush_x_start:brush_x_end]
        
        if mask_section.shape != brush_section.shape:
            return  # Skip if shapes don't match (edge case)
        
        # Direct computation without intermediate float conversion
        if self.eraser_mode:
            # Erase mode: subtract from mask
            # Convert to float only for computation
            mask_float = mask_section.astype(np.float32) / 255.0
            new_mask = mask_float * (1.0 - brush_section)
            self.current_mask[y_start:y_end, x_start:x_end] = (new_mask * 255).astype(np.uint8)
        else:
            # Paint mode: optimized maximum operation
            brush_uint8 = (brush_section * 255).astype(np.uint8)
            self.current_mask[y_start:y_end, x_start:x_end] = np.maximum(mask_section, brush_uint8)
    
    def clear_mask(self):
        """Clear the current mask."""
        self.current_mask.fill(0)
    
    def get_mask(self) -> np.ndarray:
        """Get the current mask."""
        return self.current_mask.copy()
    
    def set_mask(self, mask: np.ndarray):
        """Set the current mask."""
        if mask.shape[:2] == (self.texture_height, self.texture_width):
            if len(mask.shape) == 3:
                # Convert RGB/RGBA to grayscale
                self.current_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            else:
                self.current_mask = mask.copy()
        else:
            # Resize mask to fit texture
            self.current_mask = cv2.resize(mask, (self.texture_width, self.texture_height), 
                                         interpolation=cv2.INTER_NEAREST)
    
    def update_cursor(self, x: float, y: float, visible: bool = True):
        """Update brush cursor position."""
        self.cursor_pos = (x, y)
        self.cursor_visible = visible
    
    def render_cursor_overlay(self, base_texture: np.ndarray) -> np.ndarray:
        """Render brush cursor overlay on the base texture."""
        if not self.cursor_visible:
            return base_texture
        
        overlay = base_texture.copy()
        center_x, center_y = int(self.cursor_pos[0]), int(self.cursor_pos[1])
        radius = self.brush_size // 2
        
        # Draw cursor circle
        cv2.circle(overlay, (center_x, center_y), radius, (255, 255, 255, 128), 2)
        if radius > 5:
            cv2.circle(overlay, (center_x, center_y), radius - 2, (0, 0, 0, 64), 1)
        
        return overlay
    
    def render_mask_overlay(self, base_texture: np.ndarray, mask_color: Tuple[int, int, int] = (255, 255, 0)) -> np.ndarray:
        """Render the current mask as an overlay on the base texture."""
        if np.max(self.current_mask) == 0:
            return base_texture  # No mask to render
        
        overlay = base_texture.copy()
        
        # Create colored mask
        mask_normalized = self.current_mask.astype(np.float32) / 255.0
        
        # Apply mask color with transparency
        for c in range(3):  # RGB channels
            overlay[:, :, c] = (overlay[:, :, c] * (1.0 - mask_normalized * 0.5) + 
                              mask_color[c] * mask_normalized * 0.5).astype(np.uint8)
        
        return overlay
    
    def get_mask_for_image_coords(self, image_width: int, image_height: int, 
                                 offset_x: int = 0, offset_y: int = 0) -> np.ndarray:
        """Get the mask scaled and positioned for the actual image coordinates."""
        # The current_mask is in texture coordinates, we need to extract the image portion
        # and scale it to match the actual image dimensions
        
        # Extract the relevant portion from texture coordinates
        y_start = max(0, offset_y)
        y_end = min(self.texture_height, offset_y + image_height)
        x_start = max(0, offset_x)
        x_end = min(self.texture_width, offset_x + image_width)
        
        # Get the mask section that corresponds to the image area
        mask_section = self.current_mask[y_start:y_end, x_start:x_end]
        
        # If the extracted section doesn't match the target dimensions, resize it
        if mask_section.shape != (image_height, image_width):
            mask_section = cv2.resize(mask_section, (image_width, image_height), 
                                    interpolation=cv2.INTER_NEAREST)
        
        # Convert to binary mask (0 or 255 values)
        return (mask_section > 127).astype(np.uint8) * 255
    
    def get_mask_for_image_coords_with_transforms(self, image_width: int, image_height: int, 
                                                 offset_x: int = 0, offset_y: int = 0,
                                                 rotation_angle: float = 0, 
                                                 flip_horizontal: bool = False,
                                                 flip_vertical: bool = False,
                                                 target_width: int = None,
                                                 target_height: int = None) -> np.ndarray:
        """Get the mask scaled and positioned for the actual image coordinates with rotation/flip transforms.
        
        The coordinate transformation workflow:
        1. Original image -> rotate -> flip -> display (where brush was drawn)
        2. Brush mask (in display coordinates) -> undo flip -> undo rotate -> original coordinates
        """
        import cv2
        import numpy as np
        
        try:
            # If target dimensions not specified, use the image dimensions
            if target_width is None:
                target_width = image_width
            if target_height is None:
                target_height = image_height
                
            print(f"DEBUG: BrushRenderer transforming mask - rotation={rotation_angle}°, flips=H:{flip_horizontal},V:{flip_vertical}")
            print(f"DEBUG: Display dimensions: {image_width}x{image_height}, Target: {target_width}x{target_height}")
            print(f"DEBUG: Texture offset: ({offset_x}, {offset_y})")
                
            # Step 1: Extract the brush mask from the texture coordinates where the transformed image was displayed
            y_start = max(0, offset_y)
            y_end = min(self.texture_height, offset_y + image_height)
            x_start = max(0, offset_x)
            x_end = min(self.texture_width, offset_x + image_width)
            
            # Get the mask section that corresponds to the displayed (transformed) image area
            mask_section = self.current_mask[y_start:y_end, x_start:x_end]
            print(f"DEBUG: Extracted mask section shape: {mask_section.shape} from texture area [{y_start}:{y_end}, {x_start}:{x_end}]")
            
            # Ensure we have the correct dimensions for the displayed image
            if mask_section.shape != (image_height, image_width):
                mask_section = cv2.resize(mask_section, (image_width, image_height), 
                                        interpolation=cv2.INTER_NEAREST)
                print(f"DEBUG: Resized mask section to: {mask_section.shape}")
            
            # Step 2: Now we have the mask in the coordinate system of the transformed (rotated+flipped) image
            # We need to apply inverse transformations to get back to original image coordinates
            
            result_mask = mask_section.copy()
            
            # First, undo flips (apply in reverse order from how they were applied to the image)
            if flip_vertical:
                print("DEBUG: Undoing vertical flip")
                result_mask = cv2.flip(result_mask, 0)
                
            if flip_horizontal:
                print("DEBUG: Undoing horizontal flip") 
                result_mask = cv2.flip(result_mask, 1)
            
            # Then, undo rotation (if there was rotation)
            if abs(rotation_angle) > 0.01:
                print(f"DEBUG: Undoing rotation by {-rotation_angle}°")
                
                # The challenge: when we rotated the original image, its dimensions changed
                # We need to map from rotated coordinates back to original coordinates
                
                # If the displayed image dimensions are different from target, we're dealing with a rotated image
                if (image_width != target_width or image_height != target_height):
                    # The mask is currently in the coordinate space of the rotated image
                    # We need to "unrotate" it to get back to the original coordinate space
                    
                    # Create a larger canvas to hold the unrotated mask to avoid clipping
                    diagonal = int(np.sqrt(image_width**2 + image_height**2)) + 10
                    canvas = np.zeros((diagonal, diagonal), dtype=np.uint8)
                    
                    # Place the rotated mask in the center of the canvas
                    canvas_center_y = diagonal // 2
                    canvas_center_x = diagonal // 2
                    start_y = canvas_center_y - image_height // 2
                    start_x = canvas_center_x - image_width // 2
                    canvas[start_y:start_y + image_height, start_x:start_x + image_width] = result_mask
                    
                    # Apply inverse rotation around the canvas center
                    center = (canvas_center_x, canvas_center_y)
                    rotation_matrix = cv2.getRotationMatrix2D(center, -rotation_angle, 1.0)
                    
                    unrotated_canvas = cv2.warpAffine(
                        canvas, 
                        rotation_matrix, 
                        (diagonal, diagonal),
                        flags=cv2.INTER_NEAREST,
                        borderMode=cv2.BORDER_CONSTANT,
                        borderValue=0
                    )
                    
                    # Extract the original image area from the unrotated canvas
                    extract_start_y = canvas_center_y - target_height // 2
                    extract_start_x = canvas_center_x - target_width // 2
                    
                    result_mask = unrotated_canvas[
                        extract_start_y:extract_start_y + target_height,
                        extract_start_x:extract_start_x + target_width
                    ]
                    
                    print(f"DEBUG: After inverse rotation and extraction: {result_mask.shape}")
                    
                else:
                    # No dimension change, simple rotation around center
                    center = (result_mask.shape[1] / 2, result_mask.shape[0] / 2)
                    rotation_matrix = cv2.getRotationMatrix2D(center, -rotation_angle, 1.0)
                    
                    result_mask = cv2.warpAffine(
                        result_mask, 
                        rotation_matrix, 
                        (result_mask.shape[1], result_mask.shape[0]),
                        flags=cv2.INTER_NEAREST,
                        borderMode=cv2.BORDER_CONSTANT,
                        borderValue=0
                    )
                    print(f"DEBUG: After simple inverse rotation: {result_mask.shape}")
            
            # Step 3: Ensure final dimensions match target
            if result_mask.shape != (target_height, target_width):
                print(f"DEBUG: Final resize from {result_mask.shape} to ({target_height}, {target_width})")
                result_mask = cv2.resize(result_mask, (target_width, target_height), 
                                       interpolation=cv2.INTER_NEAREST)
            
            # Convert to binary mask (0 or 255 values)
            final_result = (result_mask > 127).astype(np.uint8) * 255
            print(f"DEBUG: Final mask has {np.sum(final_result > 0)} non-zero pixels out of {final_result.size}")
            
            return final_result
            
        except Exception as e:
            print(f"ERROR: Failed to transform brush mask: {e}")
            import traceback
            traceback.print_exc()
            # Fallback to original method
            return self.get_mask_for_image_coords(target_width if target_width else image_width, 
                                                target_height if target_height else image_height, 
                                                offset_x, offset_y)
