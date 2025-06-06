import cv2
import numpy as np
import dearpygui.dearpygui as dpg
import time
from .bounding_box_renderer import BoundingBoxRenderer, BoundingBox

class CropRotateUI:
    def __init__(self, image, image_processor):
        self.original_image = image
        self.image_processor = image_processor
        self.orig_h, self.orig_w = self.original_image.shape[:2]
        diagonal = np.sqrt(self.orig_w**2 + self.orig_h**2)
        self.default_scale = min(self.orig_w, self.orig_h) / diagonal
        self.prev_zoom = self.default_scale
        self.prev_angle = 0
        self.rotated_image = None
        self.offset_x = 0
        self.offset_y = 0

        self.texture_w = int(np.ceil(diagonal))
        self.texture_h = self.texture_w
        self.texture_tag = "crop_rotate_texture"
        self.rotation_slider = "rotation_slider"

        # Initialize the slider if it doesn't exist
        if not dpg.does_item_exist(self.rotation_slider):
            dpg.add_slider_float(tag=self.rotation_slider, default_value=0, min_value=0, max_value=360, show=False)

        # Initialize the bounding box renderer
        self.bbox_renderer = BoundingBoxRenderer(
            texture_width=self.texture_w,
            texture_height=self.texture_h,
            panel_id="Central Panel",
            min_size=20,
            handle_size=20,
            handle_threshold=50
        )
        
        # Set up bounding box callbacks
        self.bbox_renderer.set_callbacks(
            on_change=self._on_bbox_change,
            on_start_drag=self._on_bbox_start_drag,
            on_end_drag=self._on_bbox_end_drag
        )
        
        # Legacy compatibility attributes
        self.user_rect = None
        self.drag_active = False
        self.drag_mode = None
        self.rotated_texture = None
        self.max_rect = None
        self.max_area = None
        self.last_update_time = 0
        self.update_interval = 1 / 60
    
    def _on_bbox_change(self, bbox: BoundingBox) -> None:
        """Called when the bounding box changes during drag."""
        # Only process changes if we're actually dragging
        if not self.drag_active:
            return
            
        # Update legacy user_rect for compatibility
        self.user_rect = bbox.to_dict()
        self.update_rectangle_overlay()
    
    def _on_bbox_start_drag(self, bbox: BoundingBox) -> None:
        """Called when bounding box drag starts."""
        self.drag_active = True
    
    def _on_bbox_end_drag(self, bbox: BoundingBox) -> None:
        """Called when bounding box drag ends."""
        self.drag_active = False
        self.user_rect = bbox.to_dict()
        # Force immediate update when drag ends (bypass throttling)
        self.update_rectangle_overlay(force_update=True)
    
    def _update_bounding_box_from_max_rect(self) -> None:
        """Update the bounding box renderer with the calculated max rect."""
        if self.max_rect:
            # Convert to BoundingBox and set bounds
            max_bbox = BoundingBox.from_dict(self.max_rect)
            self.bbox_renderer.set_bounds(max_bbox)
            
            # Set current bounding box if not set or if angle changed
            if not self.bbox_renderer.bounding_box or (not self.drag_active and hasattr(self, 'prev_angle')):
                self.bbox_renderer.set_bounding_box(max_bbox)
                self.user_rect = max_bbox.to_dict()

    def update_image(self, sender, app_data, user_data):
        panel_w, panel_h = dpg.get_item_rect_size("Central Panel")
        if panel_w <= 0 or panel_h <= 0:
            panel_w, panel_h = dpg.get_item_width("Central Panel"), dpg.get_item_height("Central Panel")

        # Obtener ángulo y estado de crop_mode
        angle = dpg.get_value(self.rotation_slider)
        crop_mode = dpg.get_value("crop_mode") if dpg.does_item_exist("crop_mode") else False

        if crop_mode:
            # Modo edición: mostrar imagen rotada completa con rectángulo
            rotated_image = self.image_processor.rotate_image(self.original_image, angle)
            rot_h, rot_w = rotated_image.shape[:2]
            
            # Store these offset values as they'll be needed later for proper cropping
            self.offset_x = (self.texture_w - rot_w) // 2
            self.offset_y = (self.texture_h - rot_h) // 2
            
            padded_image = np.full((self.texture_h, self.texture_w, 4), [100, 100, 100, 0], dtype=np.uint8)
            
            if rotated_image.shape[2] == 3:
                rotated_image = cv2.cvtColor(rotated_image, cv2.COLOR_RGB2RGBA)
            
            if self.offset_y >= 0 and self.offset_x >= 0 and self.offset_y + rot_h <= self.texture_h and self.offset_x + rot_w <= self.texture_w:
                padded_image[self.offset_y:self.offset_y + rot_h, self.offset_x:self.offset_x + rot_w] = rotated_image
            else:
                h_slice = min(rot_h, self.texture_h)
                w_slice = min(rot_w, self.texture_w)
                padded_image[:h_slice, :w_slice] = rotated_image[:h_slice, :w_slice]
                self.offset_x = 0
                self.offset_y = 0

            self.rotated_image = rotated_image.copy()
            self.rotated_texture = padded_image.copy()
            self.rot_h, self.rot_w = rot_h, rot_w  # Store rotated dimensions

            # Calcular rectángulo máximo
            angle_rad = np.deg2rad(angle)
            inscribed_w, inscribed_h = self.image_processor.get_largest_inscribed_rect_dims(angle_rad)
            self.max_rect = {
                "x": (self.texture_w - inscribed_w) // 2,
                "y": (self.texture_h - inscribed_h) // 2,
                "w": inscribed_w,
                "h": inscribed_h
            }
            self.max_area = inscribed_w * inscribed_h

            # Update bounding box renderer
            self._update_bounding_box_from_max_rect()
            self.prev_angle = angle

            self.update_rectangle_overlay()
        else:
            # Modo resultado: mostrar la imagen con los ajustes básicos aplicados
            if self.user_rect and hasattr(self, 'rotated_image') and self.rotated_image is not None:
                # Adjust rectangle coordinates relative to the rotated image, not the texture
                rotated_image = self.image_processor.rotate_image(self.original_image, angle)
                self.rotated_image = rotated_image.copy()
                rx = int(self.user_rect["x"] - self.offset_x)
                ry = int(self.user_rect["y"] - self.offset_y)
                rw = int(self.user_rect["w"])
                rh = int(self.user_rect["h"])
                
                # Ensure the rectangle is within the bounds of the rotated image
                rx = max(0, rx)
                ry = max(0, ry)
                rx_end = min(rx + rw, self.rot_w)
                ry_end = min(ry + rh, self.rot_h)
                rw = rx_end - rx
                rh = ry_end - ry
                
                # Only crop if we have valid dimensions
                if rw > 0 and rh > 0:
                    # Apply crop directly on the rotated image (without padding)
                    cropped_image = self.rotated_image[ry:ry+rh, rx:rx+rw].copy()
                    
                    if cropped_image.shape[2] == 3:
                        cropped_image_rgba = cv2.cvtColor(cropped_image, cv2.COLOR_RGB2RGBA)
                    else:
                        cropped_image_rgba = cropped_image

                    cv2.imwrite("crop.png", cropped_image_rgba)

                    # Create background and center the cropped image
                    gray_background = np.full((self.texture_h, self.texture_w, 4), 
                                             [100, 100, 100, 0], dtype=np.uint8)
                    
                    offset_x = (self.texture_w - cropped_image.shape[1]) // 2
                    offset_y = (self.texture_h - cropped_image.shape[0]) // 2
                    
                    gray_background[offset_y:offset_y + cropped_image.shape[0], 
                                   offset_x:offset_x + cropped_image.shape[1]] = cropped_image_rgba
                    
                    # Update texture
                    texture_data = gray_background.flatten().astype(np.float32) / 255.0
                    if dpg.does_item_exist(self.texture_tag):
                        dpg.set_value(self.texture_tag, texture_data)
                    else:
                        with dpg.texture_registry():
                            dpg.add_dynamic_texture(self.texture_w, self.texture_h, texture_data, 
                                                  tag=self.texture_tag, format=dpg.mvFormat_Float_rgba)
            else:
                # Just display the original image with processing parameters but no crop
                display_image = self.original_image.copy()  # self.original_image already has all processing applied
                print(f"Updating image in non-crop mode, dimensions: {display_image.shape}")
                
                # Create background and center the image
                gray_background = np.full((self.texture_h, self.texture_w, 4), 
                                         [100, 100, 100, 0], dtype=np.uint8)
                
                # Get dimensions of the processed image
                display_h, display_w = display_image.shape[:2]
                
                # Update these attributes to make sure future operations use the correct dimensions
                self.orig_h, self.orig_w = display_h, display_w
                
                offset_x = (self.texture_w - display_w) // 2
                offset_y = (self.texture_h - display_h) // 2
                
                # Convert to RGBA if needed
                if display_image.shape[2] == 3:
                    display_image_rgba = cv2.cvtColor(display_image, cv2.COLOR_RGB2RGBA)
                else:
                    display_image_rgba = display_image
                
                # Place the image on the background
                gray_background[offset_y:offset_y + display_image.shape[0], 
                               offset_x:offset_x + display_image.shape[1]] = display_image_rgba
                
                # Update texture
                texture_data = gray_background.flatten().astype(np.float32) / 255.0
                if dpg.does_item_exist(self.texture_tag):
                    dpg.set_value(self.texture_tag, texture_data)
                else:
                    with dpg.texture_registry():
                        dpg.add_dynamic_texture(self.texture_w, self.texture_h, texture_data, 
                                              tag=self.texture_tag, format=dpg.mvFormat_Float_rgba)
        
        # Update axis limits to maintain proper aspect ratio
        self.update_axis_limits()
    
    def update_axis_limits(self):
        # Get actual image dimensions within the texture
        orig_w = self.orig_w
        orig_h = self.orig_h
        
        # Calculate where the image is positioned within the texture (centered)
        image_offset_x = (self.texture_w - orig_w) // 2
        image_offset_y = (self.texture_h - orig_h) // 2
        
        # Get current plot dimensions
        panel_w, panel_h = dpg.get_item_rect_size("Central Panel")
        if panel_w <= 0 or panel_h <= 0:
            panel_w, panel_h = dpg.get_item_width("Central Panel"), dpg.get_item_height("Central Panel")
        
        if panel_w <= 0 or panel_h <= 0:
            panel_w, panel_h = 800, 600  # Fallback
        
        # Calculate aspect ratios
        plot_aspect = panel_w / panel_h
        image_aspect = orig_w / orig_h
        
        # Calculate how to fit the actual image within the plot while maintaining aspect ratio
        # Add some padding around the image to see it properly
        padding_factor = 1.05  # 5% padding around the image
        
        if image_aspect > plot_aspect:
            # Image is wider - fit to plot width, scale height
            display_width = orig_w * padding_factor
            display_height = display_width / plot_aspect
            
            # Center view on the actual image
            x_center = image_offset_x + orig_w / 2
            y_center = image_offset_y + orig_h / 2
            x_min = x_center - display_width / 2
            x_max = x_center + display_width / 2
            y_min = y_center - display_height / 2
            y_max = y_center + display_height / 2
        else:
            # Image is taller - fit to plot height, scale width
            display_height = orig_h * padding_factor
            display_width = display_height * plot_aspect
            
            # Center view on the actual image
            x_center = image_offset_x + orig_w / 2
            y_center = image_offset_y + orig_h / 2
            x_min = x_center - display_width / 2
            x_max = x_center + display_width / 2
            y_min = y_center - display_height / 2
            y_max = y_center + display_height / 2
        
        # Don't modify the image series bounds - keep them as the full texture
        # The axis limits will control what portion of the texture is visible
        
        dpg.set_axis_limits("x_axis", x_min, x_max)
        dpg.set_axis_limits("y_axis", y_min, y_max)

    def update_rectangle_overlay(self, force_update=False):
        current_time = time.time()
        if not force_update and current_time - self.last_update_time < self.update_interval:
            return
        self.last_update_time = current_time

        if self.rotated_texture is None:
            return
        
        # Use the bounding box renderer to draw the overlay
        blended = self.bbox_renderer.render_on_texture(self.rotated_texture)
        
        texture_data = blended.flatten().astype(np.float32) / 255.0
        dpg.set_value(self.texture_tag, texture_data)

    def on_mouse_down(self, sender, app_data):
        crop_mode = dpg.get_value("crop_mode") if dpg.does_item_exist("crop_mode") else False
        if crop_mode and self.bbox_renderer.bounding_box:
            return self.bbox_renderer.on_mouse_down(sender, app_data)
        return False

    def on_mouse_drag(self, sender, app_data):
        crop_mode = dpg.get_value("crop_mode") if dpg.does_item_exist("crop_mode") else False
        # Only process drag if crop mode is active and we're actually dragging
        if crop_mode and self.bbox_renderer.bounding_box and self.drag_active:
            return self.bbox_renderer.on_mouse_drag(sender, app_data)
        return False

    def on_mouse_release(self, sender, app_data):
        crop_mode = dpg.get_value("crop_mode") if dpg.does_item_exist("crop_mode") else False
        if crop_mode and self.bbox_renderer.bounding_box:
            result = self.bbox_renderer.on_mouse_release(sender, app_data)
            # Additional safety: ensure drag_active is cleared even if bbox_renderer doesn't call end_drag
            if self.drag_active and not self.bbox_renderer.is_dragging:
                self.drag_active = False
            return result
        return False

    def set_to_max_rect(self, sender, app_data, user_data):
        if self.max_rect:
            max_bbox = BoundingBox.from_dict(self.max_rect)
            self.bbox_renderer.set_bounding_box(max_bbox)
            self.user_rect = max_bbox.to_dict()
            self.update_rectangle_overlay()

    def crop_image(self, sender, app_data, user_data):
        angle = dpg.get_value(self.rotation_slider)
        rx, ry, rw, rh = map(int, (self.user_rect["x"], self.user_rect["y"], 
                                   self.user_rect["w"], self.user_rect["h"]))
        # Adjust coordinates to the original image space
        offset_x = (self.texture_w - self.orig_w) // 2
        offset_y = (self.texture_h - self.orig_h) // 2
        crop_rect = (
            max(0, rx - offset_x),
            max(0, ry - offset_y),
            min(rw, self.orig_w - max(0, rx - offset_x)),
            min(rh, self.orig_h - max(0, ry - offset_y))
        )
        # Apply rotation and crop
        cropped = self.image_processor.crop_rotate_flip(self.original_image, crop_rect, angle)
        # Update the original_image with the cropped result
        self.original_image = cropped.copy()
        # Ensure the aspect ratio is maintained
        aspect_ratio = self.orig_w / self.orig_h
        cropped_h, cropped_w = cropped.shape[:2]
        if cropped_w / cropped_h != aspect_ratio:
            if cropped_w / cropped_h > aspect_ratio:
                new_w = int(cropped_h * aspect_ratio)
                offset = (cropped_w - new_w) // 2
                cropped = cropped[:, offset:offset + new_w]
            else:
                new_h = int(cropped_w / aspect_ratio)
                offset = (cropped_h - new_h) // 2
                cropped = cropped[offset:offset + new_h, :]
        # Show cropped image in a window
        height, width = cropped.shape[:2]
        cropped_flat = cropped.flatten() / 255.0
        if dpg.does_item_exist("cropped_texture"):
            dpg.set_value("cropped_texture", cropped_flat)
        else:
            with dpg.texture_registry():
                dpg.add_dynamic_texture(width, height, cropped_flat, tag="cropped_texture")