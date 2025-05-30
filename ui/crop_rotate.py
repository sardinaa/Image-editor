import cv2
import numpy as np
import dearpygui.dearpygui as dpg
import time

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

        # Establecer un valor inicial para el slider si no existe
        if not dpg.does_item_exist(self.rotation_slider):
            dpg.add_slider_float(tag=self.rotation_slider, default_value=0, min_value=0, max_value=360, show=False)

        self.user_rect = None
        self.drag_active = False
        self.drag_mode = None
        self.rotated_texture = None
        self.max_rect = None
        self.max_area = None
        self.last_update_time = 0
        self.update_interval = 1 / 60

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

            # Inicializar o resetear user_rect
            if not self.user_rect or (not self.drag_active and self.prev_angle != angle):
                self.user_rect = self.max_rect.copy()
            self.prev_angle = angle

            self.update_rectangle_overlay()
        else:
            # Modo resultado: mostrar la imagen con los ajustes básicos aplicados
            if self.user_rect and hasattr(self, 'rotated_image') and self.rotated_image is not None:
                # Adjust rectangle coordinates relative to the rotated image, not the texture
                rotated_image = self.image_processor.rotate_image(self.original_image, angle)
                self.rotated_image = rotated_image.copy()
                rx = self.user_rect["x"] - self.offset_x
                ry = self.user_rect["y"] - self.offset_y
                rw = self.user_rect["w"]
                rh = self.user_rect["h"]
                
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

    def update_rectangle_overlay(self):
        current_time = time.time()
        if current_time - self.last_update_time < self.update_interval:
            return
        self.last_update_time = current_time

        if self.rotated_texture is None:
            return
        blended = self.rotated_texture.copy()

        rx, ry, rw, rh = map(int, (self.user_rect["x"], self.user_rect["y"],
                                   self.user_rect["w"], self.user_rect["h"]))
        rx_clamped = max(rx, 0)
        ry_clamped = max(ry, 0)
        rx_end = min(rx + rw, self.texture_w)
        ry_end = min(ry + rh, self.texture_h)

        cv2.rectangle(blended, (rx_clamped, ry_clamped), (rx_end, ry_end), (0, 255, 0, 200), 2)
        for corner in [(rx_clamped, ry_clamped), (rx_end, ry_clamped),
                       (rx_clamped, ry_end), (rx_end, ry_end)]:
            cv2.circle(blended, corner, radius=5, color=(255, 0, 0, 255), thickness=-1)

        texture_data = blended.flatten().astype(np.float32) / 255.0
        dpg.set_value(self.texture_tag, texture_data)

    def on_mouse_down(self, sender, app_data):
        crop_mode = dpg.get_value("crop_mode") if dpg.does_item_exist("crop_mode") else False
        if crop_mode:
            if not self.user_rect:
                return
            self.drag_active = False
            mouse_pos = dpg.get_mouse_pos()
            panel_pos = dpg.get_item_pos("Central Panel")
            panel_w, panel_h = dpg.get_item_rect_size("Central Panel")
            mouse_x = (mouse_pos[0] - panel_pos[0]) / panel_w * self.texture_w
            mouse_y = (mouse_pos[1] - panel_pos[1]) / panel_h * self.texture_h

            inside_rect = self.point_in_rect(mouse_x, mouse_y, self.user_rect)
            if inside_rect:
                self.drag_active = True
                self.drag_mode = "move"
                self.drag_offset = (mouse_x - self.user_rect["x"], mouse_y - self.user_rect["y"])
                self.drag_start_rect = self.user_rect.copy()
            else:
                handle = self.hit_test_handles(mouse_x, mouse_y, self.user_rect)
                if handle:
                    self.drag_active = True
                    self.drag_mode = "resize"
                    self.drag_handle = handle
                    self.drag_start_mouse = (mouse_x, mouse_y)
                    self.drag_start_rect = self.user_rect.copy()

    def on_mouse_drag(self, sender, app_data):
        crop_mode = dpg.get_value("crop_mode") if dpg.does_item_exist("crop_mode") else False
        if crop_mode:
            if not self.drag_active:
                return
            mouse_pos = dpg.get_mouse_pos()
            panel_pos = dpg.get_item_pos("Central Panel")
            panel_w, panel_h = dpg.get_item_rect_size("Central Panel")
            mouse_x = (mouse_pos[0] - panel_pos[0]) / panel_w * self.texture_w
            mouse_y = (mouse_pos[1] - panel_pos[1]) / panel_h * self.texture_h

            if self.drag_mode == "move":
                delta_x = mouse_x - self.drag_start_rect["x"] - self.drag_offset[0]
                delta_y = mouse_y - self.drag_start_rect["y"] - self.drag_offset[1]
                new_x = self.drag_start_rect["x"] + delta_x
                new_y = self.drag_start_rect["y"] + delta_y
                new_x = max(self.max_rect["x"], min(new_x, self.max_rect["x"] + self.max_rect["w"] - self.user_rect["w"]))
                new_y = max(self.max_rect["y"], min(new_y, self.max_rect["y"] + self.max_rect["h"] - self.user_rect["h"]))
                self.user_rect["x"], self.user_rect["y"] = new_x, new_y
            elif self.drag_mode == "resize":
                dx = mouse_x - self.drag_start_mouse[0]
                dy = mouse_y - self.drag_start_mouse[1]
                new_rect = self.drag_start_rect.copy()
                fixed_corner = None
                if self.drag_handle == "tl":
                    new_rect["x"] += dx; new_rect["y"] += dy
                    new_rect["w"] -= dx; new_rect["h"] -= dy
                    fixed_corner = (self.drag_start_rect["x"] + self.drag_start_rect["w"], 
                                    self.drag_start_rect["y"] + self.drag_start_rect["h"])
                elif self.drag_handle == "tr":
                    new_rect["y"] += dy; new_rect["w"] += dx; new_rect["h"] -= dy
                    fixed_corner = (self.drag_start_rect["x"], self.drag_start_rect["y"] + self.drag_start_rect["h"])
                elif self.drag_handle == "bl":
                    new_rect["x"] += dx; new_rect["w"] -= dx; new_rect["h"] += dy
                    fixed_corner = (self.drag_start_rect["x"] + self.drag_start_rect["w"], self.drag_start_rect["y"])
                elif self.drag_handle == "br":
                    new_rect["w"] += dx; new_rect["h"] += dy
                    fixed_corner = (self.drag_start_rect["x"], self.drag_start_rect["y"])

                min_size = 20
                if new_rect["w"] < min_size: new_rect["w"] = min_size
                if new_rect["h"] < min_size: new_rect["h"] = min_size
                self.user_rect = self.clamp_rect(new_rect, self.max_rect, self.max_area, self.drag_handle, fixed_corner)

            self.update_rectangle_overlay()

    def on_mouse_release(self, sender, app_data):
        crop_mode = dpg.get_value("crop_mode") if dpg.does_item_exist("crop_mode") else False
        if crop_mode:
            self.drag_active = False
            self.drag_mode = None
            self.drag_handle = None
            self.update_rectangle_overlay()

    def clamp_rect(self, rect, max_rect, max_area, handle=None, fixed_corner=None):
        if rect["x"] < max_rect["x"]: rect["x"] = max_rect["x"]
        if rect["y"] < max_rect["y"]: rect["y"] = max_rect["y"]
        if rect["x"] + rect["w"] > max_rect["x"] + max_rect["w"]:
            rect["w"] = max_rect["x"] + max_rect["w"] - rect["x"]
        if rect["y"] + rect["h"] > max_rect["y"] + max_rect["h"]:
            rect["h"] = max_rect["y"] + max_rect["h"] - rect["y"]
        
        area = rect["w"] * rect["h"]
        if area > max_area:
            cx = rect["x"] + rect["w"] / 2
            cy = rect["y"] + rect["h"] / 2
            factor = np.sqrt(max_area / area)
            new_w = rect["w"] * factor
            new_h = rect["h"] * factor
            rect["w"] = new_w; rect["h"] = new_h
            rect["x"] = cx - new_w / 2; rect["y"] = cy - new_h / 2
            if rect["x"] < max_rect["x"]: rect["x"] = max_rect["x"]
            if rect["y"] < max_rect["y"]: rect["y"] = max_rect["y"]
        return rect

    def hit_test_handles(self, x, y, rect):
        threshold = 10
        handles = {
            "tl": (rect["x"], rect["y"]),
            "tr": (rect["x"] + rect["w"], rect["y"]),
            "bl": (rect["x"], rect["y"] + rect["h"]),
            "br": (rect["x"] + rect["w"], rect["y"] + rect["h"])
        }
        for handle, pos in handles.items():
            if abs(x - pos[0]) <= threshold and abs(y - pos[1]) <= threshold:
                return handle
        return None

    def point_in_rect(self, x, y, rect):
        margin = 5
        return (rect["x"] - margin <= x <= rect["x"] + rect["w"] + margin) and \
               (rect["y"] - margin <= y <= rect["y"] + rect["h"] + margin)

    def set_to_max_rect(self, sender, app_data, user_data):
        self.user_rect = self.max_rect.copy()
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