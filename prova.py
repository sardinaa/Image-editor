import cv2
import numpy as np
import dearpygui.dearpygui as dpg

class CropAndRotate:
    def __init__(self, image_path):
        self.original_image = self.load_image(image_path)
        self.orig_h, self.orig_w = self.original_image.shape[:2]
        self.original_image_float = self.original_image.astype(np.float32)
        d = np.sqrt(self.orig_w**2 + self.orig_h**2)
        self.default_scale = min(self.orig_w, self.orig_h) / d
        self.prev_zoom = self.default_scale
        self.prev_angle = 0

        # Se establecerán después desde main()
        self.texture_w = None
        self.texture_h = None
        self.texture_tag = "crop_rotate_texture"
        self.rotation_slider = "rotation_slider"

        # Inicializar user_rect en coordenadas de textura después de rotación inicial
        self.user_rect = None  # Se establecerá en update_image
        self.drag_active = False
        self.drag_mode = None
        self.rotated_texture = None  # Cache de la imagen rotada
        self.max_rect = None  # Cache del rectángulo máximo
        self.max_area = None  # Cache del área máxima
        self.last_update_time = 0  # Para controlar la frecuencia de actualización
        self.update_interval = 1 / 60  # 60 FPS máxim

    def load_image(self, path):
        image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
        return image

    def rotate_image(self, image, M, width, height):
        return cv2.warpAffine(image, M, (width, height), flags=cv2.INTER_LINEAR, borderValue=(100, 100, 100, 255))

    def rotatedRectWithMaxArea(self, w, h, angle):
        if w <= 0 or h <= 0:
            return 0, 0
        angle = abs(angle) % np.pi
        if angle > np.pi / 2:
            angle = np.pi - angle
        sin_a = np.sin(angle)
        cos_a = np.cos(angle)
        width_is_longer = w >= h
        if width_is_longer:
            long_side = w
            short_side = h
        else:
            long_side = h
            short_side = w
        if short_side <= 2 * sin_a * cos_a * long_side:
            x = 0.5 * short_side
            if width_is_longer:
                wr = x / sin_a
                hr = x / cos_a
            else:
                wr = x / cos_a
                hr = x / sin_a
        else:
            cos_2a = cos_a * cos_a - sin_a * sin_a
            wr = (w * cos_a - h * sin_a) / cos_2a
            hr = (h * cos_a - w * sin_a) / cos_2a
        return wr, hr

    def clamp_rect(self, rect, max_rect, max_area, handle=None, fixed_corner=None):
        if rect["x"] < max_rect["x"]:
            rect["x"] = max_rect["x"]
        if rect["y"] < max_rect["y"]:
            rect["y"] = max_rect["y"]
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
            rect["w"] = new_w
            rect["h"] = new_h
            rect["x"] = cx - new_w / 2
            rect["y"] = cy - new_h / 2
            if rect["x"] < max_rect["x"]:
                rect["x"] = max_rect["x"]
            if rect["y"] < max_rect["y"]:
                rect["y"] = max_rect["y"]
            if rect["x"] + rect["w"] > max_rect["x"] + max_rect["w"]:
                rect["w"] = max_rect["x"] + max_rect["w"] - rect["x"]
            if rect["y"] + rect["h"] > max_rect["y"] + max_rect["h"]:
                rect["h"] = max_rect["y"] + max_rect["h"] - rect["y"]
        return rect

    def update_image(self, sender, app_data, user_data):
        panel_w, panel_h = dpg.get_item_rect_size("Central Panel")
        if panel_w <= 0 or panel_h <= 0:
            panel_w, panel_h = self.texture_w, self.texture_h

        # Calcular los límites de los ejes para mantener la relación de aspecto
        plot_aspect = panel_w / panel_h
        texture_aspect = self.texture_w / self.texture_h  # Es 1, ya que la textura es cuadrada

        if plot_aspect > texture_aspect:
            # El plot es más ancho
            y_min = 0
            y_max = self.texture_h
            x_min = 0
            x_max = self.texture_h * plot_aspect
        else:
            # El plot es más alto
            x_min = 0
            x_max = self.texture_w
            y_min = 0
            y_max = self.texture_w / plot_aspect

        dpg.set_axis_limits("x_axis", x_min, x_max)
        dpg.set_axis_limits("y_axis", y_min, y_max)
        angle = dpg.get_value(self.rotation_slider)
        zoom = 1  # Ajusta si implementas zoom
        tint = np.array([1.0, 1.0, 1.0], dtype=np.float32)

        center = (self.texture_w / 2, self.texture_h / 2)
        M = cv2.getRotationMatrix2D(center, angle, zoom)

        offset_x = (self.texture_w - self.orig_w) // 2
        offset_y = (self.texture_h - self.orig_h) // 2
        padded_image = np.full((self.texture_h, self.texture_w, 4), [100, 100, 100, 255], dtype=np.float32)
        padded_image[offset_y:offset_y + self.orig_h, offset_x:offset_x + self.orig_w] = self.original_image_float

        rotated = self.rotate_image(padded_image, M, self.texture_w, self.texture_h)
        rotated[..., :3] *= tint.reshape(1, 1, 3)
        np.clip(rotated, 0, 255, out=rotated)
        rotated_uint8 = rotated.astype(np.uint8)

        # Cachear la imagen rotada sin el rectángulo
        self.rotated_texture = rotated_uint8.copy()
        
        # Calcular y cachear max_rect y max_area
        angle_rad = np.deg2rad(angle)
        inscribed_w, inscribed_h = self.rotatedRectWithMaxArea(self.orig_w, self.orig_h, angle_rad)
        inscribed_w = int(inscribed_w * zoom)
        inscribed_h = int(inscribed_h * zoom)
        self.max_rect = {
            "x": offset_x + (self.orig_w - inscribed_w) // 2,
            "y": offset_y + (self.orig_h - inscribed_h) // 2,
            "w": inscribed_w,
            "h": inscribed_h
        }
        self.max_area = inscribed_w * inscribed_h

        # Inicializar o resetear user_rect
        if not self.user_rect or (not self.drag_active and self.prev_angle != angle):
            self.user_rect = self.max_rect.copy()
        self.prev_angle = angle

        # Actualizar la visualización con el rectángulo
        self.update_rectangle_overlay()

    def update_rectangle_overlay(self):
        import time
        current_time = time.time()
        if current_time - self.last_update_time < self.update_interval:
            return  # Limitar la frecuencia de actualización
        self.last_update_time = current_time

        if self.rotated_texture is None:
            return
        # Usar una copia ligera o trabajar directamente si es posible
        blended = self.rotated_texture.copy()

        rx = int(self.user_rect["x"])
        ry = int(self.user_rect["y"])
        rw = int(self.user_rect["w"])
        rh = int(self.user_rect["h"])
        rx_clamped = max(rx, 0)
        ry_clamped = max(ry, 0)
        rx_end = min(rx + rw, self.texture_w)
        ry_end = min(ry + rh, self.texture_h)

        # Dibujar solo el rectángulo (sin overlay completo para mejorar rendimiento)
        cv2.rectangle(blended, (rx_clamped, ry_clamped), (rx_end, ry_end), (0, 255, 0, 200), 2)
        for corner in [(rx_clamped, ry_clamped), (rx_end, ry_clamped),
                       (rx_clamped, ry_end), (rx_end, ry_end)]:
            cv2.circle(blended, corner, radius=5, color=(255, 0, 0, 255), thickness=-1)

        texture_data = blended.flatten().astype(np.float32) / 255.0
        dpg.set_value(self.texture_tag, texture_data)

    def on_mouse_down(self, sender, app_data):
        if not self.user_rect:
            return

        self.drag_active = False
        mouse_pos = dpg.get_mouse_pos()
        panel_pos = dpg.get_item_pos("Central Panel")
        panel_w, panel_h = dpg.get_item_rect_size("Central Panel")
        x_min, x_max = dpg.get_axis_limits("x_axis")
        y_min, y_max = dpg.get_axis_limits("y_axis")

        # Ajustar coordenadas al espacio de la textura
        mouse_x = mouse_pos[0] - panel_pos[0]
        mouse_y = mouse_pos[1] - panel_pos[1]
        if not (0 <= mouse_x <= panel_w and 0 <= mouse_y <= panel_h):
            return

        # Mapear al espacio de la textura
        texture_mouse_x = (mouse_x / panel_w) * (self.texture_w)
        texture_mouse_y = (mouse_y / panel_h) * (self.texture_h)

        # Verificar si el clic está dentro del rectángulo
        inside_rect = self.point_in_rect(texture_mouse_x, texture_mouse_y, self.user_rect)

        if inside_rect:
            # Solo permitir movimiento dentro del rectángulo
            self.drag_active = True
            self.drag_mode = "move"
            self.drag_offset = (texture_mouse_x - self.user_rect["x"], texture_mouse_y - self.user_rect["y"])
            self.drag_start_rect = self.user_rect.copy()
        else:
            # Fuera del rectángulo: solo permitir redimensionar si se está cerca de un handle
            handle = self.hit_test_handles(texture_mouse_x, texture_mouse_y, self.user_rect)
            if handle:
                self.drag_active = True
                self.drag_mode = "resize"
                self.drag_handle = handle
                self.drag_start_mouse = (texture_mouse_x, texture_mouse_y)
                self.drag_start_rect = self.user_rect.copy()

    def on_mouse_drag(self, sender, app_data):
        if not self.drag_active:
            return

        mouse_pos = dpg.get_mouse_pos()
        panel_pos = dpg.get_item_pos("Central Panel")
        panel_w, panel_h = dpg.get_item_rect_size("Central Panel")
        
        # Mapear al espacio de la textura
        mouse_x = mouse_pos[0] - panel_pos[0]
        mouse_y = mouse_pos[1] - panel_pos[1]
        texture_mouse_x = (mouse_x / panel_w) * (self.texture_w)
        texture_mouse_y = (mouse_y / panel_h) * (self.texture_h)

        if self.drag_mode == "move":
            # Movimiento dentro del rectángulo
            delta_x = texture_mouse_x - self.drag_start_rect["x"] - self.drag_offset[0]
            delta_y = texture_mouse_y - self.drag_start_rect["y"] - self.drag_offset[1]
            new_x = self.drag_start_rect["x"] + delta_x
            new_y = self.drag_start_rect["y"] + delta_y
            new_x = max(self.max_rect["x"], min(new_x, self.max_rect["x"] + self.max_rect["w"] - self.user_rect["w"]))
            new_y = max(self.max_rect["y"], min(new_y, self.max_rect["y"] + self.max_rect["h"] - self.user_rect["h"]))
            self.user_rect["x"] = new_x
            self.user_rect["y"] = new_y
        elif self.drag_mode == "resize":
            # Redimensionar fuera del rectángulo
            dx = texture_mouse_x - self.drag_start_mouse[0]
            dy = texture_mouse_y - self.drag_start_mouse[1]
            new_rect = self.drag_start_rect.copy()
            fixed_corner = None
            if self.drag_handle == "tl":
                new_rect["x"] += dx
                new_rect["y"] += dy
                new_rect["w"] -= dx
                new_rect["h"] -= dy
                fixed_corner = (self.drag_start_rect["x"] + self.drag_start_rect["w"],
                                self.drag_start_rect["y"] + self.drag_start_rect["h"])
            elif self.drag_handle == "tr":
                new_rect["y"] += dy
                new_rect["w"] += dx
                new_rect["h"] -= dy
                fixed_corner = (self.drag_start_rect["x"], self.drag_start_rect["y"] + self.drag_start_rect["h"])
            elif self.drag_handle == "bl":
                new_rect["x"] += dx
                new_rect["w"] -= dx
                new_rect["h"] += dy
                fixed_corner = (self.drag_start_rect["x"] + self.drag_start_rect["w"], self.drag_start_rect["y"])
            elif self.drag_handle == "br":
                new_rect["w"] += dx
                new_rect["h"] += dy
                fixed_corner = (self.drag_start_rect["x"], self.drag_start_rect["y"])

            min_size = 20
            if new_rect["w"] < min_size:
                new_rect["w"] = min_size
                if self.drag_handle in ["tl", "bl"]:
                    new_rect["x"] = self.drag_start_rect["x"] + self.drag_start_rect["w"] - min_size
            if new_rect["h"] < min_size:
                new_rect["h"] = min_size
                if self.drag_handle in ["tl", "tr"]:
                    new_rect["y"] = self.drag_start_rect["y"] + self.drag_start_rect["h"] - min_size

            self.user_rect = self.clamp_rect(new_rect, self.max_rect, self.max_area, self.drag_handle, fixed_corner)

        self.update_rectangle_overlay()

    def hit_test_handles(self, x, y, rect):
        # Aumentar el umbral para facilitar la interacción
        threshold = 10  # Fijo para hacerlo más accesible, o mantener dinámico si prefieres
        handles = {
            "tl": (rect["x"], rect["y"]),
            "tr": (rect["x"] + rect["w"], rect["y"]),
            "bl": (rect["x"], rect["y"] + rect["h"]),
            "br": (rect["x"] + rect["w"], rect["y"] + rect["h"]),
            "top": (rect["x"] + rect["w"] / 2, rect["y"]),
            "bottom": (rect["x"] + rect["w"] / 2, rect["y"] + rect["h"]),
            "left": (rect["x"], rect["y"] + rect["h"] / 2),
            "right": (rect["x"] + rect["w"], rect["y"] + rect["h"] / 2)
        }
        for handle, pos in handles.items():
            if abs(x - pos[0]) <= threshold and abs(y - pos[1]) <= threshold:
                return handle
        return None

    def point_in_rect(self, x, y, rect):
        # Añadir un pequeño margen para facilitar el clic
        margin = 5
        return (rect["x"] - margin <= x <= rect["x"] + rect["w"] + margin) and \
               (rect["y"] - margin <= y <= rect["y"] + rect["h"] + margin)

    def on_mouse_release(self, sender, app_data):
        self.drag_active = False
        self.drag_mode = None
        self.drag_handle = None
        self.drag_start_mouse = None
        self.drag_start_rect = {}
        self.drag_offset = (0, 0)
        self.update_rectangle_overlay()

    def crop_image(self, sender, app_data, user_data):
        angle = dpg.get_value(self.rotation_slider)
        zoom = self.default_scale
        M = cv2.getRotationMatrix2D((self.orig_w // 2, self.orig_h // 2), angle, zoom)
        rotated = self.rotate_image(self.original_image, M)
        rx, ry, rw, rh = map(int, (self.user_rect["x"], self.user_rect["y"], self.user_rect["w"], self.user_rect["h"]))
        cropped = rotated[ry:ry+rh, rx:rx+rw].copy()
        height, width = cropped.shape[:2]
        cropped_flat = cropped.flatten() / 255.0
        if dpg.does_item_exist("cropped_texture"):
            dpg.set_value("cropped_texture", cropped_flat)
        else:
            with dpg.texture_registry():
                dpg.add_dynamic_texture(width, height, cropped_flat, tag="cropped_texture")
        if not dpg.does_item_exist("CroppedWindow"):
            with dpg.window(label="Cropped Image", tag="CroppedWindow"):
                dpg.add_image("cropped_texture")

    def set_to_max_rect(self, sender, app_data, user_data):
        angle = dpg.get_value(self.rotation_slider)
        angle_rad = np.deg2rad(angle)
        orig_w, orig_h = self.orig_w, self.orig_h
        inscribed_w, inscribed_h = self.rotatedRectWithMaxArea(orig_w, orig_h, angle_rad)
        zoom = 1  # O self.default_scale si prefieres mantenerlo
        inscribed_w = int(inscribed_w * zoom)
        inscribed_h = int(inscribed_h * zoom)
        x = (orig_w - inscribed_w) / 2
        y = (orig_h - inscribed_h) / 2
        self.user_rect = {
            "x": int(x),
            "y": int(y),
            "w": inscribed_w,
            "h": inscribed_h
        }
        self.prev_zoom = zoom
        self.prev_angle = angle
        self.update_image(None, None, None)