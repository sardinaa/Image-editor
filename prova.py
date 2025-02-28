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
        
        # Inicializar el rectángulo del usuario (centrado)
        angle = 0
        zoom = self.default_scale
        angle_rad = np.deg2rad(angle)
        inscribed_w, inscribed_h = self.rotatedRectWithMaxArea(self.orig_w, self.orig_h, angle_rad)
        inscribed_w = int(inscribed_w * zoom)
        inscribed_h = int(inscribed_h * zoom)
        self.user_rect = {
            "x": int((self.orig_w - inscribed_w) / 2),
            "y": int((self.orig_h - inscribed_h) / 2),
            "w": inscribed_w,
            "h": inscribed_h
        }
        
        # Variables de interacción
        self.drag_active = False
        self.drag_mode = None
        self.texture_tag = "crop_rotate_texture"
        self.rotation_slider = "rotation_slider"
        self.zoom_slider = "zoom_slider"

    def load_image(self, path):
        image = cv2.imread(path, cv2.IMREAD_UNCHANGED) 
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
        return image

    def rotate_image(self, image, M):
        # Aplicar la transformación afín (rotación y zoom)
        return cv2.warpAffine(image, M, (self.orig_w, self.orig_h), flags=cv2.INTER_LINEAR)


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

    def point_in_rect(self, x, y, rect):
        return (rect["x"] <= x <= rect["x"] + rect["w"]) and (rect["y"] <= y <= rect["y"] + rect["h"])

    def hit_test_handles(self, mouse_x, mouse_y, rect):
        dynamic_threshold = max(10, int(min(rect["w"], rect["h"]) * 0.10))
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
            if abs(mouse_x - pos[0]) <= dynamic_threshold and abs(mouse_y - pos[1]) <= dynamic_threshold:
                return handle
        return None

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
        # Obtener dimensiones actuales del panel central
        panel_w, panel_h = dpg.get_item_rect_size("Central Panel")
        if panel_w <= 0 or panel_h <= 0:
            panel_w, panel_h = self.orig_w, self.orig_h

        # Recuperar parámetros actuales
        angle = dpg.get_value(self.rotation_slider)
        zoom  = dpg.get_value(self.zoom_slider)
        tint = dpg.get_value("color_picker") if dpg.does_item_exist("color_picker") else [1.0, 1.0, 1.0, 1.0]
        tint = np.array(tint[:3], dtype=np.float32)  # Usar solo RGB

        # Calcular la matriz de transformación (rotación y escalado)
        center = (self.orig_w // 2, self.orig_h // 2)
        M = cv2.getRotationMatrix2D(center, angle, zoom)
        
        # Aplicar la transformación en una sola llamada, usando la imagen en float32
        rotated = self.rotate_image(self.original_image_float, M)
        
        # Aplicar tinte de color de forma vectorizada (in-place)
        rotated[..., :3] *= tint.reshape(1, 1, 3)
        np.clip(rotated, 0, 255, out=rotated)
        
        # Convertir a uint8 para operaciones gráficas posteriores
        rotated_uint8 = rotated.astype(np.uint8)
        
        # Recalcular el rectángulo máximo (área inscripta) según ángulo y zoom
        angle_rad = np.deg2rad(angle)
        inscribed_w, inscribed_h = self.rotatedRectWithMaxArea(self.orig_w, self.orig_h, angle_rad)
        inscribed_w = int(inscribed_w * zoom)
        inscribed_h = int(inscribed_h * zoom)
        max_rect = {
            "x": (self.orig_w - inscribed_w) // 2,
            "y": (self.orig_h - inscribed_h) // 2,
            "w": inscribed_w,
            "h": inscribed_h
        }
        max_area = inscribed_w * inscribed_h

        # Ajustar el rectángulo del usuario según cambios de zoom o movimiento
        if not self.user_rect:
            self.user_rect = max_rect.copy()
        else:
            if not self.drag_active and zoom != self.prev_zoom:
                # Calcular parámetros normalizados para mantener la posición relativa
                old_inscribed_w, old_inscribed_h = self.rotatedRectWithMaxArea(self.orig_w, self.orig_h, angle_rad)
                old_inscribed_w = int(old_inscribed_w * self.prev_zoom)
                old_inscribed_h = int(old_inscribed_h * self.prev_zoom)
                old_max_rect = {
                    "x": (self.orig_w - old_inscribed_w) // 2,
                    "y": (self.orig_h - old_inscribed_h) // 2,
                    "w": old_inscribed_w,
                    "h": old_inscribed_h
                }
                norm_x = (self.user_rect["x"] - old_max_rect["x"]) / old_max_rect["w"]
                norm_y = (self.user_rect["y"] - old_max_rect["y"]) / old_max_rect["h"]
                norm_w = self.user_rect["w"] / old_max_rect["w"]
                norm_h = self.user_rect["h"] / old_max_rect["h"]
                self.user_rect["x"] = max_rect["x"] + norm_x * max_rect["w"]
                self.user_rect["y"] = max_rect["y"] + norm_y * max_rect["h"]
                self.user_rect["w"] = norm_w * max_rect["w"]
                self.user_rect["h"] = norm_h * max_rect["h"]
                self.prev_zoom = zoom
            else:
                # Si se está moviendo el recorte, asegurar que se mantenga dentro de los límites
                new_rect = self.user_rect.copy()
                new_rect["x"] = max(max_rect["x"], min(new_rect["x"], max_rect["x"] + max_rect["w"] - new_rect["w"]))
                new_rect["y"] = max(max_rect["y"], min(new_rect["y"], max_rect["y"] + max_rect["h"] - new_rect["h"]))
                self.user_rect = new_rect

        # Forzar la actualización del rectángulo si no se está haciendo drag o si el ángulo cambió
        if not self.drag_active or self.prev_angle != angle:
            tolerance = 1  # Umbral reducido
            if (abs(self.user_rect["x"] - max_rect["x"]) > tolerance or
                abs(self.user_rect["y"] - max_rect["y"]) > tolerance or
                abs(self.user_rect["w"] - max_rect["w"]) > tolerance or
                abs(self.user_rect["h"] - max_rect["h"]) > tolerance):
                self.user_rect = max_rect.copy()
        self.prev_angle = angle

        # ---------------------------
        # RECORTE Y CENTRADO DEL ÁREA VISIBLE
        # ---------------------------
        # Si la imagen es más grande que el panel, se recorta el área central de tamaño (panel_w x panel_h)
        if self.orig_w > panel_w or self.orig_h > panel_h:
            start_x = max((self.orig_w - panel_w) // 2, 0)
            start_y = max((self.orig_h - panel_h) // 2, 0)
            end_x = start_x + panel_w
            end_y = start_y + panel_h
        else:
            start_x, start_y = 0, 0
            end_x, end_y = self.orig_w, self.orig_h

        cropped_image = rotated_uint8[start_y:end_y, start_x:end_x]

        # Ajustar las coordenadas del rectángulo del usuario al sistema de coordenadas del recorte
        rx = int(self.user_rect["x"]) - start_x
        ry = int(self.user_rect["y"]) - start_y
        rw = int(self.user_rect["w"])
        rh = int(self.user_rect["h"])
        # Aplicar clamp para que el rectángulo quede dentro del cropped_image
        rx_clamped = max(rx, 0)
        ry_clamped = max(ry, 0)
        rx_end = min(rx + rw, cropped_image.shape[1])
        ry_end = min(ry + rh, cropped_image.shape[0])

        # Crear overlay: oscurecer la zona exterior al rectángulo inscripto
        overlay = np.full(cropped_image.shape, (30, 30, 30, 100), dtype=np.uint8)
        overlay[ry_clamped:ry_end, rx_clamped:rx_end] = 0  # Área del recorte sin oscurecimiento

        base = cropped_image.astype(np.float32)
        over = overlay.astype(np.float32)
        alpha = over[..., 3:4] / 255.0
        blended = base * (1 - alpha) + over * alpha
        blended = np.clip(blended, 0, 255).astype(np.uint8)

        # Dibujar borde y "handles" en el rectángulo inscripto sobre la imagen recortada
        cv2.rectangle(blended, (rx_clamped, ry_clamped), (rx_end, ry_end), (255, 255, 255, 200), 1)
        for corner in [(rx_clamped, ry_clamped), (rx_end, ry_clamped),
                    (rx_clamped, ry_end), (rx_end, ry_end)]:
            cv2.circle(blended, corner, radius=5, color=(255, 0, 0, 255), thickness=-1)
        # ---------------------------
        # FIN RECORTE Y CENTRADO
        # ---------------------------
        
        # Actualizar la textura en Dear PyGui (se normaliza la imagen a rango [0,1])
        texture_data = blended.flatten().astype(np.float32) / 255.0
        dpg.set_value(self.texture_tag, texture_data)
        # Centrar el widget de imagen en el panel (puedes ajustar según tus necesidades)
        self.center_image_in_panel()



    def center_image_in_panel(self):
        # Centrar el widget d'imatge dins del "Central Panel"
        panel_pos = dpg.get_item_pos("Central Panel")
        panel_w, panel_h = dpg.get_item_rect_size("Central Panel")
        image_w = dpg.get_item_width("central_image")
        image_h = dpg.get_item_height("central_image")
        offset_x = (panel_w - self.orig_w) // 2
        offset_y = (panel_h - self.orig_h) // 2
        dpg.set_item_pos("central_image", [offset_x, offset_y])

    def on_mouse_down(self, sender, app_data):
        self.drag_active = False
        mouse_pos = dpg.get_mouse_pos()
        # Utilitza "central_image" per obtenir la posició del widget
        image_pos = dpg.get_item_pos("central_image")
        local_mouse = (mouse_pos[0] - image_pos[0], mouse_pos[1] - image_pos[1])
        handle = self.hit_test_handles(local_mouse[0], local_mouse[1], self.user_rect)
        if handle:
            self.drag_active = True
            self.drag_mode = "resize"
            self.drag_handle = handle
            self.drag_start_mouse = local_mouse
            self.drag_start_rect = self.user_rect.copy()
        elif self.point_in_rect(local_mouse[0], local_mouse[1], self.user_rect):
            self.drag_active = True
            self.drag_mode = "move"
            self.drag_offset = (local_mouse[0] - self.user_rect["x"], local_mouse[1] - self.user_rect["y"])
            self.drag_start_rect = self.user_rect.copy()

    def on_mouse_drag(self, sender, app_data):
        if not self.drag_active:
            return
        mouse_pos = dpg.get_mouse_pos()
        # Utilitza "central_image" per obtenir la posició del widget
        image_pos = dpg.get_item_pos("central_image")
        local_mouse = (mouse_pos[0] - image_pos[0], mouse_pos[1] - image_pos[1])
        
        orig_h, orig_w = self.orig_h, self.orig_w
        angle = dpg.get_value(self.rotation_slider)
        zoom  = dpg.get_value(self.zoom_slider)
        angle_rad = np.deg2rad(angle)
        inscribed_w, inscribed_h = self.rotatedRectWithMaxArea(orig_w, orig_h, angle_rad)
        inscribed_w = int(inscribed_w * zoom)
        inscribed_h = int(inscribed_h * zoom)
        max_area = inscribed_w * inscribed_h
        max_rect = {
            "x": int((orig_w - inscribed_w) / 2),
            "y": int((orig_h - inscribed_h) / 2),
            "w": inscribed_w,
            "h": inscribed_h
        }
        
        if self.drag_mode == "move":
            new_rect = self.drag_start_rect.copy()
            new_rect["x"] = local_mouse[0] - self.drag_offset[0]
            new_rect["y"] = local_mouse[1] - self.drag_offset[1]
            new_rect["x"] = max(max_rect["x"], min(new_rect["x"], max_rect["x"] + max_rect["w"] - new_rect["w"]))
            new_rect["y"] = max(max_rect["y"], min(new_rect["y"], max_rect["y"] + max_rect["h"] - new_rect["h"]))
            self.user_rect = new_rect
        elif self.drag_mode == "resize":
            dx = local_mouse[0] - self.drag_start_mouse[0]
            dy = local_mouse[1] - self.drag_start_mouse[1]
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
                fixed_corner = (self.drag_start_rect["x"],
                                self.drag_start_rect["y"] + self.drag_start_rect["h"])
            elif self.drag_handle == "bl":
                new_rect["x"] += dx
                new_rect["w"] -= dx
                new_rect["h"] += dy
                fixed_corner = (self.drag_start_rect["x"] + self.drag_start_rect["w"],
                                self.drag_start_rect["y"])
            elif self.drag_handle == "br":
                new_rect["w"] += dx
                new_rect["h"] += dy
                fixed_corner = (self.drag_start_rect["x"], self.drag_start_rect["y"])
            elif self.drag_handle == "top":
                new_rect["y"] += dy
                new_rect["h"] -= dy
            elif self.drag_handle == "bottom":
                new_rect["h"] += dy
            elif self.drag_handle == "left":
                new_rect["x"] += dx
                new_rect["w"] -= dx
            elif self.drag_handle == "right":
                new_rect["w"] += dx

            min_size = 20
            if new_rect["w"] < min_size:
                new_rect["w"] = min_size
                if self.drag_handle in ["tl", "bl", "left"]:
                    new_rect["x"] = self.drag_start_rect["x"] + self.drag_start_rect["w"] - min_size
            if new_rect["h"] < min_size:
                new_rect["h"] = min_size
                if self.drag_handle in ["tl", "tr", "top"]:
                    new_rect["y"] = self.drag_start_rect["y"] + self.drag_start_rect["h"] - min_size

            self.user_rect = self.clamp_rect(new_rect, max_rect, max_area, self.drag_handle, fixed_corner)
        
        self.update_image(None, None, None)

    def on_mouse_release(self, sender, app_data):
        self.drag_active = False
        self.drag_mode = None
        self.drag_handle = None
        self.drag_start_mouse = None
        self.drag_start_rect = {}
        self.drag_offset = (0,0)
        self.update_image(None, None, None)

    def crop_image(self, sender, app_data, user_data):
        angle = dpg.get_value(self.rotation_slider)
        zoom  = dpg.get_value(self.zoom_slider)
        rotated = self.rotate_image(self.original_image, angle, zoom)
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
        zoom  = dpg.get_value(self.zoom_slider)
        orig_h, orig_w = self.orig_h, self.orig_w
        angle_rad = np.deg2rad(angle)
        inscribed_w, inscribed_h = self.rotatedRectWithMaxArea(orig_w, orig_h, angle_rad)
        inscribed_w = int(inscribed_w * zoom)
        inscribed_h = int(inscribed_h * zoom)
        max_rect = {
            "x": int((orig_w - inscribed_w) / 2),
            "y": int((orig_h - inscribed_h) / 2),
            "w": inscribed_w,
            "h": inscribed_h
        }
        self.user_rect = max_rect.copy()
        self.prev_zoom = zoom
        self.update_image(None, None, None)

    """
    -----------------------------------------
    -----------------------------------------
    Deltete this code for the new refactor
    -----------------------------------------
    -----------------------------------------
    """
    def setup_ui(self, parent_image, parent_controls):
        # Registrar la textura dinámica para la imagen
        with dpg.texture_registry():
            dpg.add_dynamic_texture(self.orig_w, self.orig_h, self.original_image.flatten() / 255.0, tag=self.texture_tag)

        # Agregar la imagen resultante en el grupo destinado a su visualización
        with dpg.group(parent=parent_image):
            dpg.add_image(self.texture_tag)
        
        # Agregar controles de transformación en el grupo destinado a ellos
        with dpg.group(parent=parent_controls):
            with dpg.tree_node(label="Color Picker & Edit"):
                with dpg.group(horizontal=True):
                    dpg.add_knob_float(label="Ángulo de Rotación", tag=self.rotation_slider,
                                        min_value=0, max_value=360, default_value=0, callback=self.update_image)
                    dpg.add_knob_float(label="Zoom", tag=self.zoom_slider,
                                        default_value=self.default_scale, min_value=0.1, max_value=2.0, callback=self.update_image)
                with dpg.group(horizontal=True):
                    dpg.add_button(label="Máxima Área", callback=self.set_to_max_rect)
                    dpg.add_button(label="Crop", callback=self.crop_image)
        
        # Registrar los controladores de eventos del ratón (se aplican globalmente)
        with dpg.handler_registry():
            dpg.add_mouse_down_handler(callback=self.on_mouse_down)
            dpg.add_mouse_drag_handler(callback=self.on_mouse_drag)
            dpg.add_mouse_release_handler(callback=self.on_mouse_release)
        
        # Actualizar la imagen inicialmente para mostrar el overlay y el rectángulo
        self.update_image(None, None, None)
