import cv2
import numpy as np
import dearpygui.dearpygui as dpg

def load_image(path):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
    return image

def rotate_image(image, angle, scale=1.0):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated

def rotatedRectWithMaxArea(w, h, angle):
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

# Variables globales para la interacción
drag_active = False
drag_mode = None  # "resize" o "move"
drag_handle = None
drag_start_mouse = None
drag_start_rect = {}
drag_offset = (0,0)  # Diferencia entre el punto clic y la esquina superior izquierda
user_rect = {}     # Rectángulo interactivo actual
prev_zoom = None   # Para mantener la escala relativa al cambiar zoom

def point_in_rect(x, y, rect):
    return (rect["x"] <= x <= rect["x"] + rect["w"]) and (rect["y"] <= y <= rect["y"] + rect["h"])

def hit_test_handles(mouse_x, mouse_y, rect):
    # Área de detección dinámica: mínimo 10 o 5% del tamaño menor del rectángulo.
    dynamic_threshold = max(10, int(min(rect["w"], rect["h"]) * 0.05))
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

def clamp_rect(rect, max_rect, max_area, handle=None, fixed_corner=None):
    # Forzamos que el rectángulo quede dentro de max_rect.
    if rect["x"] < max_rect["x"]:
        rect["x"] = max_rect["x"]
    if rect["y"] < max_rect["y"]:
        rect["y"] = max_rect["y"]
    if rect["x"] + rect["w"] > max_rect["x"] + max_rect["w"]:
        rect["w"] = max_rect["x"] + max_rect["w"] - rect["x"]
    if rect["y"] + rect["h"] > max_rect["y"] + max_rect["h"]:
        rect["h"] = max_rect["y"] + max_rect["h"] - rect["y"]
    
    # Si el área es mayor que la permitida, se escala uniformemente manteniendo el centro.
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

def update_image(sender, app_data, user_data):
    global user_rect, drag_mode, prev_zoom
    angle = dpg.get_value("rotation_slider")
    zoom  = dpg.get_value("zoom_slider")
    
    rotated_image = rotate_image(original_image, angle, zoom)
    orig_h, orig_w = original_image.shape[:2]
    angle_rad = np.deg2rad(angle)
    inscribed_w, inscribed_h = rotatedRectWithMaxArea(orig_w, orig_h, angle_rad)
    inscribed_w = int(inscribed_w * zoom)
    inscribed_h = int(inscribed_h * zoom)
    max_area = inscribed_w * inscribed_h
    max_rect = {
        "x": int((orig_w - inscribed_w) / 2),
        "y": int((orig_h - inscribed_h) / 2),
        "w": inscribed_w,
        "h": inscribed_h
    }
    
    # Inicializar prev_zoom si es la primera vez
    if prev_zoom is None:
        prev_zoom = zoom

    if not user_rect:
        user_rect = max_rect.copy()
    else:
        # Si se cambia el zoom y NO se está arrastrando, se reajusta user_rect para conservar
        # la posición y dimensiones relativas respecto a max_rect.
        if not drag_active and zoom != prev_zoom:
            old_zoom = prev_zoom
            old_inscribed_w, old_inscribed_h = rotatedRectWithMaxArea(orig_w, orig_h, angle_rad)
            old_inscribed_w = int(old_inscribed_w * old_zoom)
            old_inscribed_h = int(old_inscribed_h * old_zoom)
            old_max_rect = {
                "x": int((orig_w - old_inscribed_w) / 2),
                "y": int((orig_h - old_inscribed_h) / 2),
                "w": old_inscribed_w,
                "h": old_inscribed_h
            }
            norm_x = (user_rect["x"] - old_max_rect["x"]) / old_max_rect["w"]
            norm_y = (user_rect["y"] - old_max_rect["y"]) / old_max_rect["h"]
            norm_w = user_rect["w"] / old_max_rect["w"]
            norm_h = user_rect["h"] / old_max_rect["h"]
            user_rect["x"] = max_rect["x"] + norm_x * max_rect["w"]
            user_rect["y"] = max_rect["y"] + norm_y * max_rect["h"]
            user_rect["w"] = norm_w * max_rect["w"]
            user_rect["h"] = norm_h * max_rect["h"]
            prev_zoom = zoom
        else:
            if drag_mode != "move":
                user_rect = clamp_rect(user_rect.copy(), max_rect, max_area)
            else:
                new_rect = user_rect.copy()
                new_rect["x"] = max(max_rect["x"], min(new_rect["x"], max_rect["x"] + max_rect["w"] - new_rect["w"]))
                new_rect["y"] = max(max_rect["y"], min(new_rect["y"], max_rect["y"] + max_rect["h"] - new_rect["h"]))
                user_rect = new_rect

    overlay = np.full(rotated_image.shape, (30, 30, 30, 128), dtype=np.uint8)
    rx, ry, rw, rh = map(int, (user_rect["x"], user_rect["y"], user_rect["w"], user_rect["h"]))
    overlay[ry:ry+rh, rx:rx+rw] = (0, 0, 0, 0)
    
    base = rotated_image.astype(np.float32)
    over = overlay.astype(np.float32)
    alpha = over[..., 3:4] / 255.0
    blended = base * (1 - alpha) + over * alpha
    blended = np.clip(blended, 0, 255).astype(np.uint8)
    
    # Dibujar borde del rectángulo
    border_color = (255, 255, 255, 200)
    cv2.rectangle(blended, (rx, ry), (rx + rw, ry + rh), border_color, 1)
    
    # Dibujar marcadores (handles) con radio dinámico
    handles = {
        "tl": (user_rect["x"], user_rect["y"]),
        "tr": (user_rect["x"] + user_rect["w"], user_rect["y"]),
        "bl": (user_rect["x"], user_rect["y"] + user_rect["h"]),
        "br": (user_rect["x"] + user_rect["w"], user_rect["y"] + user_rect["h"]),
        "top": (user_rect["x"] + user_rect["w"] // 2, user_rect["y"]),
        "bottom": (user_rect["x"] + user_rect["w"] // 2, user_rect["y"] + user_rect["h"]),
        "left": (user_rect["x"], user_rect["y"] + user_rect["h"] // 2),
        "right": (user_rect["x"] + user_rect["w"], user_rect["y"] + user_rect["h"] // 2)
    }
    dynamic_radius = max(3, int(min(user_rect["w"], user_rect["h"]) * 0.03))
    for pos in handles.values():
        cv2.circle(blended, (int(pos[0]), int(pos[1])), dynamic_radius, (0,255,0,255), -1)
    
    dpg.set_value("texture_tag", blended.flatten() / 255.0)

def on_mouse_down(sender, app_data):
    global drag_active, drag_mode, drag_handle, drag_start_mouse, drag_start_rect, drag_offset
    mouse_pos = dpg.get_mouse_pos()
    image_pos = dpg.get_item_pos("texture_tag")
    local_mouse = (mouse_pos[0] - image_pos[0], mouse_pos[1] - image_pos[1])
    handle = hit_test_handles(local_mouse[0], local_mouse[1], user_rect)
    if handle:
        drag_active = True
        drag_mode = "resize"
        drag_handle = handle
        drag_start_mouse = local_mouse
        drag_start_rect = user_rect.copy()
    elif point_in_rect(local_mouse[0], local_mouse[1], user_rect):
        drag_active = True
        drag_mode = "move"
        drag_offset = (local_mouse[0] - user_rect["x"], local_mouse[1] - user_rect["y"])
        drag_start_rect = user_rect.copy()

def on_mouse_drag(sender, app_data):
    global drag_active, drag_mode, drag_handle, drag_start_mouse, drag_start_rect, user_rect, drag_offset
    if not drag_active:
        return
    mouse_pos = dpg.get_mouse_pos()
    image_pos = dpg.get_item_pos("texture_tag")
    local_mouse = (mouse_pos[0] - image_pos[0], mouse_pos[1] - image_pos[1])
    
    orig_h, orig_w = original_image.shape[:2]
    angle = dpg.get_value("rotation_slider")
    zoom  = dpg.get_value("zoom_slider")
    angle_rad = np.deg2rad(angle)
    inscribed_w, inscribed_h = rotatedRectWithMaxArea(orig_w, orig_h, angle_rad)
    inscribed_w = int(inscribed_w * zoom)
    inscribed_h = int(inscribed_h * zoom)
    max_area = inscribed_w * inscribed_h
    max_rect = {
        "x": int((orig_w - inscribed_w) / 2),
        "y": int((orig_h - inscribed_h) / 2),
        "w": inscribed_w,
        "h": inscribed_h
    }
    
    if drag_mode == "move":
        new_rect = drag_start_rect.copy()
        new_rect["x"] = local_mouse[0] - drag_offset[0]
        new_rect["y"] = local_mouse[1] - drag_offset[1]
        new_rect["x"] = max(max_rect["x"], min(new_rect["x"], max_rect["x"] + max_rect["w"] - new_rect["w"]))
        new_rect["y"] = max(max_rect["y"], min(new_rect["y"], max_rect["y"] + max_rect["h"] - new_rect["h"]))
        user_rect = new_rect
    elif drag_mode == "resize":
        dx = local_mouse[0] - drag_start_mouse[0]
        dy = local_mouse[1] - drag_start_mouse[1]
        new_rect = drag_start_rect.copy()
        fixed_corner = None
        if drag_handle == "tl":
            new_rect["x"] += dx
            new_rect["y"] += dy
            new_rect["w"] -= dx
            new_rect["h"] -= dy
            fixed_corner = (drag_start_rect["x"] + drag_start_rect["w"],
                            drag_start_rect["y"] + drag_start_rect["h"])
        elif drag_handle == "tr":
            new_rect["y"] += dy
            new_rect["w"] += dx
            new_rect["h"] -= dy
            fixed_corner = (drag_start_rect["x"],
                            drag_start_rect["y"] + drag_start_rect["h"])
        elif drag_handle == "bl":
            new_rect["x"] += dx
            new_rect["w"] -= dx
            new_rect["h"] += dy
            fixed_corner = (drag_start_rect["x"] + drag_start_rect["w"],
                            drag_start_rect["y"])
        elif drag_handle == "br":
            new_rect["w"] += dx
            new_rect["h"] += dy
            fixed_corner = (drag_start_rect["x"], drag_start_rect["y"])
        elif drag_handle == "top":
            new_rect["y"] += dy
            new_rect["h"] -= dy
        elif drag_handle == "bottom":
            new_rect["h"] += dy
        elif drag_handle == "left":
            new_rect["x"] += dx
            new_rect["w"] -= dx
        elif drag_handle == "right":
            new_rect["w"] += dx

        min_size = 20
        if new_rect["w"] < min_size:
            new_rect["w"] = min_size
            if drag_handle in ["tl", "bl", "left"]:
                new_rect["x"] = drag_start_rect["x"] + drag_start_rect["w"] - min_size
        if new_rect["h"] < min_size:
            new_rect["h"] = min_size
            if drag_handle in ["tl", "tr", "top"]:
                new_rect["y"] = drag_start_rect["y"] + drag_start_rect["h"] - min_size

        new_rect = clamp_rect(new_rect, max_rect, max_area, drag_handle, fixed_corner)
        user_rect = new_rect

    update_image(None, None, None)

def on_mouse_release(sender, app_data):
    global drag_active, drag_mode, drag_handle, drag_start_mouse, drag_start_rect, drag_offset
    drag_active = False
    drag_mode = None
    drag_handle = None
    drag_start_mouse = None
    drag_start_rect = {}
    drag_offset = (0,0)
    update_image(None, None, None)

def crop_image(sender, app_data, user_data):
    global user_rect
    angle = dpg.get_value("rotation_slider")
    zoom  = dpg.get_value("zoom_slider")
    rotated = rotate_image(original_image, angle, zoom)
    rx, ry, rw, rh = map(int, (user_rect["x"], user_rect["y"], user_rect["w"], user_rect["h"]))
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

def set_to_max_rect(sender, app_data, user_data):
    global user_rect, prev_zoom
    angle = dpg.get_value("rotation_slider")
    zoom  = dpg.get_value("zoom_slider")
    orig_h, orig_w = original_image.shape[:2]
    angle_rad = np.deg2rad(angle)
    inscribed_w, inscribed_h = rotatedRectWithMaxArea(orig_w, orig_h, angle_rad)
    inscribed_w = int(inscribed_w * zoom)
    inscribed_h = int(inscribed_h * zoom)
    max_rect = {
        "x": int((orig_w - inscribed_w) / 2),
        "y": int((orig_h - inscribed_h) / 2),
        "w": inscribed_w,
        "h": inscribed_h
    }
    user_rect = max_rect.copy()
    prev_zoom = zoom
    update_image(None, None, None)

# Cargar la imagen original y calcular el zoom por defecto
original_image = load_image('sample.png')
orig_h, orig_w = original_image.shape[:2]
d = np.sqrt(orig_w**2 + orig_h**2)
default_scale = min(orig_w, orig_h) / d
prev_zoom = default_scale

# Inicializar user_rect con el rectángulo inscripto (ángulo 0, zoom por defecto)
angle = 0
zoom = default_scale
angle_rad = np.deg2rad(angle)
inscribed_w, inscribed_h = rotatedRectWithMaxArea(orig_w, orig_h, angle_rad)
inscribed_w = int(inscribed_w * zoom)
inscribed_h = int(inscribed_h * zoom)
user_rect = {
    "x": int((orig_w - inscribed_w) / 2),
    "y": int((orig_h - inscribed_h) / 2),
    "w": inscribed_w,
    "h": inscribed_h
}

dpg.create_context()
with dpg.texture_registry():
    dpg.add_dynamic_texture(orig_w, orig_h, original_image.flatten() / 255.0, tag="texture_tag")

with dpg.window(label="Rotación, Zoom, Overlay y Borde"):
    dpg.add_image("texture_tag")
    dpg.add_slider_float(label="Ángulo de Rotación", tag="rotation_slider",
                         min_value=0, max_value=360, default_value=0, callback=update_image)
    dpg.add_slider_float(label="Zoom", tag="zoom_slider",
                         default_value=default_scale, min_value=0.1, max_value=2.0, callback=update_image)
    dpg.add_button(label="Máxima Área", callback=set_to_max_rect)
    dpg.add_button(label="Crop", callback=crop_image)

with dpg.handler_registry():
    dpg.add_mouse_down_handler(callback=on_mouse_down)
    dpg.add_mouse_drag_handler(callback=on_mouse_drag)
    dpg.add_mouse_release_handler(callback=on_mouse_release)

dpg.create_viewport(title='Visualizador de Imágenes', width=orig_w*2, height=orig_h*2+60)
dpg.setup_dearpygui()
dpg.show_viewport()
dpg.start_dearpygui()
dpg.destroy_context()
