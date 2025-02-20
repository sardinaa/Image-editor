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
    """
    Calcula el ancho (wr) y alto (hr) del rectángulo (inscrito y con ejes paralelos al original)
    de área máxima que cabe en un rectángulo de dimensiones (w, h) rotado un ángulo 'angle'
    (en radianes). La función normaliza el ángulo a [0, π/2] para cubrir todos los cuadrantes.
    """
    if w <= 0 or h <= 0:
        return 0, 0

    # Normalizar el ángulo para que esté entre 0 y π/2
    angle = abs(angle) % np.pi
    if angle > np.pi / 2:
        angle = np.pi - angle

    sin_a = np.sin(angle)
    cos_a = np.cos(angle)

    # Determinar cuál lado es mayor (esto nos ayuda a definir el caso)
    width_is_longer = w >= h
    if width_is_longer:
        long_side = w
        short_side = h
    else:
        long_side = h
        short_side = w

    # Caso "half constrained": el lado corto es tan pequeño que limita el rectángulo inscrito
    if short_side <= 2 * sin_a * cos_a * long_side:
        x = 0.5 * short_side
        if width_is_longer:
            wr = x / sin_a
            hr = x / cos_a
        else:
            wr = x / cos_a
            hr = x / sin_a
    else:
        # Caso general: se usa la fórmula derivada de la geometría del problema
        cos_2a = cos_a * cos_a - sin_a * sin_a
        wr = (w * cos_a - h * sin_a) / cos_2a
        hr = (h * cos_a - w * sin_a) / cos_2a

    return wr, hr

def update_image(sender, app_data, user_data):
    angle = dpg.get_value("rotation_slider")
    zoom  = dpg.get_value("zoom_slider")
    
    # Rotar y escalar la imagen original
    rotated_image = rotate_image(original_image, angle, zoom)
    
    # Calcular el rectángulo inscrito máximo (usando las dimensiones originales y el ángulo en radianes)
    orig_h, orig_w = original_image.shape[:2]
    angle_rad = np.deg2rad(angle)
    inscribed_w, inscribed_h = rotatedRectWithMaxArea(orig_w, orig_h, angle_rad)
    inscribed_w = int(inscribed_w * zoom)
    inscribed_h = int(inscribed_h * zoom)
    
    # Calcular la posición para centrar el rectángulo en el canvas
    x = int((orig_w - inscribed_w) / 2)
    y = int((orig_h - inscribed_h) / 2)
    
    # Crear el overlay: una imagen del mismo tamaño que la imagen rotada, rellena con un color semitransparente
    overlay = np.full(rotated_image.shape, (30, 30, 30, 128), dtype=np.uint8)
    # Borrar (poner alfa 0) la zona interior del rectángulo para dejarla transparente
    overlay[y:y+inscribed_h, x:x+inscribed_w] = (0, 0, 0, 0)
    
    # Componer el overlay sobre la imagen rotada
    base = rotated_image.astype(np.float32)
    over = overlay.astype(np.float32)
    alpha = over[..., 3:4] / 255.0
    blended = base * (1 - alpha) + over * alpha
    blended = np.clip(blended, 0, 255).astype(np.uint8)
    
    # Dibujar el borde del rectángulo inscrito (por ejemplo, en rojo, grosor 2)
    border_color = (255, 255, 255, 200)
    border_thickness = 1
    cv2.rectangle(blended, (x, y), (x + inscribed_w, y + inscribed_h), border_color, border_thickness)
    
    # Actualizar la textura en Dear PyGui (normalizando a [0, 1])
    dpg.set_value("texture_tag", blended.flatten() / 255.0)

# Cargar la imagen original
original_image = load_image('sample.png')
orig_h, orig_w = original_image.shape[:2]
d = np.sqrt(orig_w**2 + orig_h**2)
# Calcular una escala predeterminada para que la imagen no se corte al rotar (basada en la diagonal)
default_scale = min(orig_w, orig_h) / d

dpg.create_context()

with dpg.texture_registry():
    dpg.add_dynamic_texture(orig_w, orig_h, original_image.flatten() / 255.0, tag="texture_tag")

with dpg.window(label="Rotación, Zoom, Overlay y Borde"):
    dpg.add_image("texture_tag")
    dpg.add_slider_float(label="Ángulo de Rotación", tag="rotation_slider",
                         min_value=0, max_value=360, default_value=0, callback=update_image)
    dpg.add_slider_float(label="Zoom", tag="zoom_slider",
                         default_value=default_scale, min_value=0.1, max_value=2.0, callback=update_image)

dpg.create_viewport(title='Visualizador de Imágenes', width=orig_w*2, height=orig_h*2+60)
dpg.setup_dearpygui()
dpg.show_viewport()
dpg.start_dearpygui()
dpg.destroy_context()
