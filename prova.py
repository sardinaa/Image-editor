import cv2
import numpy as np
import dearpygui.dearpygui as dpg

# Función para cargar y preparar la imagen
def load_image(path):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
    return image

# Función para rotar la imagen con un factor de escala
def rotate_image(image, angle, scale=1.0):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated

# Función que calcula el rectángulo inscrito más grande dentro de un rectángulo rotado.
# La función toma el ancho y alto originales y el ángulo de rotación (en radianes).
import numpy as np

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
        # En este caso, el rectángulo inscrito resulta ser un cuadrado (o casi)
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

# Callback para actualizar la imagen (rotación, zoom y dibujo del rectángulo)
def update_image(sender, app_data, user_data):
    angle = dpg.get_value("rotation_slider")
    zoom  = dpg.get_value("zoom_slider")
    
    # Rotar y escalar la imagen
    rotated_image = rotate_image(original_image, angle, zoom)
    
    # Calcular el rectángulo inscrito (se usa el ancho y alto originales y se convierte el ángulo a radianes)
    angle_rad = np.deg2rad(angle)
    orig_h, orig_w = original_image.shape[:2]
    rect_w, rect_h = rotatedRectWithMaxArea(orig_w, orig_h, angle_rad)
    # Aplicar el factor de zoom (pues la imagen se escaló)
    rect_w *= zoom
    rect_h *= zoom
    
    # Calcular la posición para centrar el rectángulo en el canvas (que tiene tamaño orig_w x orig_h)
    x = int((orig_w - rect_w) / 2)
    y = int((orig_h - rect_h) / 2)
    rect_w = int(rect_w)
    rect_h = int(rect_h)
    
    # Dibujar el rectángulo sobre una copia de la imagen rotada (en color rojo, grosor 2)
    rotated_with_rect = rotated_image.copy()
    cv2.rectangle(rotated_with_rect, (x, y), (x + rect_w, y + rect_h), (255, 0, 0, 255), 2)
    
    # Actualizar la textura de Dear PyGui con la imagen resultante (normalizando a [0,1])
    dpg.set_value("texture_tag", rotated_with_rect.flatten() / 255.0)

# Cargar la imagen original
original_image = load_image('sample.png')
orig_h, orig_w = original_image.shape[:2]
d = np.sqrt(orig_w**2 + orig_h**2)
# Calcular una escala predeterminada para que, al rotar, la imagen no se corte (la escala es menor que 1)
default_scale = min(orig_w, orig_h) / d

# Crear el contexto de Dear PyGui
dpg.create_context()

# Crear la textura inicial (imagen sin transformar)
with dpg.texture_registry():
    dpg.add_dynamic_texture(orig_w, orig_h, original_image.flatten() / 255.0, tag="texture_tag")

# Crear la ventana principal con sliders para rotación y zoom
with dpg.window(label="Rotación, Zoom y Rectángulo Inscrito"):
    dpg.add_image("texture_tag")
    dpg.add_slider_float(label="Ángulo de Rotación", tag="rotation_slider", min_value=0, max_value=360, default_value=0, callback=update_image)
    dpg.add_slider_float(label="Zoom", tag="zoom_slider", default_value=default_scale, min_value=0.1, max_value=2.0, callback=update_image)

# Configurar y mostrar la ventana principal
dpg.create_viewport(title='Visualizador de Imágenes', width=orig_w*2, height=orig_h*2 + 60)
dpg.setup_dearpygui()
dpg.show_viewport()
dpg.start_dearpygui()
dpg.destroy_context()
