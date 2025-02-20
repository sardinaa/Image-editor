# ui/preview_panel.py
import uuid
import numpy as np
import cv2
import dearpygui.dearpygui as dpg
import math

def convert_to_rgba(image):
    if image.shape[2] == 3:
        alpha = np.full((image.shape[0], image.shape[1], 1), 255, dtype=image.dtype)
        image = np.concatenate([image, alpha], axis=2)
    return image

class PreviewPanel:
    def __init__(self, initial_width, initial_height):
        # initial_width, initial_height se fijan al cargar la imagen (diagonal D)
        self.base_width = initial_width
        self.base_height = initial_height  # Incluye espacio para el slider (por ejemplo, +40)
        self.zoom = 1.0
        self.offset = [0.0, 0.0]  # Para panning
        self.last_mouse_pos = None
        self.current_image = None  # Imagen procesada (rotada, etc.)
        self.texture_tag = None
        self.texture_dimensions = None  # (D, D)
        # Se almacenarán los datos de la imagen dentro de la textura:
        self.image_offset = (0, 0)  # offset dentro de la textura donde se colocó la imagen
        self.image_size = (0, 0)    # tamaño (w, h) de la imagen procesada
        self.uv_min = (0, 0)
        self.uv_max = (1, 1)
        # Etiquetas de contenedores
        self.child_container_tag = "preview_child"
        self.drawlist_container_tag = "preview_drawlist_container"
        self.drawlist_tag = "preview_drawlist"
        self.slider_tag = "zoom_slider"
        self._create_texture(self.base_width, self.base_height)
        self.drawlist_size = (0, 0)  # Tamaño dinámico basado en D y zoom
        self.texture_rotated = None  # Nueva textura para la imagen rotada

    def _create_texture(self, width, height):
        """
        Crea una textura dinámica con dimensiones específicas.
        """
        default_image = np.zeros((height, width, 4), dtype=np.float32)
        default_data = default_image.flatten().tolist()
        new_tag = f"preview_texture_{uuid.uuid4()}"
        with dpg.texture_registry(show=False):
            dpg.add_dynamic_texture(width, height, default_data, tag=new_tag)
        self.texture_tag = new_tag
        self.texture_dimensions = (width, height)

    def update_texture(self, image):
        """
        Actualiza la textura dinámica con la nueva imagen.
        Se espera la imagen en formato RGB (o RGBA).
        """
        self.current_image = image.copy()
        image = convert_to_rgba(image)
        image_data = (image.astype(np.float32) / 255.0).flatten().tolist()
        h, w = image.shape[:2]
        if self.texture_dimensions != (w, h):
            self._create_texture(w, h)
        dpg.set_value(self.texture_tag, image_data)

    def update_image(self, image, reset_offset=False):
        if reset_offset:
            self.offset = [0.0, 0.0]
        self.update_texture(image)
        self.draw_image()

    def update_zoom(self, zoom_value):
        self.zoom = zoom_value
        self.draw_image()

    def draw_image(self):
        if not dpg.does_item_exist(self.drawlist_tag):
            return
        dpg.delete_item(self.drawlist_tag, children_only=True)
        if self.current_image is None:
            return

        h, w = self.current_image.shape[:2]
        slider_height = 40
        preview_w = self.base_width
        preview_h = self.base_height - slider_height

        zoomed_w = int(w * self.zoom)
        zoomed_h = int(h * self.zoom)
        center_x = preview_w / 2 + self.offset[0]
        center_y = preview_h / 2 + self.offset[1]
        new_x = center_x - zoomed_w / 2
        new_y = center_y - zoomed_h / 2

        dpg.draw_image(self.texture_tag,
                    pmin=[new_x, new_y],
                    pmax=[new_x + zoomed_w, new_y + zoomed_h],
                    parent=self.drawlist_tag)

        if dpg.get_value("crop_mode"):
            crop_x = dpg.get_value("crop_x")
            crop_y = dpg.get_value("crop_y")
            crop_w = dpg.get_value("crop_w")
            crop_h = dpg.get_value("crop_h")
            drawn_crop_x = new_x + int(crop_x * self.zoom)
            drawn_crop_y = new_y + int(crop_y * self.zoom)
            drawn_crop_w = int(crop_w * self.zoom)
            drawn_crop_h = int(crop_h * self.zoom)
            dpg.draw_rectangle(pmin=[drawn_crop_x, drawn_crop_y],
                            pmax=[drawn_crop_x + drawn_crop_w, drawn_crop_y + drawn_crop_h],
                            color=[0,255,0,255], thickness=2, fill=[0,255,0,50],
                            parent=self.drawlist_tag)


    def set_size(self, new_width, new_height):
        """
        Actualiza el tamaño del lienzo (no se modifica la textura).
        Este método se llama solo al cargar una nueva imagen.
        """
        self.base_width = new_width
        self.base_height = new_height
        dpg.configure_item(self.child_container_tag, width=new_width, height=new_height)
        if dpg.does_item_exist(self.drawlist_container_tag):
            dpg.configure_item(self.drawlist_container_tag, width=new_width, height=new_height)
        if dpg.does_item_exist(self.drawlist_tag):
            dpg.configure_item(self.drawlist_tag, width=new_width, height=new_height)
        if dpg.does_item_exist(self.slider_tag):
            dpg.configure_item(self.slider_tag, width=new_width)
        self.draw_image()

    def mouse_drag_handler(self, sender, app_data, user_data):
        if dpg.is_key_down(dpg.mvKey_LControl):
            current_pos = dpg.get_mouse_pos()
            if self.last_mouse_pos is not None:
                dx = current_pos[0] - self.last_mouse_pos[0]
                dy = current_pos[1] - self.last_mouse_pos[1]
                self.offset[0] += dx
                self.offset[1] += dy
                self.draw_image()
            self.last_mouse_pos = current_pos
        else:
            self.last_mouse_pos = None

    def zoom_handler(self, sender, app_data, user_data):
        if dpg.is_key_down(dpg.mvKey_LControl):
            scroll_direction = app_data
            zoom_step = 0.1
            new_zoom = self.zoom + (zoom_step if scroll_direction > 0 else -zoom_step)
            new_zoom = min(max(new_zoom, 0.1), 3.0)
            self.update_zoom(new_zoom)
            dpg.set_value(self.slider_tag, new_zoom)

    def draw(self):
        with dpg.theme() as transparent_theme:
            with dpg.theme_component(dpg.mvChildWindow):
                dpg.add_theme_color(dpg.mvThemeCol_ChildBg, [255, 0, 0, 200])
        with dpg.child_window(width=self.base_width, height=self.base_height, tag=self.child_container_tag, no_scrollbar=True):
            with dpg.group(horizontal=False):
                image_area_height = self.base_height - 40
                with dpg.child_window(width=self.base_width, height=image_area_height, no_scrollbar=True, tag=self.drawlist_container_tag):
                    D, _ = self.texture_dimensions
                    initial_size = int(D * self.zoom)
                    # Drawlist dinámico basado en D y zoom
                    with dpg.drawlist(width=initial_size, height=initial_size, tag=self.drawlist_tag):
                        self.draw_image()
                        dpg.bind_item_theme(self.drawlist_tag, transparent_theme)  # Asegúrate de tener 'transparent_theme' definido
                dpg.add_slider_float(label="Zoom", tag=self.slider_tag,
                                     default_value=self.zoom, min_value=0.1, max_value=3.0,
                                     width=self.base_width,
                                     callback=lambda s, a, u: self.update_zoom(a))
            with dpg.handler_registry():
                dpg.add_mouse_drag_handler(callback=self.mouse_drag_handler)
                dpg.add_mouse_wheel_handler(callback=self.zoom_handler)
