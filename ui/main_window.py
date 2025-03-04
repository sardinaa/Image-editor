import dearpygui.dearpygui as dpg
from ui.tool_panel import ToolPanel
from ui.crop_rotate import CropRotateUI

class MainWindow:
    def __init__(self, image, update_callback, load_callback, save_callback):
        self.image = image
        self.update_callback = update_callback
        self.load_callback = load_callback
        self.save_callback = save_callback
        self.tool_panel = None
        self.crop_rotate_ui = None
        self.window_tag = "main_window"
        self.central_panel_tag = "Central Panel"
        self.right_panel_tag = "right_panel"

    def setup(self):
        viewport_width = dpg.get_viewport_client_width()
        viewport_height = dpg.get_viewport_client_height()
        right_panel_width = int(viewport_width * 0.2)
        central_panel_width = viewport_width - right_panel_width

        with dpg.window(label="Photo Editor", tag=self.window_tag, width=viewport_width, height=viewport_height):
            with dpg.group(horizontal=True):
                # Crear el Central Panel con su contenido desde el inicio
                with dpg.child_window(tag=self.central_panel_tag, width=central_panel_width, height=viewport_height):
                    with dpg.plot(label="Image Plot", no_mouse_pos=False, height=-1, width=-1):
                        dpg.add_plot_axis(dpg.mvXAxis, label="X", no_gridlines=True, tag="x_axis")
                        dpg.add_plot_axis(dpg.mvYAxis, label="Y", no_gridlines=True, tag="y_axis")
                        # No añadimos image_series aquí; se hará dinámicamente en set_crop_rotate_ui
                with dpg.child_window(tag=self.right_panel_tag, width=right_panel_width, height=viewport_height):
                    self.tool_panel = ToolPanel(callback=self.update_callback,
                                                load_callback=self.load_callback,
                                                save_callback=self.save_callback,
                                                crop_and_rotate_ref=lambda: self.crop_rotate_ui)
                    self.tool_panel.draw()

        dpg.set_primary_window(self.window_tag, True)
        dpg.set_viewport_resize_callback(self.on_resize)

    def create_central_panel_content(self):
        # Añadir el image_series y los manejadores cuando tengamos crop_rotate_ui
        if not dpg.does_item_exist("central_image") and self.crop_rotate_ui:
            with dpg.plot(parent=self.central_panel_tag):  # Usar el plot existente
                y_axis = dpg.get_item_children(dpg.get_item_children(self.central_panel_tag, slot=1)[0], slot=1)[1]  # Obtener y_axis
                dpg.add_image_series(self.crop_rotate_ui.texture_tag,
                                     bounds_min=[0, 0],
                                     bounds_max=[self.crop_rotate_ui.texture_w, self.crop_rotate_ui.texture_h],
                                     parent=y_axis,
                                     tag="central_image")
            # Registrar manejadores de mouse solo una vez
            if not dpg.does_item_exist("mouse_handler_registry"):
                with dpg.handler_registry(tag="mouse_handler_registry"):
                    dpg.add_mouse_down_handler(callback=self.crop_rotate_ui.on_mouse_down)
                    dpg.add_mouse_drag_handler(callback=self.crop_rotate_ui.on_mouse_drag)
                    dpg.add_mouse_release_handler(callback=self.crop_rotate_ui.on_mouse_release)

    def on_resize(self, sender, app_data):
        viewport_width = dpg.get_viewport_client_width()
        viewport_height = dpg.get_viewport_client_height()
        right_panel_width = int(viewport_width * 0.2)
        central_panel_width = viewport_width - right_panel_width
        dpg.configure_item(self.central_panel_tag, width=central_panel_width, height=viewport_height)
        dpg.configure_item(self.right_panel_tag, width=right_panel_width, height=viewport_height)
        if self.crop_rotate_ui:
            self.crop_rotate_ui.update_image(None, None, None)

    def get_tool_parameters(self):
        if self.tool_panel:
            return self.tool_panel.get_parameters()
        return {}

    def set_crop_rotate_ui(self, crop_rotate_ui):
        self.crop_rotate_ui = crop_rotate_ui
        self.create_central_panel_content()  # Añadir contenido dinámicamente
        self.crop_rotate_ui.update_image(None, None, None)  # Actualizar la imagen inicial

    def update_preview(self, image, reset_offset=False):
        if self.crop_rotate_ui:
            self.crop_rotate_ui.original_image = image
            self.crop_rotate_ui.update_image(None, None, None)