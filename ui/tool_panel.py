import dearpygui.dearpygui as dpg

class ToolPanel:
    def __init__(self, callback, load_callback, save_callback, crop_and_rotate_ref):
        self.callback = callback
        self.load_callback = load_callback
        self.save_callback = save_callback
        self.crop_and_rotate_ref = crop_and_rotate_ref  # Función para obtener la instancia de CropAndRotate
        self.curves = {"r": None, "g": None, "b": None}
        self.panel_tag = "tool_panel"

    def draw(self):
        with dpg.child_window(tag=self.panel_tag, autosize_x=True, autosize_y=True):
            dpg.add_text("Basic Editing Tools")
            dpg.add_slider_int(label="Exposure", tag="exposure", default_value=0, min_value=-100, max_value=100, callback=self._param_changed)
            dpg.add_slider_float(label="Illumination", tag="illumination", default_value=0.0, min_value=-50.0, max_value=50.0, callback=self._param_changed)
            dpg.add_slider_float(label="Contrast", tag="contrast", default_value=1.0, min_value=0.5, max_value=3.0, callback=self._param_changed)
            dpg.add_slider_int(label="Shadow", tag="shadow", default_value=0, min_value=-100, max_value=100, callback=self._param_changed)
            dpg.add_slider_int(label="Whites", tag="whites", default_value=0, min_value=-100, max_value=100, callback=self._param_changed)
            dpg.add_slider_int(label="Blacks", tag="blacks", default_value=0, min_value=-100, max_value=100, callback=self._param_changed)
            dpg.add_slider_float(label="Saturation", tag="saturation", default_value=1.0, min_value=0.0, max_value=3.0, callback=self._param_changed)
            dpg.add_slider_int(label="Texture", tag="texture", default_value=0, min_value=0, max_value=10, callback=self._param_changed)
            dpg.add_slider_int(label="Grain", tag="grain", default_value=0, min_value=0, max_value=50, callback=self._param_changed)
            dpg.add_slider_int(label="Temperature", tag="temperature", default_value=0, min_value=-50, max_value=50, callback=self._param_changed)
            
            dpg.add_separator()
            dpg.add_text("RGB Curves")
            dpg.add_button(label="Edit Curves", callback=self.open_curves_editor)
            
            dpg.add_separator()
            dpg.add_checkbox(label="Crop & Rotate", tag="crop_mode", default_value=False, callback=self.toggle_crop_mode)
            with dpg.child_window(tag="crop_panel", autosize_x=True, show=False):
                dpg.add_slider_float(label="Ángulo de Rotación", tag="rotation_slider",
                                    min_value=0, max_value=360, default_value=0, callback=self._update_crop_rotate)
                dpg.add_button(label="Máxima Área", callback=self._set_max_rect)
                # dpg.add_button(label="Crop", callback=self._crop_image)  # Remove this line
            
            dpg.add_separator()
            dpg.add_button(label="Load Image", callback=self._load_image)
            dpg.add_button(label="Save Image", callback=self._save_image)

    def _param_changed(self, sender, app_data, user_data):
        if self.callback:
            self.callback()

    def toggle_crop_mode(self, sender, app_data, user_data):
        current = dpg.get_value("crop_mode")
        dpg.configure_item("crop_panel", show=current)
        if not current:  # Apply crop when checkbox is unchecked
            self._crop_image(sender, app_data, user_data)
        self._param_changed(sender, app_data, user_data)

    def _update_crop_rotate(self, sender, app_data, user_data):
        crop_and_rotate = self.crop_and_rotate_ref()
        if crop_and_rotate:
            crop_and_rotate.update_image(sender, app_data, user_data)
        self._param_changed(sender, app_data, user_data)

    def _set_max_rect(self, sender, app_data, user_data):
        crop_and_rotate = self.crop_and_rotate_ref()
        if crop_and_rotate:
            crop_and_rotate.set_to_max_rect(sender, app_data, user_data)
        self._param_changed(sender, app_data, user_data)

    def _crop_image(self, sender, app_data, user_data):
        crop_and_rotate = self.crop_and_rotate_ref()
        if crop_and_rotate:
            crop_and_rotate.crop_image(sender, app_data, user_data)
        self._param_changed(sender, app_data, user_data)

    def open_curves_editor(self, sender, app_data, user_data):
        self.curves = {"r": [(0,0), (128,140), (255,255)],
                       "g": [(0,0), (128,128), (255,255)],
                       "b": [(0,0), (128,120), (255,255)]}
        self._param_changed(sender, app_data, user_data)

    def _load_image(self, sender, app_data, user_data):
        if self.load_callback:
            self.load_callback()

    def _save_image(self, sender, app_data, user_data):
        if self.save_callback:
            self.save_callback()

    def get_parameters(self):
        params = {
            'exposure': dpg.get_value("exposure"),
            'illumination': dpg.get_value("illumination"),
            'contrast': dpg.get_value("contrast"),
            'shadow': dpg.get_value("shadow"),
            'whites': dpg.get_value("whites"),
            'blacks': dpg.get_value("blacks"),
            'saturation': dpg.get_value("saturation"),
            'texture': dpg.get_value("texture"),
            'grain': dpg.get_value("grain"),
            'temperature': dpg.get_value("temperature"),
            'curves': self.curves,
            'rotate_angle': dpg.get_value("rotation_slider") if dpg.does_item_exist("rotation_slider") else 0,
            'crop_mode': dpg.get_value("crop_mode") if dpg.does_item_exist("crop_mode") else False
        }
        return params