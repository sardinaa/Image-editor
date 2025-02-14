# ui/tool_panel.py
import dearpygui.dearpygui as dpg

class ToolPanel:
    def __init__(self, callback, load_callback, save_callback):
        """
        callback: Invoked when any editing parameter changes.
        load_callback: Invoked when "Load Image" is clicked.
        save_callback: Invoked when "Save Image" is clicked.
        """
        self.callback = callback
        self.load_callback = load_callback
        self.save_callback = save_callback
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
            dpg.add_button(label="Crop and Rotate", callback=self._crop_and_rotate)
            dpg.add_button(label="Adjust RGB Curves", callback=self._rgb_curves)
            dpg.add_separator()
            dpg.add_button(label="Load Image", callback=self._load_image)
            dpg.add_button(label="Save Image", callback=self._save_image)

    def _param_changed(self, sender, app_data, user_data):
        if self.callback:
            self.callback()

    def _crop_and_rotate(self, sender, app_data, user_data):
        print("Crop and Rotate functionality is not yet implemented.")

    def _rgb_curves(self, sender, app_data, user_data):
        print("RGB Curves functionality is not yet implemented.")

    def _load_image(self, sender, app_data, user_data):
        if self.load_callback:
            self.load_callback()

    def _save_image(self, sender, app_data, user_data):
        if self.save_callback:
            self.save_callback()

    def get_parameters(self):
        return {
            'exposure': dpg.get_value("exposure"),
            'illumination': dpg.get_value("illumination"),
            'contrast': dpg.get_value("contrast"),
            'shadow': dpg.get_value("shadow"),
            'whites': dpg.get_value("whites"),
            'blacks': dpg.get_value("blacks"),
            'saturation': dpg.get_value("saturation"),
            'texture': dpg.get_value("texture"),
            'grain': dpg.get_value("grain"),
            'temperature': dpg.get_value("temperature")
        }
