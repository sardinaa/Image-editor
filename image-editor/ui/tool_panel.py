# ui/tool_panel.py
import dearpygui.dearpygui as dpg
flip_horizontal = False
flip_vertical = False

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
        self.curves = {"r": None, "g": None, "b": None}
        self.flip_horizontal = False
        self.flip_vertical = False
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
            # For simplicity, a button to open a separate curves editor window.
            dpg.add_button(label="Edit Curves", callback=self.open_curves_editor)
            # (The curves data will be stored in a hidden parameter or state; here we assume None by default.)
            
            dpg.add_separator()
            # Toggle Crop & Rotate panel    
            dpg.add_checkbox(label="Crop & Rotate", tag="crop_mode", default_value=False, callback=self.toggle_crop_mode)
            # This child window is hidden by default; it shows crop settings and grid mode.
            with dpg.child_window(tag="crop_panel", autosize_x=True, show=False):
                dpg.add_slider_int(label="Crop X", tag="crop_x", default_value=0, min_value=0, max_value=100, callback=self._param_changed)
                dpg.add_slider_int(label="Crop Y", tag="crop_y", default_value=0, min_value=0, max_value=100, callback=self._param_changed)
                dpg.add_slider_int(label="Crop Width", tag="crop_w", default_value=100, min_value=1, max_value=100, callback=self._param_changed)
                dpg.add_slider_int(label="Crop Height", tag="crop_h", default_value=100, min_value=1, max_value=100, callback=self._param_changed)
                dpg.add_slider_int(label="Rotation Angle", tag="rotate_angle", default_value=0, min_value=-180, max_value=180, callback=self._param_changed)
                dpg.add_button(label="Flip Horizontal", callback=self.flip_horizontal_callback)
                dpg.add_button(label="Flip Vertical", callback=self.flip_vertical_callback)
                dpg.add_button(label="Rotate Left", callback=lambda s,a,u: self.rotate_callback("left"))
                dpg.add_button(label="Rotate Right", callback=lambda s,a,u: self.rotate_callback("right"))
            
            dpg.add_separator()

            dpg.add_separator()
            dpg.add_button(label="Load Image", callback=self._load_image)
            dpg.add_button(label="Save Image", callback=self._save_image)

    def _param_changed(self, sender, app_data, user_data):
        if self.callback:
            self.callback()

    def toggle_crop_mode(self, sender, app_data, user_data):
        """Toggles the visibility of the crop panel."""
        current = dpg.get_value("crop_mode")
        if current is None:
            current = False
        new_val = current
        dpg.set_value("crop_mode", new_val)
        if not current:
            self._param_changed(sender, app_data, user_data)

        dpg.configure_item("crop_panel", show=new_val)

    def open_curves_editor(self, sender, app_data, user_data):
        # Here you could open a new window to edit curves.
        # For now, we'll just simulate by setting default curves.
        self.curves = {"r": [(0,0), (128,140), (255,255)],
                       "g": [(0,0), (128,128), (255,255)],
                       "b": [(0,0), (128,120), (255,255)]}
        self._param_changed(sender, app_data, user_data)
    
    def _param_changed(self, sender, app_data, user_data):
        if self.callback:
            self.callback()

    def flip_horizontal_callback(self, sender, app_data, user_data):
        self.flip_horizontal = not self.flip_horizontal
        self._param_changed(sender, app_data, user_data)
    
    def flip_vertical_callback(self, sender, app_data, user_data):
        self.flip_vertical = not self.flip_vertical
        self._param_changed(sender, app_data, user_data)
    
    def rotate_callback(self, direction):
        current_angle = dpg.get_value("rotate_angle")
        new_angle = current_angle - 90 if direction == "left" else current_angle + 90
        dpg.set_value("rotate_angle", new_angle)
        self._param_changed(None, None, None)

    def _load_image(self, sender, app_data, user_data):
        if self.load_callback:
            self.load_callback()

    def _save_image(self, sender, app_data, user_data):
        if self.save_callback:
            self.save_callback()

    def get_parameters(self):
        # Gather all parameters from the UI controls
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
            # Curves parameter; here we assume it's stored as None if not modified.
            'curves': self.curves,
            'crop_rect': (
                dpg.get_value("crop_x"),
                dpg.get_value("crop_y"),
                dpg.get_value("crop_w"),
                dpg.get_value("crop_h")
            ),
            'rotate_angle': dpg.get_value("rotate_angle"),
            'flip_horizontal': self.flip_horizontal,
            'flip_vertical': self.flip_vertical,
            'crop_mode': dpg.get_value("crop_mode") if dpg.does_item_exist("crop_mode") else False
        }
        return params

    
