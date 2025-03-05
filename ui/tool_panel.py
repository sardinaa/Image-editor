import dearpygui.dearpygui as dpg
from ui.curves_panel import CurvesPanel

class ToolPanel:
    def __init__(self, callback, crop_and_rotate_ref):
        self.callback = callback
        self.crop_and_rotate_ref = crop_and_rotate_ref  # Función para obtener la instancia de CropAndRotate
        self.curves = {"r": [(0, 0), (128, 128), (255, 255)],
                      "g": [(0, 0), (128, 128), (255, 255)],
                      "b": [(0, 0), (128, 128), (255, 255)]}
        self.panel_tag = "tool_panel"
        self.curves_panel = None

    def draw(self):
        with dpg.group(horizontal=False):
            dpg.add_text("Basic Editing Tools", color=[176, 204, 255])
            dpg.add_separator()
            dpg.add_spacer(height=5)
            dpg.add_checkbox(label="Crop & Rotate", tag="crop_mode", default_value=False, callback=self.toggle_crop_mode)
            with dpg.child_window(tag="crop_panel", height=100, autosize_x=True, show=False, border=True):
                dpg.add_slider_float(label="Rotation Angle", tag="rotation_slider",
                                    min_value=0, max_value=360, default_value=0, callback=self._update_crop_rotate)
                dpg.add_button(label="Maximum Area", callback=self._set_max_rect, width=-1)

            dpg.add_separator()
            
            # Add some spacing for better readability
            dpg.add_spacer(height=5)
            
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
            dpg.add_spacer(height=5)
            dpg.add_text("RGB Curves", color=[176, 204, 255])
            
            # Embed the curves panel directly in the tool panel
            self.curves_panel = CurvesPanel(callback=self._param_changed)
            self.curves_panel.curves = self.curves
            self.curves_panel.show()
            
            # New Masks section
            dpg.add_separator()
            dpg.add_text("Masks", color=[176, 204, 255])
            dpg.add_listbox(items=[], tag="mask_list", callback=self._mask_selected, num_items=4)

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
            'curves': self.curves_panel.get_curves() if self.curves_panel else self.curves,
            'rotate_angle': dpg.get_value("rotation_slider") if dpg.does_item_exist("rotation_slider") else 0,
            'crop_mode': dpg.get_value("crop_mode") if dpg.does_item_exist("crop_mode") else False
        }
        return params

    def _mask_selected(self, sender, app_data, user_data):
        # Placeholder callback when a mask is selected from the listbox.
        print("Selected mask:", app_data)
        
    def update_masks(self, masks):
        # Create listbox entries based on the number of masks
        items = [f"Mask {idx+1}" for idx in range(len(masks))]
        if dpg.does_item_exist("mask_list"):
            dpg.configure_item("mask_list", items=items)
        else:
            print("Mask list widget does not exist.")