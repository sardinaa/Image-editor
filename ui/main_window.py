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

        with dpg.window(label="Photo Editor", tag=self.window_tag, width=viewport_width, height=viewport_height, no_scrollbar=True, no_move=True, no_resize=True):
            self.create_menu()
            with dpg.group(horizontal=True):
                # Disable scrollbars in the Central Panel
                with dpg.child_window(tag=self.central_panel_tag, width=central_panel_width, height=viewport_height, border=False, no_scrollbar=True):
                    pass
                # Disable scrollbars in the Right Panel
                with dpg.child_window(tag=self.right_panel_tag, width=right_panel_width, height=viewport_height, no_scrollbar=True):
                    self.tool_panel = ToolPanel(callback=self.update_callback,
                                                crop_and_rotate_ref=lambda: self.crop_rotate_ui)
                    self.tool_panel.draw()

        dpg.set_primary_window(self.window_tag, True)
        dpg.set_viewport_resize_callback(self.on_resize)

    def create_menu(self):
        with dpg.menu_bar():
            with dpg.menu(label="Archivo"):
                dpg.add_menu_item(label="Load", callback=self.load_callback)
                dpg.add_menu_item(label="Save", callback=self.save_callback)
                dpg.add_menu_item(label="Exit", callback=lambda: dpg.stop_dearpygui())
            with dpg.menu(label="Editar"):
                dpg.add_menu_item(label="Undo", callback=lambda: print("Deshacer"))
                dpg.add_menu_item(label="Redo", callback=lambda: print("Rehacer"))

    def create_central_panel_content(self):
        # Añadir el image_series y los manejadores cuando tengamos crop_rotate_ui
        if not dpg.does_item_exist("central_image") and self.crop_rotate_ui:
            with dpg.plot(label="Image Plot", no_mouse_pos=False, height=-1, width=-1, 
                          parent=self.central_panel_tag, 
                          anti_aliased=True,
                          pan_button=dpg.mvMouseButton_Left,  # Enable panning with left mouse button
                          fit_button=dpg.mvMouseButton_Middle,  # Fit view with middle mouse button
                          track_offset=True,  # Allow tracking offset for panning
                          tag="image_plot"):
                dpg.add_plot_axis(dpg.mvXAxis, label="X", no_gridlines=True, tag="x_axis")
                dpg.add_plot_axis(dpg.mvYAxis, label="Y", no_gridlines=True, tag="y_axis")
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
                    dpg.add_mouse_wheel_handler(callback=self.on_mouse_wheel)  # Add wheel handler for zoom
            
            # Set initial limits only once
            self.update_axis_limits(initial=True)

    def update_axis_limits(self, initial=False):
        if not initial and dpg.does_item_exist("image_plot"):
            # If not initial setup, don't override user's pan/zoom
            return
            
        viewport_width = dpg.get_viewport_client_width()
        viewport_height = dpg.get_viewport_client_height()
        panel_w, panel_h = dpg.get_item_rect_size("Central Panel")
        if panel_w <= 0 or panel_h <= 0:
            panel_w, panel_h = dpg.get_item_width("Central Panel"), dpg.get_item_height("Central Panel")
        plot_aspect = panel_w / panel_h
        texture_aspect = self.crop_rotate_ui.texture_w / self.crop_rotate_ui.texture_h
        
        # Center the texture in the plot
        if plot_aspect > texture_aspect:
            # Plot is wider than the texture - center horizontally
            display_width = self.crop_rotate_ui.texture_h * plot_aspect
            x_center = self.crop_rotate_ui.texture_w / 2
            x_min = x_center - display_width / 2
            x_max = x_center + display_width / 2
            y_min = 0
            y_max = self.crop_rotate_ui.texture_h
        else:
            # Plot is taller than the texture - center vertically
            display_height = self.crop_rotate_ui.texture_w / plot_aspect
            y_center = self.crop_rotate_ui.texture_h / 2
            y_min = y_center - display_height / 2
            y_max = y_center + display_height / 2
            x_min = 0
            x_max = self.crop_rotate_ui.texture_w

        if dpg.does_item_exist("central_image"):
            dpg.configure_item("central_image", bounds_min=[0, 0], bounds_max=[self.crop_rotate_ui.texture_w, self.crop_rotate_ui.texture_h])
        else:
            print("Error: 'central_image' does not exist.")
            
        dpg.set_axis_limits("x_axis", x_min, x_max)
        dpg.set_axis_limits("y_axis", y_min, y_max)

    def on_resize(self, sender, app_data):
        viewport_width = dpg.get_viewport_client_width()
        viewport_height = dpg.get_viewport_client_height()
        right_panel_width = int(viewport_width * 0.2)
        central_panel_width = viewport_width - right_panel_width
        dpg.configure_item(self.central_panel_tag, width=central_panel_width, height=viewport_height)
        dpg.configure_item(self.right_panel_tag, width=right_panel_width, height=viewport_height)
        if self.crop_rotate_ui:
            self.crop_rotate_ui.update_image(None, None, None)
            # On resize, we want to reset the view
            self.update_axis_limits(initial=True)

    def on_mouse_wheel(self, sender, app_data):
        # Custom zoom handling
        if dpg.does_item_exist("image_plot") and self.crop_rotate_ui:
            # Only handle zoom if we're not in crop mode
            crop_mode = dpg.get_value("crop_mode") if dpg.does_item_exist("crop_mode") else False
            if not crop_mode:
                # Get current plot limits
                x_limits = dpg.get_axis_limits("x_axis")
                y_limits = dpg.get_axis_limits("y_axis")
                
                # Calculate zoom factor based on wheel direction
                zoom_factor = 0.9 if app_data > 0 else 1.1
                
                # Calculate new limits
                x_range = x_limits[1] - x_limits[0]
                y_range = y_limits[1] - y_limits[0]
                x_center = (x_limits[0] + x_limits[1]) / 2
                y_center = (y_limits[0] + y_limits[1]) / 2
                
                new_x_range = x_range * zoom_factor
                new_y_range = y_range * zoom_factor
                
                # Set new limits
                dpg.set_axis_limits("x_axis", x_center - new_x_range/2, x_center + new_x_range/2)
                dpg.set_axis_limits("y_axis", y_center - new_y_range/2, y_center + new_y_range/2)
                
                return True  # Consume the event
        return False

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