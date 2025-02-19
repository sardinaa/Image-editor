# ui/main_window.py
import dearpygui.dearpygui as dpg
from ui.preview_panel import PreviewPanel
from ui.tool_panel import ToolPanel

class MainWindow:
    def __init__(self, image, update_callback, load_callback, save_callback):
        self.image = image
        self.update_callback = update_callback
        self.load_callback = load_callback
        self.save_callback = save_callback
        self.preview_panel = None
        self.tool_panel = None
        self.window_tag = "main_window"
        self.left_panel_tag = "left_panel"
        self.right_panel_tag = "right_panel"

    def setup(self):
        viewport_width = dpg.get_viewport_client_width()
        viewport_height = dpg.get_viewport_client_height()
        right_panel_width = int(viewport_width * 0.2)
        left_panel_width = viewport_width - right_panel_width

        with dpg.window(label="Photo Editor", tag=self.window_tag, width=viewport_width, height=viewport_height):
            with dpg.group(horizontal=True):
                with dpg.child_window(tag=self.left_panel_tag, width=left_panel_width, height=viewport_height):
                    self.preview_panel = PreviewPanel(left_panel_width, viewport_height)
                    self.preview_panel.draw()
                with dpg.child_window(tag=self.right_panel_tag, width=right_panel_width, height=viewport_height):
                    self.tool_panel = ToolPanel(callback=self.update_callback,
                                                load_callback=self.load_callback,
                                                save_callback=self.save_callback)
                    self.tool_panel.draw()
        dpg.set_primary_window(self.window_tag, True)
        dpg.set_viewport_resize_callback(self.on_resize)

    def on_resize(self, sender, app_data):
        viewport_width = dpg.get_viewport_client_width()
        viewport_height = dpg.get_viewport_client_height()
        right_panel_width = int(viewport_width * 0.2)
        left_panel_width = viewport_width - right_panel_width
        dpg.configure_item(self.left_panel_tag, width=left_panel_width, height=viewport_height)
        dpg.configure_item(self.right_panel_tag, width=right_panel_width, height=viewport_height)
        if self.preview_panel:
            self.preview_panel.set_size(left_panel_width, viewport_height)

    def get_tool_parameters(self):
        if self.tool_panel:
            return self.tool_panel.get_parameters()
        return {}

    def update_preview(self, image, reset_offset=False):
        if self.preview_panel:
            self.preview_panel.update_image(image, reset_offset)
