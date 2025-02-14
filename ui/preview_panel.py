# ui/preview_panel.py
import uuid
import numpy as np
import cv2
import dearpygui.dearpygui as dpg

def convert_to_rgba(image):
    """
    Ensure the image has 4 channels (RGBA).
    If the image has 3 channels (RGB), add an alpha channel.
    """
    if image.shape[2] == 3:
        alpha = np.full((image.shape[0], image.shape[1], 1), 255, dtype=image.dtype)
        image = np.concatenate([image, alpha], axis=2)
    return image

class PreviewPanel:
    def __init__(self, width, height):
        self.base_width = width
        self.base_height = height
        self.zoom = 1.0
        self.offset = [0.0, 0.0]  # x, y offset for panning
        self.last_mouse_pos = None  # For manual tracking of mouse drag
        self.current_image = None  # The processed image (without zoom/pan applied)
        self.texture_tag = None
        self.texture_dimensions = None  # (width, height) of current texture
        # Tags for internal containers:
        self.child_container_tag = "preview_child"
        self.drawlist_container_tag = "preview_drawlist_container"
        self.drawlist_tag = "preview_drawlist"
        self.slider_tag = "zoom_slider"
        self._create_texture(width, height)
    
    def _create_texture(self, width, height):
        """
        Creates a dynamic texture with a unique tag and stores its dimensions.
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
        Updates the dynamic texture with new image data.
        Expects image in RGB (or RGBA) format.
        """
        self.current_image = image.copy()
        image = convert_to_rgba(image)
        image_data = (image.astype(np.float32) / 255.0).flatten().tolist()
        h, w = image.shape[:2]
        if self.texture_dimensions != (w, h):
            self._create_texture(w, h)
        dpg.set_value(self.texture_tag, image_data)

    def update_image(self, image, reset_offset=False):
        """
        Updates the preview image.
        If reset_offset is True, the pan offset is reset.
        """
        if reset_offset:
            self.offset = [0.0, 0.0]
        self.update_texture(image)
        self.draw_image()

    def update_zoom(self, zoom_value):
        """
        Updates the zoom factor and re-renders the image.
        """
        self.zoom = zoom_value
        self.draw_image()

    def draw_image(self):
        """
        Clears the drawlist and redraws the image using the current zoom and offset.
        Ensures zooming happens from the center of the image, not from the top-left corner.
        """
        if not dpg.does_item_exist(self.drawlist_tag):
            return
        dpg.delete_item(self.drawlist_tag, children_only=True)
        if self.current_image is None:
            return

        h, w = self.current_image.shape[:2]

        # Calculate zoomed dimensions
        zoomed_w = int(w * self.zoom)
        zoomed_h = int(h * self.zoom)
        slider_height = 40  # reserved for the zoom slider
        preview_w = self.base_width
        preview_h = self.base_height - slider_height
        center_x = preview_w / 2
        center_y = preview_h / 2
        # Center the image in the preview area
        new_x = center_x - zoomed_w / 2 + self.offset[0]
        new_y = center_y - zoomed_h / 2 + self.offset[1]
        
        # Draw the image
        dpg.draw_image(self.texture_tag,
                    pmin=[new_x, new_y],
                    pmax=[new_x + zoomed_w, new_y + zoomed_h],
                    parent=self.drawlist_tag)
            
        # --- Draw the crop grid overlay ---
        # Get crop parameters from the tool panel (assume they are set)
        if dpg.get_value("crop_mode"):
            crop_x = dpg.get_value("crop_x")
            crop_y = dpg.get_value("crop_y")
            crop_w = dpg.get_value("crop_w")
            crop_h = dpg.get_value("crop_h")
            # Scale crop rectangle according to zoom factor
            drawn_crop_x = new_x + crop_x * self.zoom
            drawn_crop_y = new_y + crop_y * self.zoom
            drawn_crop_w = crop_w * self.zoom
            drawn_crop_h = crop_h * self.zoom
            
            # Draw a semi-transparent rectangle for the crop area
            dpg.draw_rectangle(pmin=[drawn_crop_x, drawn_crop_y],
                            pmax=[drawn_crop_x + drawn_crop_w, drawn_crop_y + drawn_crop_h],
                            color=[0,255,0,255], thickness=2, fill=[0,255,0,50],
                            parent=self.drawlist_tag)

    def mouse_drag_handler(self, sender, app_data, user_data):
        """
        Handles mouse drag events.
        When Ctrl is held, calculates movement based on the current and last mouse positions.
        """
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
        """
        Handles zooming in and out with Ctrl + Mouse Wheel.
        Zooming is applied from the center of the preview area.
        The new zoom level respects the slider's boundaries and updates the slider value.
        """
        if dpg.is_key_down(dpg.mvKey_LControl):
            # Scroll direction: positive for up (zoom in), negative for down (zoom out)
            scroll_direction = app_data  
            zoom_step = 0.1  # Adjust zoom speed as needed

            # Calculate new zoom level within allowed limits (min: 0.1, max: 3.0)
            new_zoom = self.zoom + (zoom_step if scroll_direction > 0 else -zoom_step)
            new_zoom = min(max(new_zoom, 0.1), 3.0)  # Clamp between 0.1 and 3.0

            # Update the zoom value and the slider
            self.zoom = new_zoom
            dpg.set_value(self.slider_tag, new_zoom)
            self.update_zoom(new_zoom)


    def draw(self):
        """
        Draws the preview panel:
        - A child window (with no scrollbars) that holds a vertical group.
        - The vertical group contains:
            • A child window for the image (drawlist container) occupying all available space except a fixed height for the slider.
            • A zoom slider at the bottom.
        - Mouse drag and wheel handlers are attached.
        """
        with dpg.child_window(width=self.base_width, height=self.base_height, tag=self.child_container_tag, no_scrollbar=True):
            with dpg.group(horizontal=False):
                # Reserve a fixed height for the slider (e.g., 40 pixels)
                image_area_height = self.base_height - 40
                with dpg.child_window(width=self.base_width, height=image_area_height, no_scrollbar=True, tag=self.drawlist_container_tag):
                    with dpg.drawlist(width=self.base_width, height=image_area_height, tag=self.drawlist_tag):
                        self.draw_image()
                # Add the zoom slider (make it exactly the full width)
                dpg.add_slider_float(label="Zoom", tag=self.slider_tag,
                                    default_value=self.zoom, min_value=0.1, max_value=3.0,
                                    width=self.base_width,
                                    callback=lambda s, a, u: self.update_zoom(a))
            # Attach mouse event handlers for panning and zooming.
            with dpg.handler_registry():
                dpg.add_mouse_drag_handler(callback=self.mouse_drag_handler)
                dpg.add_mouse_wheel_handler(callback=self.zoom_handler)


    def set_size(self, new_width, new_height):
        """
        Updates the preview panel's size and reconfigures internal containers.
        """
        self.base_width = new_width
        self.base_height = new_height
        # Update the outer container size.
        dpg.configure_item(self.child_container_tag, width=new_width, height=new_height)
        # Reserve 40 pixels for the slider.
        image_area_height = new_height - 100
        if dpg.does_item_exist(self.drawlist_container_tag):
            dpg.configure_item(self.drawlist_container_tag, width=new_width, height=image_area_height)
        if dpg.does_item_exist(self.drawlist_tag):
            dpg.configure_item(self.drawlist_tag, width=new_width, height=image_area_height)
        if dpg.does_item_exist(self.slider_tag):
            dpg.configure_item(self.slider_tag, width=new_width)
        self.draw_image()

