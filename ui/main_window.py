import dearpygui.dearpygui as dpg
from ui.tool_panel import ToolPanel
from ui.crop_rotate import CropRotateUI
from ui.segmentation import ImageSegmenter  # New import
import cv2
import numpy as np

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
        self.layer_masks = []  # Store masks
        
        # Box selection state
        self.box_selection_mode = False
        self.box_selection_active = False
        self.box_start = [0, 0]
        self.box_end = [0, 0]
        self.box_rect_tag = "box_selection_rect"
        self.draw_list_tag = "box_selection_draw_list"

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
                                                crop_and_rotate_ref=lambda: self.crop_rotate_ui,
                                                main_window=self)  # Pass self reference to ToolPanel
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
                # Segmentation menu options
                with dpg.menu(label="Segmentation"):
                    dpg.add_menu_item(label="Automatic Segment", callback=self.segment_current_image)
                    dpg.add_menu_item(label="Box Selection Mode", callback=self.toggle_box_selection_mode, 
                                     check=True, tag="box_selection_mode_menu")

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
                    dpg.add_mouse_down_handler(callback=self.on_mouse_down)
                    dpg.add_mouse_drag_handler(callback=self.on_mouse_drag)
                    dpg.add_mouse_release_handler(callback=self.on_mouse_release)
                    dpg.add_mouse_wheel_handler(callback=self.on_mouse_wheel)  # Add wheel handler for zoom
            
            # Set initial limits only once
            self.update_axis_limits(initial=True)
    
        # Replace draw_layer with a drawlist widget
        if not dpg.does_item_exist(self.draw_list_tag) and dpg.does_item_exist("image_plot"):
            y_axis = dpg.get_item_children(dpg.get_item_children(self.central_panel_tag, slot=1)[0], slot=1)[1]
            # Create a drawlist with a fixed size (adjust as needed)
            dpg.add_drawlist(width=600, height=400, tag=self.draw_list_tag, parent=y_axis, show=False)
            dpg.draw_rectangle([0, 0], [0, 0], color=[255, 255, 0], thickness=2, 
                               tag=self.box_rect_tag, parent=self.draw_list_tag)

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

    def toggle_box_selection_mode(self, sender, app_data):
        """Toggle the box selection mode on/off"""
        self.box_selection_mode = dpg.get_value(sender)
        print(f"Box selection mode: {self.box_selection_mode}")
        
        # Hide the drawing layer when turning off the mode
        if dpg.does_item_exist(self.draw_list_tag) and not self.box_selection_mode:
            dpg.configure_item(self.draw_list_tag, show=False)
        
        # Reset box selection state
        self.box_selection_active = False
    
    def update_box_rectangle(self):
        """Helper to update the selection rectangle by re-adding it with new coordinates."""
        if dpg.does_item_exist(self.box_rect_tag):
            dpg.delete_item(self.box_rect_tag)
        dpg.draw_rectangle(self.box_start, self.box_end, color=[255, 255, 0], thickness=2,
                           tag=self.box_rect_tag, parent=self.draw_list_tag)

    def on_mouse_down(self, sender, app_data):
        """Handle mouse down events for box selection"""
        # First check if the box selection mode is active
        if not self.box_selection_mode:
            # Forward to crop_rotate_ui if needed
            if self.crop_rotate_ui:
                return self.crop_rotate_ui.on_mouse_down(sender, app_data)
            return False
            
        # Check if clicked inside the plot area
        if dpg.is_item_hovered("image_plot") and app_data[0] == 0:  # Left button
            self.box_selection_active = True
            
            # Convert mouse coordinates to plot coordinates
            plot_pos = dpg.get_plot_mouse_pos()
            self.box_start = list(plot_pos)
            self.box_end = list(plot_pos)  # Initialize end pos to start pos
            
            # Make the box rect visible and set initial points
            if dpg.does_item_exist(self.box_rect_tag):
                # Show the drawing layer
                dpg.configure_item(self.draw_list_tag, show=True)
                self.update_box_rectangle()  # Replace modify_draw_command with helper function
            return True
        return False
    
    def on_mouse_drag(self, sender, app_data):
        """Handle mouse drag events for box selection"""
        if not self.box_selection_mode or not self.box_selection_active:
            # Forward to crop_rotate_ui if needed
            if self.crop_rotate_ui:
                return self.crop_rotate_ui.on_mouse_drag(sender, app_data)
            return False
            
        # Update box end position
        if dpg.is_item_hovered("image_plot") or self.box_selection_active:
            plot_pos = dpg.get_plot_mouse_pos()
            self.box_end = list(plot_pos)
            
            # Update rectangle
            if dpg.does_item_exist(self.box_rect_tag):
                self.update_box_rectangle()  # Update rectangle with new coordinates
            return True
        return False
    
    def on_mouse_release(self, sender, app_data):
        """Handle mouse release events for box selection"""
        if not self.box_selection_mode or not self.box_selection_active:
            # Forward to crop_rotate_ui if needed
            if self.crop_rotate_ui:
                return self.crop_rotate_ui.on_mouse_release(sender, app_data)
            return False
            
        # Finalize the box selection
        if self.box_selection_active:
            self.box_selection_active = False
            
            # Ensure box dimensions are valid
            if abs(self.box_end[0] - self.box_start[0]) > 10 and abs(self.box_end[1] - self.box_start[1]) > 10:
                # Format the box in [x1, y1, x2, y2] format
                box = [
                    min(self.box_start[0], self.box_end[0]),
                    min(self.box_start[1], self.box_end[1]),
                    max(self.box_start[0], self.box_end[0]),
                    max(self.box_start[1], self.box_end[1])
                ]
                
                # Perform segmentation with the box
                self.segment_with_box(box)
            
            # Hide the drawing layer containing the rectangle
            if dpg.does_item_exist(self.draw_list_tag):
                dpg.configure_item(self.draw_list_tag, show=False)
            return True
        return False
    
    def segment_with_box(self, box):
        """Segment the current image using a bounding box as input"""
        if self.crop_rotate_ui and hasattr(self.crop_rotate_ui, "original_image"):
            # Create segmenter on demand
            segmenter = ImageSegmenter()
            image = self.crop_rotate_ui.original_image
            
            # Get the original image dimensions
            h, w = image.shape[:2]
            
            # Scale the box coordinates to match the original image
            # (convert from plot coords to image coords)
            scaled_box = [
                max(0, min(w-1, int(box[0]))),
                max(0, min(h-1, int(box[1]))),
                max(0, min(w-1, int(box[2]))),
                max(0, min(h-1, int(box[3])))
            ]
            
            print(f"Segmenting with box: {scaled_box}")
            masks = segmenter.segment_with_box(image, scaled_box)
            self.layer_masks = masks  # Store masks for further editing
            self.tool_panel.update_masks(masks)
            self.update_mask_overlays(masks)
            print(f"Generated {len(masks)} masks using box input")
        else:
            print("No image available for segmentation.")
    
    def segment_current_image(self):
        # Segment the current image using SAM
        if self.crop_rotate_ui and hasattr(self.crop_rotate_ui, "original_image"):
            segmenter = ImageSegmenter()  # Alternatively, cache this instance
            image = self.crop_rotate_ui.original_image
            masks = segmenter.segment(image)
            self.layer_masks = masks  # Store masks for further editing
            self.tool_panel.update_masks(masks)
            self.update_mask_overlays(masks)
            print("Segmented masks:", len(masks))
        else:
            print("No image available for segmentation.")

    def update_mask_overlays(self, masks):
        # Remove existing mask overlays if any
        for idx in range(len(masks)):
            tag_series = f"mask_series_{idx}"
            if dpg.does_item_exist(tag_series):
                dpg.delete_item(tag_series)
        # Define overlay colors (RGBA) for mask layers
        colors = [(255, 0, 0, 100), (0, 255, 0, 100), (0, 0, 255, 100), (255, 255, 0, 100)]
        w = self.crop_rotate_ui.texture_w
        h = self.crop_rotate_ui.texture_h
        # Get the y_axis from the central panel (where the image series is attached)
        y_axis = dpg.get_item_children(dpg.get_item_children(self.central_panel_tag, slot=1)[0], slot=1)[1]
        
        # Create all mask overlays first
        for idx, mask in enumerate(masks):
            binary_mask = mask.get("segmentation")
            if binary_mask is None:
                continue
                
            # Resize the binary mask to match texture dimensions if needed
            binary_mask = cv2.resize(binary_mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
            overlay = np.zeros((h, w, 4), dtype=np.uint8)
            color = colors[idx % len(colors)]
            # Create a colored overlay where mask is 1
            for channel in range(4):
                overlay[..., channel] = np.where(binary_mask==1, color[channel], 0)
            texture_data = overlay.flatten().astype(np.float32) / 255.0
            texture_tag = f"mask_overlay_{idx}"
            
            # Delete texture if it already exists
            if dpg.does_item_exist(texture_tag):
                dpg.delete_item(texture_tag)
                
            # Create texture using add_raw_texture instead of add_dynamic_texture
            with dpg.texture_registry():
                dpg.add_raw_texture(w, h, texture_data, tag=texture_tag, format=dpg.mvFormat_Float_rgba)
            
            # Delete the series if it exists to avoid duplicates
            if dpg.does_item_exist(f"mask_series_{idx}"):
                dpg.delete_item(f"mask_series_{idx}")
                
            # Add the image series
            dpg.add_image_series(texture_tag,
                                bounds_min=[0, 0],
                                bounds_max=[w, h],
                                parent=y_axis,
                                tag=f"mask_series_{idx}")
            
        # After creating all overlays, show only the first mask if any exists
        if masks:
            self.show_selected_mask(0)  # Show first mask by default
        
    def show_selected_mask(self, selected_index):
        """Show only the selected mask and hide others"""
        if not hasattr(self, 'layer_masks') or not self.layer_masks:
            print("No masks available to show")
            return
            
        print(f"Showing mask {selected_index}, total masks: {len(self.layer_masks)}")
        
        # Hide all masks first
        for idx in range(len(self.layer_masks)):
            mask_tag = f"mask_series_{idx}"
            if dpg.does_item_exist(mask_tag):
                print(f"Hiding mask {idx}")
                dpg.configure_item(mask_tag, show=False)
            else:
                print(f"Mask {idx} doesn't exist")
        
        # Show only the selected mask
        selected_mask_tag = f"mask_series_{selected_index}"
        if dpg.does_item_exist(selected_mask_tag):
            print(f"Showing selected mask {selected_index}")
            dpg.configure_item(selected_mask_tag, show=True)
            
            # Make sure it's visible in the plot
            if dpg.does_item_exist("image_plot"):
                dpg.fit_axis_data("x_axis")
                dpg.fit_axis_data("y_axis")
        else:
            print(f"Selected mask {selected_index} doesn't exist")