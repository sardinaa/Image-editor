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
        menu_bar_height = 30  # Account for menu bar + padding
        available_height = viewport_height - menu_bar_height
        right_panel_width = int(viewport_width * 0.25)  # 25% for tools panel
        central_panel_width = viewport_width - right_panel_width

        with dpg.window(label="Photo Editor", tag=self.window_tag, width=viewport_width, height=viewport_height, no_scrollbar=True, no_move=True, no_resize=True, no_collapse=True):
            self.create_menu()
            with dpg.group(horizontal=True):
                # Central Panel - image preview
                with dpg.child_window(tag=self.central_panel_tag, width=central_panel_width, height=available_height, border=False, no_scrollbar=True):
                    pass
                # Right Panel - tools with compact layout
                with dpg.child_window(tag=self.right_panel_tag, width=right_panel_width, height=available_height, no_scrollbar=True, horizontal_scrollbar=False):
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
            # Get the actual central panel dimensions
            try:
                # Try to get actual central panel dimensions
                central_panel_width = dpg.get_item_width(self.central_panel_tag)
                central_panel_height = dpg.get_item_height(self.central_panel_tag)
                
                # Fallback to viewport calculations if panel dimensions aren't available yet
                if central_panel_width <= 0 or central_panel_height <= 0:
                    viewport_width = dpg.get_viewport_client_width()
                    viewport_height = dpg.get_viewport_client_height()
                    menu_bar_height = 30
                    central_panel_height = viewport_height - menu_bar_height
                    central_panel_width = int(viewport_width * 0.75)  # 75% of viewport (100% - 25% for tools)
            except:
                # Fallback if there are any issues
                viewport_width = dpg.get_viewport_client_width()
                viewport_height = dpg.get_viewport_client_height()
                menu_bar_height = 30
                central_panel_height = viewport_height - menu_bar_height
                central_panel_width = int(viewport_width * 0.75)
            
            # Make the plot use the full rectangular space available
            margin = 5  # Very minimal margin for plot borders
            plot_width = max(central_panel_width - margin, 200)  # Use full width
            plot_height = max(central_panel_height - margin, 200)  # Use full height
            
            # Create a container that fills the entire central panel
            with dpg.child_window(parent=self.central_panel_tag, width=-1, height=-1, 
                                border=False, no_scrollbar=True):
                # No centering spacers - let the plot fill the space
                # Create the plot to fill the entire available rectangular space
                with dpg.plot(label="Image Plot", no_mouse_pos=False, 
                              height=plot_height, width=plot_width, 
                                  anti_aliased=True,
                                  pan_button=dpg.mvMouseButton_Left,
                                  fit_button=dpg.mvMouseButton_Middle,
                                  track_offset=True,
                                  # Remove equal_aspects to allow full rectangular usage
                                  tag="image_plot"):
                        dpg.add_plot_axis(dpg.mvXAxis, label="X", no_gridlines=True, tag="x_axis")
                        dpg.add_plot_axis(dpg.mvYAxis, label="Y", no_gridlines=True, tag="y_axis")
                        y_axis = dpg.last_item()  # Get the y_axis reference
                        
                        # Calculate the actual image position within the texture
                        image_offset_x = (self.crop_rotate_ui.texture_w - self.crop_rotate_ui.orig_w) // 2
                        image_offset_y = (self.crop_rotate_ui.texture_h - self.crop_rotate_ui.orig_h) // 2
                        
                        # Create image series that shows the full texture but properly bounded
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
            # Get the y_axis from the plot
            plot_children = dpg.get_item_children("image_plot", slot=1)
            if plot_children and len(plot_children) >= 2:
                y_axis = plot_children[1]  # Second child is usually the y_axis
                # Create a drawlist with the same size as the plot
                plot_width = dpg.get_item_width("image_plot")
                plot_height = dpg.get_item_height("image_plot")
                dpg.add_drawlist(width=plot_width, height=plot_height, tag=self.draw_list_tag, parent=y_axis, show=False)
                dpg.draw_rectangle([0, 0], [0, 0], color=[255, 255, 0], thickness=2, 
                                   tag=self.box_rect_tag, parent=self.draw_list_tag)

    def update_axis_limits(self, initial=False):
        if not initial and dpg.does_item_exist("image_plot"):
            # If not initial setup, don't override user's pan/zoom
            return
            
        if not self.crop_rotate_ui:
            return
            
        # Get actual image dimensions and texture dimensions
        orig_w = self.crop_rotate_ui.orig_w
        orig_h = self.crop_rotate_ui.orig_h
        texture_w = self.crop_rotate_ui.texture_w
        texture_h = self.crop_rotate_ui.texture_h
        
        # Calculate where the image is positioned within the texture (centered)
        image_offset_x = (texture_w - orig_w) // 2
        image_offset_y = (texture_h - orig_h) // 2
        
        # Get current plot dimensions
        plot_width = dpg.get_item_width("image_plot") if dpg.does_item_exist("image_plot") else 800
        plot_height = dpg.get_item_height("image_plot") if dpg.does_item_exist("image_plot") else 600
        
        if plot_width <= 0 or plot_height <= 0:
            plot_width, plot_height = 800, 600  # Fallback dimensions
        
        # Calculate aspect ratios
        plot_aspect = plot_width / plot_height
        image_aspect = orig_w / orig_h
        
        # Calculate how to fit the actual image (not texture) within the plot while maintaining aspect ratio
        # We want to show the image area with some padding to see the whole image properly
        padding_factor = 1.05  # 5% padding around the image
        
        if image_aspect > plot_aspect:
            # Image is wider relative to plot - fit to plot width
            # Show the full image width with padding
            display_width = orig_w * padding_factor
            display_height = display_width / plot_aspect
            
            # Center the view on the actual image
            x_center = image_offset_x + orig_w / 2
            y_center = image_offset_y + orig_h / 2
            x_min = x_center - display_width / 2
            x_max = x_center + display_width / 2
            y_min = y_center - display_height / 2
            y_max = y_center + display_height / 2
        else:
            # Image is taller relative to plot - fit to plot height
            # Show the full image height with padding
            display_height = orig_h * padding_factor
            display_width = display_height * plot_aspect
            
            # Center the view on the actual image
            x_center = image_offset_x + orig_w / 2
            y_center = image_offset_y + orig_h / 2
            x_min = x_center - display_width / 2
            x_max = x_center + display_width / 2
            y_min = y_center - display_height / 2
            y_max = y_center + display_height / 2

        # Don't modify the image series bounds - keep them as the full texture
        # The texture contains the properly centered and positioned image
        
        dpg.set_axis_limits("x_axis", x_min, x_max)
        dpg.set_axis_limits("y_axis", y_min, y_max)

    def on_resize(self, sender, app_data):
        viewport_width = dpg.get_viewport_client_width()
        viewport_height = dpg.get_viewport_client_height()
        menu_bar_height = 30  # Account for menu bar height + padding
        available_height = viewport_height - menu_bar_height
        right_panel_width = int(viewport_width * 0.25)  # 25% for tools
        central_panel_width = viewport_width - right_panel_width
        
        # Update panel sizes
        dpg.configure_item(self.central_panel_tag, width=central_panel_width, height=available_height)
        dpg.configure_item(self.right_panel_tag, width=right_panel_width, height=available_height)
        
        # Update plot size to use full rectangular space and maintain image aspect ratio
        if dpg.does_item_exist("image_plot"):
            # Use actual central panel dimensions when available
            try:
                actual_central_width = dpg.get_item_width(self.central_panel_tag)
                actual_central_height = dpg.get_item_height(self.central_panel_tag)
                if actual_central_width > 0 and actual_central_height > 0:
                    central_panel_width = actual_central_width
                    available_height = actual_central_height
            except:
                pass  # Use calculated dimensions as fallback
                
            margin = 5  # Very minimal margin for more space
            plot_width = max(central_panel_width - margin, 200)  # Use full width
            plot_height = max(available_height - margin, 200)  # Use full height
            
            # Update the plot size to use full rectangular space
            dpg.configure_item("image_plot", width=plot_width, height=plot_height)
            
            # Update drawlist size to match plot size
            if dpg.does_item_exist(self.draw_list_tag):
                dpg.configure_item(self.draw_list_tag, width=plot_width, height=plot_height)
        
        # Update curves panel plot size to maintain square aspect ratio
        if self.tool_panel and self.tool_panel.curves_panel:
            self.tool_panel.curves_panel.resize_plot()
        
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

    def on_key_press(self, sender, app_data):
        """Handle keyboard events"""
        
        # Check if Delete key was pressed (key code 261)
        if app_data == 261:  # Delete key
            # Forward to curves panel if it exists and has focus
            if self.tool_panel and hasattr(self.tool_panel, 'curves_panel') and self.tool_panel.curves_panel:
                self.tool_panel.curves_panel.delete_selected_point()
                return True
        
        return False

    def on_mouse_down(self, sender, app_data):
        """Handle mouse down events for box selection"""
        # First check if the mouse is over the curves plot
        if self.tool_panel and self.tool_panel.curves_panel:
            if self.tool_panel.curves_panel.is_mouse_over_plot():
                self.tool_panel.curves_panel.on_click(sender, app_data)
                return True
        
        # Check if the box selection mode is active
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
        # First check if the mouse is over the curves plot
        if self.tool_panel and self.tool_panel.curves_panel:
            if self.tool_panel.curves_panel.is_mouse_over_plot() or self.tool_panel.curves_panel.dragging_point is not None:
                self.tool_panel.curves_panel.on_drag(sender, app_data)
                return True
        
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
        # First check if we're dealing with curves panel drag release
        if self.tool_panel and self.tool_panel.curves_panel:
            if self.tool_panel.curves_panel.dragging_point is not None:
                self.tool_panel.curves_panel.on_release(sender, app_data)
                return True
        
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
        # Get the y_axis from the plot (where the image series is attached)
        plot_children = dpg.get_item_children("image_plot", slot=1)
        if not plot_children or len(plot_children) < 2:
            print("Error: Could not find y_axis in plot")
            return
        y_axis = plot_children[1]  # Second child is usually the y_axis
        
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