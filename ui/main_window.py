import dearpygui.dearpygui as dpg
from ui.tool_panel import ToolPanel
from ui.crop_rotate import CropRotateUI
from ui.segmentation import ImageSegmenter  # New import
from ui.bounding_box_renderer import BoundingBoxRenderer, BoundingBox  # Add BoundingBox import
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
        self.mask_names = []  # Store custom names for masks
        
        # Box selection state
        self.box_selection_mode = False
        self.box_selection_active = False
        self.box_start = [0, 0]
        self.box_end = [0, 0]
        self.box_rect_tag = "box_selection_rect"
        self.draw_list_tag = "box_selection_draw_list"
        
        # Segmentation instance - created once to avoid GPU memory issues
        self.segmenter = None
        
        # Segmentation bounding box renderer
        self.segmentation_bbox_renderer = None
        self.segmentation_mode = False
        self.segmentation_texture = None
        self.pending_segmentation_box = None

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
        # Custom zoom handling - only when mouse is over the image plot
        if dpg.does_item_exist("image_plot") and self.crop_rotate_ui and dpg.is_item_hovered("image_plot"):
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
    
    def get_segmenter(self):
        """Get or create the segmenter instance. Created once to avoid GPU memory issues."""
        if self.segmenter is None:
            try:
                print("Creating ImageSegmenter instance...")
                # Use auto device selection for better memory management
                self.segmenter = ImageSegmenter(device="auto")
                print("ImageSegmenter instance created successfully")
            except Exception as e:
                print(f"Error creating ImageSegmenter: {e}")
                # Try fallback to CPU if GPU fails
                try:
                    print("Attempting to create ImageSegmenter with CPU...")
                    self.segmenter = ImageSegmenter(device="cpu")
                    print("ImageSegmenter instance created successfully on CPU")
                except Exception as cpu_e:
                    print(f"Error creating ImageSegmenter on CPU: {cpu_e}")
                    return None
        return self.segmenter

    def set_crop_rotate_ui(self, crop_rotate_ui):
        self.crop_rotate_ui = crop_rotate_ui
        self.create_central_panel_content()  # Añadir contenido dinámicamente
        self.crop_rotate_ui.update_image(None, None, None)  # Actualizar la imagen inicial
        
        # Initialize segmentation bounding box renderer
        self._initialize_segmentation_bbox_renderer()
    
    def _initialize_segmentation_bbox_renderer(self):
        """Initialize the segmentation bounding box renderer"""
        if not self.crop_rotate_ui:
            return
            
        self.segmentation_bbox_renderer = BoundingBoxRenderer(
            texture_width=self.crop_rotate_ui.texture_w,
            texture_height=self.crop_rotate_ui.texture_h,
            panel_id="Central Panel",
            min_size=20,
            handle_size=20,
            handle_threshold=50
        )
        
        # Set visual style to match crop mode for consistency
        self.segmentation_bbox_renderer.set_visual_style(
            box_color=(64, 64, 64, 255),  # Dark gray box (same as crop mode)
            handle_color=(13, 115, 184, 255),  # Blue handles (same as crop mode)
            box_thickness=2  # Same thickness as crop mode
        )
        
        # Set up callbacks
        self.segmentation_bbox_renderer.set_callbacks(
            on_change=self._on_segmentation_bbox_change,
            on_start_drag=self._on_segmentation_bbox_start_drag,
            on_end_drag=self._on_segmentation_bbox_end_drag
        )
    
    def _on_segmentation_bbox_change(self, bbox: BoundingBox) -> None:
        """Called when segmentation bounding box changes during drag."""
        if self.segmentation_mode and self.segmentation_texture is not None:
            self._update_segmentation_overlay()
    
    def _on_segmentation_bbox_start_drag(self, bbox: BoundingBox) -> None:
        """Called when segmentation bounding box drag starts."""
        pass  # No special handling needed for start
    
    def _on_segmentation_bbox_end_drag(self, bbox: BoundingBox) -> None:
        """Called when segmentation bounding box drag ends."""
        if self.segmentation_mode and self.segmentation_texture is not None:
            self.pending_segmentation_box = bbox.to_dict()
            self._update_segmentation_overlay(force_update=True)
    
    def _update_segmentation_overlay(self, force_update=False):
        """Update the segmentation overlay with the current bounding box"""
        if not self.segmentation_bbox_renderer or self.segmentation_texture is None:
            return
            
        # Render the bounding box on the texture
        blended = self.segmentation_bbox_renderer.render_on_texture(self.segmentation_texture)
        
        # Update the texture in DearPyGUI
        if self.crop_rotate_ui and self.crop_rotate_ui.texture_tag:
            texture_data = blended.flatten().astype(np.float32) / 255.0
            dpg.set_value(self.crop_rotate_ui.texture_tag, texture_data)

    def update_preview(self, image, reset_offset=False):
        if self.crop_rotate_ui:
            self.crop_rotate_ui.original_image = image
            self.crop_rotate_ui.update_image(None, None, None)

    def toggle_box_selection_mode(self, sender, app_data):
        """Toggle the box selection mode on/off"""
        # Get the current state - could be from either the tool panel or sender
        if sender and dpg.does_item_exist(sender):
            self.box_selection_mode = dpg.get_value(sender)
        elif dpg.does_item_exist("box_selection_mode"):
            # If called from tool panel, get value from tool panel checkbox
            self.box_selection_mode = dpg.get_value("box_selection_mode")
        else:
            # Fallback - toggle current state
            self.box_selection_mode = not self.box_selection_mode
        
        # Sync both checkboxes to the same state
        if dpg.does_item_exist("box_selection_mode"):
            dpg.set_value("box_selection_mode", self.box_selection_mode)
        
        print(f"Box selection mode: {self.box_selection_mode} (always accumulates masks)")
        
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
        
        # Check if segmentation mode is active
        if self.segmentation_mode and self.segmentation_bbox_renderer:
            if dpg.is_item_hovered("image_plot"):
                if self.segmentation_bbox_renderer.on_mouse_down(sender, app_data):
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
        
        # Check if segmentation mode is active
        if self.segmentation_mode and self.segmentation_bbox_renderer:
            if self.segmentation_bbox_renderer.on_mouse_drag(sender, app_data):
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
        
        # Check if segmentation mode is active
        if self.segmentation_mode and self.segmentation_bbox_renderer:
            if self.segmentation_bbox_renderer.on_mouse_release(sender, app_data):
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
        """
        Segment the current image using a bounding box as input.
        
        Args:
            box: Bounding box coordinates [x1, y1, x2, y2] in texture coordinate system
                 (Y=0 at top, Y increases downward) from BoundingBoxRenderer
        """
        if self.crop_rotate_ui and hasattr(self.crop_rotate_ui, "original_image"):
            # Get the reusable segmenter instance
            segmenter = self.get_segmenter()
            if segmenter is None:
                print("Failed to create segmenter instance")
                return
                
            image = self.crop_rotate_ui.original_image
            
            # Get the original image dimensions
            h, w = image.shape[:2]
            
            # Calculate the texture offsets where the image is centered within the texture
            texture_w = self.crop_rotate_ui.texture_w
            texture_h = self.crop_rotate_ui.texture_h
            offset_x = (texture_w - w) // 2
            offset_y = (texture_h - h) // 2
            
            # The box coordinates are already in texture coordinate system (from BoundingBoxRenderer)
            # We just need to adjust for the texture offset to get image coordinates
            # Note: No Y-axis flipping needed since both are in texture/image coordinate system
            scaled_box = [
                max(0, min(w-1, int(box[0] - offset_x))),     # X1: Adjust X for texture offset
                max(0, min(h-1, int(box[1] - offset_y))),     # Y1: Adjust Y for texture offset
                max(0, min(w-1, int(box[2] - offset_x))),     # X2: Adjust X for texture offset
                max(0, min(h-1, int(box[3] - offset_y)))      # Y2: Adjust Y for texture offset
            ]
            
            print(f"Original box (texture coords): {box}")
            print(f"Texture dimensions: {texture_w}x{texture_h}")
            print(f"Image dimensions: {w}x{h}")
            print(f"Image offset in texture: ({offset_x}, {offset_y})")
            print(f"Segmenting with box (image coords): {scaled_box}")
            try:
                # Show loading indicator
                self._show_segmentation_loading("Segmenting selected area...")
                
                # Clean up memory before segmentation
                self.cleanup_segmenter_memory()
                
                new_masks = segmenter.segment_with_box(image, scaled_box)
                
                # Initialize layer_masks if it doesn't exist
                if not hasattr(self, 'layer_masks') or self.layer_masks is None:
                    self.layer_masks = []
                
                # Always accumulate new masks with existing ones
                start_index = len(self.layer_masks)
                self.layer_masks.extend(new_masks)
                
                # Add names for new masks
                if not hasattr(self, 'mask_names'):
                    self.mask_names = []
                # Ensure we have names for all existing masks first
                while len(self.mask_names) < start_index:
                    self.mask_names.append(f"Mask {len(self.mask_names) + 1}")
                # Add names for new masks
                for idx in range(len(new_masks)):
                    self.mask_names.append(f"Mask {start_index + idx + 1}")
                
                print(f"Generated {len(new_masks)} new masks, total masks: {len(self.layer_masks)}")
                
                # Update UI with all masks
                self.tool_panel.update_masks(self.layer_masks, self.mask_names)
                self.update_mask_overlays(self.layer_masks)
            except Exception as e:
                print(f"Error during box segmentation: {e}")
                # On error, try to clean up memory and potentially reset segmenter
                try:
                    self.cleanup_segmenter_memory()
                    # If it's a memory error, suggest resetting the segmenter
                    if "memory" in str(e).lower() or "cuda" in str(e).lower():
                        print("Memory-related error detected. Consider resetting segmenter for next use.")
                        self.reset_segmenter()
                except Exception as cleanup_e:
                    print(f"Error during cleanup: {cleanup_e}")
            finally:
                # Always hide loading indicator
                self._hide_segmentation_loading()
        else:
            print("No image available for segmentation.")
    
    def segment_current_image(self):
        # Segment the current image using SAM
        if self.crop_rotate_ui and hasattr(self.crop_rotate_ui, "original_image"):
            # Show loading indicator
            self._show_segmentation_loading("Segmenting image...")
            
            # Get the reusable segmenter instance
            segmenter = self.get_segmenter()
            if segmenter is None:
                print("Failed to create segmenter instance")
                self._hide_segmentation_loading()
                return
                
            image = self.crop_rotate_ui.original_image
            try:
                # Clean up memory before segmentation
                self.cleanup_segmenter_memory()
                
                masks = segmenter.segment(image)
                # For automatic segmentation, replace all existing masks
                self.layer_masks = masks  # Store masks for further editing
                
                # Reset mask names for automatic segmentation
                self.mask_names = [f"Mask {idx + 1}" for idx in range(len(masks))]
                
                self.tool_panel.update_masks(masks, self.mask_names)
                self.update_mask_overlays(masks)
                print(f"Automatic segmentation completed with {len(masks)} masks (replaced all previous masks)")
            except Exception as e:
                print(f"Error during automatic segmentation: {e}")
                # On error, try to clean up memory and potentially reset segmenter
                try:
                    self.cleanup_segmenter_memory()
                    # If it's a memory error, suggest resetting the segmenter
                    if "memory" in str(e).lower() or "cuda" in str(e).lower():
                        print("Memory-related error detected. Consider resetting segmenter for next use.")
                        self.reset_segmenter()
                except Exception as cleanup_e:
                    print(f"Error during cleanup: {cleanup_e}")
            finally:
                # Always hide loading indicator
                self._hide_segmentation_loading()
        else:
            print("No image available for segmentation.")

    def update_mask_overlays(self, masks):
        print(f"Updating mask overlays for {len(masks)} masks")
        
        # Check if we have valid mask data
        if not masks:
            print("No masks to display")
            return
            
        # Get basic dimensions and plot info
        w = self.crop_rotate_ui.texture_w
        h = self.crop_rotate_ui.texture_h
        original_image = self.crop_rotate_ui.original_image
        orig_h, orig_w = original_image.shape[:2]
        
        # Get the y_axis from the plot 
        plot_children = dpg.get_item_children("image_plot", slot=1)
        if not plot_children or len(plot_children) < 2:
            print("Error: Could not find y_axis in plot")
            return
        y_axis = plot_children[1]
        
        print(f"Texture dimensions: {w}x{h}")
        print(f"Original image dimensions: {orig_w}x{orig_h}")
        
        # Define overlay colors (RGBA) for mask layers - cycle through colors if more masks than colors
        colors = [(255, 0, 0, 100), (0, 255, 0, 100), (0, 0, 255, 100), (255, 255, 0, 100),
                 (255, 0, 255, 100), (0, 255, 255, 100), (128, 255, 0, 100), (255, 128, 0, 100),
                 (255, 128, 128, 100), (128, 255, 128, 100), (128, 128, 255, 100), (255, 255, 128, 100),
                 (255, 128, 255, 100), (128, 255, 255, 100), (192, 64, 0, 100), (0, 192, 64, 100)]
        
        # Process all available masks (no artificial limit)
        max_masks = len(masks)  # Support unlimited masks
        
        # Calculate offset once
        offset_x = (w - orig_w) // 2
        offset_y = (h - orig_h) // 2
        
        # Create a unique timestamp for this batch of masks to avoid tag conflicts
        import time
        timestamp = int(time.time() * 1000000)  # Microsecond timestamp for uniqueness
        
        # Remove existing series (but not textures) to avoid conflicts
        # Clean up old series - support up to 100 masks for large accumulations
        for idx in range(100):
            old_series_tag = f"mask_series_{idx}"
            if dpg.does_item_exist(old_series_tag):
                try:
                    dpg.delete_item(old_series_tag)
                    print(f"Deleted old series {old_series_tag}")
                except Exception as e:
                    print(f"Error deleting old series {old_series_tag}: {e}")
        
        # Create mask overlays with unique texture tags to avoid conflicts
        successful_masks = 0
        for idx in range(max_masks):
            try:
                mask = masks[idx]
                binary_mask = mask.get("segmentation")
                if binary_mask is None:
                    print(f"Skipping mask {idx} - no segmentation data")
                    continue
                
                print(f"Processing mask {idx} with shape: {binary_mask.shape}")
                
                # Create overlay texture
                overlay = np.zeros((h, w, 4), dtype=np.uint8)
                color = colors[idx % len(colors)]
                
                # Apply mask with color
                for channel in range(4):
                    overlay[offset_y:offset_y + orig_h, offset_x:offset_x + orig_w, channel] = np.where(binary_mask == 1, color[channel], 0)
                
                # Convert to texture format
                texture_data = overlay.flatten().astype(np.float32) / 255.0
                
                # Use unique texture tag with timestamp to avoid conflicts
                texture_tag = f"mask_overlay_{idx}_{timestamp}"
                series_tag = f"mask_series_{idx}"  # Keep series tag simple for easier management
                
                # Create new texture with unique tag
                with dpg.texture_registry():
                    dpg.add_raw_texture(w, h, texture_data, tag=texture_tag, format=dpg.mvFormat_Float_rgba)
                
                # Create new image series
                dpg.add_image_series(texture_tag,
                                    bounds_min=[0, 0],
                                    bounds_max=[w, h],
                                    parent=y_axis,
                                    tag=series_tag,
                                    show=False)  # Start hidden
                
                successful_masks += 1
                print(f"Successfully created mask overlay {idx} with texture {texture_tag}")
                
            except Exception as e:
                print(f"Error creating mask overlay {idx}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Hide any unused existing overlays (beyond current mask count)
        # Check up to 100 potential existing overlays to handle large numbers of accumulated masks
        for idx in range(max_masks, 100):
            series_tag = f"mask_series_{idx}"
            if dpg.does_item_exist(series_tag):
                try:
                    dpg.configure_item(series_tag, show=False)
                    print(f"Hid unused overlay {idx}")
                except Exception as e:
                    print(f"Error hiding overlay {idx}: {e}")
        
        # Show first mask if any were created successfully and masks should be visible
        if successful_masks > 0:
            try:
                # Only show masks if:
                # 1. Masks are enabled
                # 2. Crop mode is not active
                # 3. Show mask overlay is enabled
                masks_enabled = dpg.does_item_exist("mask_section_toggle") and dpg.get_value("mask_section_toggle")
                crop_mode_active = dpg.does_item_exist("crop_mode") and dpg.get_value("crop_mode")
                show_overlay = dpg.does_item_exist("show_mask_overlay") and dpg.get_value("show_mask_overlay")
                
                if masks_enabled and not crop_mode_active and show_overlay:
                    self.show_selected_mask(0)
                    print(f"Successfully processed {successful_masks} mask overlays and showing first mask")
                else:
                    print(f"Successfully processed {successful_masks} mask overlays but not showing (masks_enabled={masks_enabled}, crop_mode_active={crop_mode_active}, show_overlay={show_overlay})")
            except Exception as e:
                print(f"Error showing selected mask: {e}")
        else:
            print("No mask overlays were processed")
    
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
    
    def clear_all_masks(self):
        """Clear all accumulated masks and reset the segmentation state"""
        print("Clearing all accumulated masks")
        
        # Clear the stored masks
        self.layer_masks = []
        self.mask_names = []
        
        # Hide and clean up all mask overlays
        # Clean up all possible mask series - support up to 100 masks
        for idx in range(100):
            series_tag = f"mask_series_{idx}"
            if dpg.does_item_exist(series_tag):
                try:
                    dpg.configure_item(series_tag, show=False)
                    dpg.delete_item(series_tag)
                    print(f"Deleted mask overlay {idx}")
                except Exception as e:
                    print(f"Error deleting mask overlay {idx}: {e}")
        
        # Clear the mask list in the tool panel
        if self.tool_panel:
            self.tool_panel.update_masks([], [])
        
        print("All masks cleared successfully")
    
    def delete_mask(self, mask_index):
        """Delete a specific mask by index"""
        if not hasattr(self, 'layer_masks') or not self.layer_masks:
            print("No masks available to delete")
            return
            
        if mask_index < 0 or mask_index >= len(self.layer_masks):
            print(f"Invalid mask index: {mask_index}")
            return
        
        print(f"Deleting mask at index {mask_index}")
        
        # Remove the mask and its name
        del self.layer_masks[mask_index]
        if hasattr(self, 'mask_names') and mask_index < len(self.mask_names):
            del self.mask_names[mask_index]
        
        # Clean up the visual representation
        self._cleanup_mask_overlay(mask_index)
        
        # Update the UI components
        self._update_ui_after_mask_change()
        
        print(f"Successfully deleted mask at index {mask_index}, remaining masks: {len(self.layer_masks)}")
    
    def rename_mask(self, mask_index, new_name):
        """Rename a specific mask by index"""
        if not hasattr(self, 'layer_masks') or not self.layer_masks:
            print("No masks available to rename")
            return
            
        if mask_index < 0 or mask_index >= len(self.layer_masks):
            print(f"Invalid mask index: {mask_index}")
            return
        
        # Ensure mask_names list exists and is properly sized
        if not hasattr(self, 'mask_names'):
            self.mask_names = []
        
        # Extend mask_names list if needed
        while len(self.mask_names) <= mask_index:
            self.mask_names.append(f"Mask {len(self.mask_names) + 1}")
        
        # Update the name
        old_name = self.mask_names[mask_index]
        self.mask_names[mask_index] = new_name
        
        print(f"Renamed mask at index {mask_index} from '{old_name}' to '{new_name}'")
        
        # Update the UI
        self._update_ui_after_mask_change()
    
    def _cleanup_mask_overlay(self, mask_index):
        """Clean up the visual overlay for a specific mask"""
        series_tag = f"mask_series_{mask_index}"
        if dpg.does_item_exist(series_tag):
            try:
                dpg.delete_item(series_tag)
                print(f"Deleted mask overlay {mask_index}")
            except Exception as e:
                print(f"Error deleting mask overlay {mask_index}: {e}")
    
    def _update_ui_after_mask_change(self):
        """Update UI components after mask changes (delete/rename)"""
        # Update the tool panel mask list
        if self.tool_panel:
            # Create mask list with custom names if available
            mask_display_names = []
            for idx in range(len(self.layer_masks)):
                if hasattr(self, 'mask_names') and idx < len(self.mask_names):
                    mask_display_names.append(self.mask_names[idx])
                else:
                    mask_display_names.append(f"Mask {idx + 1}")
            
            # Update the mask list items directly
            if dpg.does_item_exist("mask_list"):
                dpg.configure_item("mask_list", items=mask_display_names)
        
        # Refresh mask overlays
        if self.layer_masks:
            self.update_mask_overlays(self.layer_masks)
    
    def cleanup_segmenter_memory(self):
        """Clean up segmenter memory when needed"""
        if hasattr(self, 'segmenter') and self.segmenter:
            try:
                self.segmenter.cleanup_memory()
                print("Segmenter memory cleaned up")
            except Exception as e:
                print(f"Error cleaning up segmenter memory: {e}")
    
    def reset_segmenter(self):
        """Reset the segmenter instance to free memory"""
        if hasattr(self, 'segmenter') and self.segmenter:
            try:
                # Clean up memory first
                self.cleanup_segmenter_memory()
                # Reset the segmenter
                self.segmenter = None
                print("Segmenter instance reset")
            except Exception as e:
                print(f"Error resetting segmenter: {e}")

    def enable_segmentation_mode(self):
        """Enable segmentation mode with real-time bounding box"""
        if not self.crop_rotate_ui or not self.segmentation_bbox_renderer:
            print("Cannot enable segmentation mode: missing components")
            return False
            
        # Disable crop mode if active
        if dpg.does_item_exist("crop_mode") and dpg.get_value("crop_mode"):
            dpg.set_value("crop_mode", False)
            if self.tool_panel:
                self.tool_panel.toggle_crop_mode(None, None, None)
        
        self.segmentation_mode = True
        
        # Store the current texture for overlay
        if hasattr(self.crop_rotate_ui, 'rotated_texture') and self.crop_rotate_ui.rotated_texture is not None:
            self.segmentation_texture = self.crop_rotate_ui.rotated_texture.copy()
        else:
            # Fallback: create texture from current image
            self._create_segmentation_texture()
        
        # Set bounds for the bounding box (limit to image area)
        if self.crop_rotate_ui.original_image is not None:
            h, w = self.crop_rotate_ui.original_image.shape[:2]
            offset_x = (self.crop_rotate_ui.texture_w - w) // 2
            offset_y = (self.crop_rotate_ui.texture_h - h) // 2
            
            bounds = BoundingBox(
                x=offset_x,
                y=offset_y,
                width=w,
                height=h
            )
            self.segmentation_bbox_renderer.set_bounds(bounds)
        
        print("Segmentation mode enabled - click and drag to select area")
        return True
    
    def disable_segmentation_mode(self):
        """Disable segmentation mode"""
        self.segmentation_mode = False
        self.segmentation_texture = None
        self.pending_segmentation_box = None
        
        if self.segmentation_bbox_renderer:
            self.segmentation_bbox_renderer.reset()
        
        # Refresh the display without segmentation overlay
        if self.crop_rotate_ui:
            self.crop_rotate_ui.update_image(None, None, None)
        
        print("Segmentation mode disabled")
    
    def _create_segmentation_texture(self):
        """Create segmentation texture from current image"""
        if not self.crop_rotate_ui or self.crop_rotate_ui.original_image is None:
            return
            
        image = self.crop_rotate_ui.original_image
        h, w = image.shape[:2]
        
        # Create texture with gray background
        texture = np.full((self.crop_rotate_ui.texture_h, self.crop_rotate_ui.texture_w, 4), 
                         [100, 100, 100, 0], dtype=np.uint8)
        
        # Center the image in the texture
        offset_x = (self.crop_rotate_ui.texture_w - w) // 2
        offset_y = (self.crop_rotate_ui.texture_h - h) // 2
        
        # Convert image to RGBA if needed
        if image.shape[2] == 3:
            image_rgba = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)
        else:
            image_rgba = image
        
        # Place image in texture
        texture[offset_y:offset_y + h, offset_x:offset_x + w] = image_rgba
        
        self.segmentation_texture = texture
    
    def confirm_segmentation_selection(self):
        """Confirm the current segmentation selection and perform segmentation"""
        if not self.pending_segmentation_box or not self.segmentation_mode:
            print("No segmentation area selected")
            return
            
        # Convert bounding box to the format expected by segment_with_box
        bbox = self.pending_segmentation_box
        # Box format: [x1, y1, x2, y2]
        box = [bbox["x"], bbox["y"], bbox["x"] + bbox["w"], bbox["y"] + bbox["h"]]
        
        print(f"Confirming segmentation with box: {box}")
        
        # Disable segmentation mode
        self.disable_segmentation_mode()
        
        # Show loading indicator before performing segmentation
        self._show_segmentation_loading("Processing selection...")
        
        try:
            # Perform segmentation
            self.segment_with_box(box)
        finally:
            # Hide loading indicator (segment_with_box has its own loading management, 
            # but this ensures it's hidden if there are any early returns)
            self._hide_segmentation_loading()
        
        # Update tool panel to reflect the change
        if self.tool_panel:
            self.tool_panel.set_segmentation_mode(False)
    
    def cancel_segmentation_selection(self):
        """Cancel the current segmentation selection"""
        print("Segmentation selection cancelled")
        self.disable_segmentation_mode()
        
        # Update tool panel to reflect the change
        if self.tool_panel:
            self.tool_panel.set_segmentation_mode(False)
    
    def _show_segmentation_loading(self, message="Processing..."):
        """Show loading indicator for segmentation process"""
        print(f"Attempting to show loading indicator with message: {message}")
        if dpg.does_item_exist("segmentation_loading_group"):
            print("Loading group exists, showing indicator")
            # Update the message and show the indicator
            if dpg.does_item_exist("segmentation_loading_text"):
                dpg.set_value("segmentation_loading_text", message)
                print(f"Updated loading text to: {message}")
            dpg.configure_item("segmentation_loading_group", show=True)
            print("Loading indicator should now be visible")
        else:
            print("Loading group does not exist!")
    
    def _hide_segmentation_loading(self):
        """Hide loading indicator for segmentation process"""
        print("Attempting to hide loading indicator")
        if dpg.does_item_exist("segmentation_loading_group"):
            dpg.configure_item("segmentation_loading_group", show=False)
            print("Loading indicator hidden")
        else:
            print("Loading group does not exist for hiding!")