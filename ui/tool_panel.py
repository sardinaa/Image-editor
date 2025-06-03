import dearpygui.dearpygui as dpg
from ui.curves_panel import CurvesPanel
from ui.histogram_panel import HistogramPanel

class ToolPanel:
    def __init__(self, callback, crop_and_rotate_ref, main_window=None):
        self.callback = callback
        self.crop_and_rotate_ref = crop_and_rotate_ref  # Función para obtener la instancia de CropAndRotate
        self.main_window = main_window  # Add reference to main window
        self.curves = {"r": [(0, 0), (128, 128), (255, 255)],
                      "g": [(0, 0), (128, 128), (255, 255)],
                      "b": [(0, 0), (128, 128), (255, 255)]}
        self.panel_tag = "tool_panel"
        self.curves_panel = None
        self.histogram_panel = None  # Add histogram panel reference

    def draw(self):
        with dpg.group(horizontal=False):
            # Add histogram panel at the top
            self.histogram_panel = HistogramPanel()
            self.histogram_panel.show()
            
            dpg.add_separator()
            dpg.add_spacer(height=2)
            
            dpg.add_text("Basic Editing Tools", color=[176, 204, 255])
            dpg.add_separator()
            dpg.add_spacer(height=2)
            
            # Crop section - make it more compact
            dpg.add_checkbox(label="Crop & Rotate", tag="crop_mode", default_value=False, callback=self.toggle_crop_mode)
            with dpg.child_window(tag="crop_panel", height=65, autosize_x=True, show=False, border=True):
                dpg.add_slider_float(label="Rotation", tag="rotation_slider", height=18,
                                    min_value=0, max_value=360, default_value=0, callback=self._update_crop_rotate)
                dpg.add_button(label="Max Area", callback=self._set_max_rect, width=-1, height=18)

            dpg.add_separator()
            dpg.add_spacer(height=2)
            
            # Exposure & Lighting group
            dpg.add_text("Exposure & Lighting", color=[200, 200, 200])
            dpg.add_slider_int(label="Exposure", tag="exposure", default_value=0, min_value=-100, max_value=100, height=18, callback=self._param_changed)
            dpg.add_slider_float(label="Illumination", tag="illumination", default_value=0.0, min_value=-50.0, max_value=50.0, height=18, callback=self._param_changed)
            dpg.add_slider_float(label="Contrast", tag="contrast", default_value=1.0, min_value=0.5, max_value=3.0, height=18, callback=self._param_changed)
            
            dpg.add_spacer(height=1)
            
            # Tone Adjustments group
            dpg.add_text("Tone Adjustments", color=[200, 200, 200])
            dpg.add_slider_int(label="Shadow", tag="shadow", default_value=0, min_value=-100, max_value=100, height=18, callback=self._param_changed)
            dpg.add_slider_int(label="Whites", tag="whites", default_value=0, min_value=-100, max_value=100, height=18, callback=self._param_changed)
            dpg.add_slider_int(label="Blacks", tag="blacks", default_value=0, min_value=-100, max_value=100, height=18, callback=self._param_changed)
            
            dpg.add_spacer(height=1)
            
            # Color & Effects group
            dpg.add_text("Color & Effects", color=[200, 200, 200])
            dpg.add_slider_float(label="Saturation", tag="saturation", default_value=1.0, min_value=0.0, max_value=3.0, height=18, callback=self._param_changed)
            dpg.add_slider_int(label="Texture", tag="texture", default_value=0, min_value=0, max_value=10, height=18, callback=self._param_changed)
            dpg.add_slider_int(label="Grain", tag="grain", default_value=0, min_value=0, max_value=50, height=18, callback=self._param_changed)
            dpg.add_slider_int(label="Temperature", tag="temperature", default_value=0, min_value=-50, max_value=50, height=18, callback=self._param_changed)
            
            dpg.add_separator()
            dpg.add_spacer(height=2)
            dpg.add_text("RGB Curves", color=[176, 204, 255])
            
            # Embed the curves panel directly in the tool panel
            self.curves_panel = CurvesPanel(callback=self._param_changed)
            self.curves_panel.curves = self.curves
            self.curves_panel.show()
            
            # Masks section with toggle
            dpg.add_text("Advance settings", color=[176, 204, 255])
            dpg.add_separator()
            dpg.add_spacer(height=2)
            dpg.add_checkbox(label="Masks", tag="mask_section_toggle", default_value=False, callback=self.toggle_mask_section)
            with dpg.child_window(tag="mask_panel", height=240, autosize_x=True, show=False, border=False):
                # Segmentation options
                dpg.add_text("Segmentation", color=[200, 200, 200])
                with dpg.group(horizontal=True):
                    dpg.add_button(label="Auto Segment", tag="auto_segment_btn", callback=self._auto_segment, width=85, height=20)
                    dpg.add_checkbox(label="Box Selection Mode", tag="box_selection_mode", 
                                default_value=False, callback=self._toggle_box_selection)
                
                dpg.add_separator()
                dpg.add_spacer(height=5)
                
                # Show/hide mask overlay control
                dpg.add_checkbox(label="Show Mask Overlay", tag="show_mask_overlay", 
                               default_value=True, callback=self._toggle_mask_overlay)
                
                dpg.add_spacer(height=5)
                
                # Mask management buttons
                with dpg.group(horizontal=True):
                    dpg.add_button(label="Delete Mask", tag="delete_mask_btn", callback=self._delete_selected_mask, width=82, height=20)
                    dpg.add_button(label="Rename Mask", tag="rename_mask_btn", callback=self._rename_selected_mask, width=82, height=20)
                    dpg.add_button(label="Clear Masks", tag="clear_all_masks_btn", callback=self._clear_all_masks, width=85, height=20)
                dpg.add_listbox(items=[], tag="mask_list", callback=self._mask_selected, num_items=5)
            
            

    def _param_changed(self, sender, app_data, user_data):
        # Sync curves from curves panel if it exists
        if self.curves_panel:
            self.curves = self.curves_panel.get_curves()
        
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
        # Sync curves from curves panel before returning parameters
        if self.curves_panel:
            self.curves = self.curves_panel.get_curves()
            
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
        print(f"Tool panel returning curves: {self.curves}")  # Debug output
        return params

    def _mask_selected(self, sender, app_data, user_data):
        # When a mask is selected, show only that mask and hide others
        # app_data contains the selected mask name, we need to find its index
        if not app_data:
            return
            
        # Get the current mask list to find the index
        if dpg.does_item_exist("mask_list"):
            current_items = dpg.get_item_configuration("mask_list")["items"]
            try:
                selected_index = current_items.index(app_data)
                print(f"Selected mask: {app_data}, index: {selected_index}")
                
                # Update mask visibility if main_window is available and overlay is enabled
                if self.main_window:
                    # Check if overlay should be shown
                    show_overlay = True
                    if dpg.does_item_exist("show_mask_overlay"):
                        show_overlay = dpg.get_value("show_mask_overlay")
                    
                    if show_overlay:
                        self.main_window.show_selected_mask(selected_index)
                    
                    # Enable mask editing for the selected mask automatically
                    if (hasattr(self.main_window, 'crop_rotate_ui') and 
                        self.main_window.crop_rotate_ui and 
                        hasattr(self.main_window.crop_rotate_ui, 'image_processor') and
                        hasattr(self.main_window, 'layer_masks') and
                        selected_index < len(self.main_window.layer_masks)):
                        
                        selected_mask = self.main_window.layer_masks[selected_index]['segmentation']
                        self.main_window.crop_rotate_ui.image_processor.set_mask_editing(True, selected_mask)
                        print(f"Mask editing enabled for: {app_data}")
                else:
                    print("Main window reference not available")
                    
            except ValueError as e:
                print(f"Error finding mask index for '{app_data}': {e}")
        else:
            print("Mask list widget does not exist")
    
    def update_masks(self, masks, mask_names=None):
        # Create listbox entries based on custom names or default naming
        if mask_names and len(mask_names) >= len(masks):
            items = mask_names[:len(masks)]
        else:
            items = [f"Mask {idx+1}" for idx in range(len(masks))]
        
        if dpg.does_item_exist("mask_list"):
            dpg.configure_item("mask_list", items=items)
        else:
            print("Mask list widget does not exist.")
            
        # Auto-disable mask editing if no masks are available
        if len(masks) == 0:
            self.reset_mask_editing()
    
    def reset_mask_editing(self):
        """Reset mask editing mode and clear overlays"""
        # Hide all mask overlays
        if dpg.does_item_exist("show_mask_overlay"):
            dpg.set_value("show_mask_overlay", False)
        
        # Update the processor to disable mask editing
        if (self.main_window and hasattr(self.main_window, 'crop_rotate_ui') and 
            self.main_window.crop_rotate_ui and 
            hasattr(self.main_window.crop_rotate_ui, 'image_processor')):
            
            self.main_window.crop_rotate_ui.image_processor.set_mask_editing(False, None)
    
    def update_histogram(self, image):
        """Update the histogram with new image data"""
        if self.histogram_panel:
            self.histogram_panel.update_histogram(image)
    
    def _delete_selected_mask(self, sender, app_data, user_data):
        """Delete the currently selected mask"""
        # Get the currently selected mask from the listbox
        if not dpg.does_item_exist("mask_list"):
            return
            
        current_selection = dpg.get_value("mask_list")
        if not current_selection:
            print("No mask selected for deletion")
            return
        
        # Find the index of the selected mask in the list
        current_items = dpg.get_item_configuration("mask_list")["items"]
        try:
            selected_index = current_items.index(current_selection)
            print(f"Deleting mask: {current_selection}, index: {selected_index}")
            
            # Delete the mask through main window
            if self.main_window:
                self.main_window.delete_mask(selected_index)
            else:
                print("Main window reference not available")
        except ValueError as e:
            print(f"Error finding mask index for '{current_selection}': {e}")
    
    def _rename_selected_mask(self, sender, app_data, user_data):
        """Rename the currently selected mask"""
        # Get the currently selected mask from the listbox
        if not dpg.does_item_exist("mask_list"):
            return
            
        current_selection = dpg.get_value("mask_list")
        if not current_selection:
            print("No mask selected for renaming")
            return
        
        # Find the index of the selected mask in the list
        current_items = dpg.get_item_configuration("mask_list")["items"]
        try:
            selected_index = current_items.index(current_selection)
            print(f"Renaming mask: {current_selection}, index: {selected_index}")
            
            # Show rename dialog
            self._show_rename_dialog(selected_index, current_selection)
        except ValueError as e:
            print(f"Error finding mask index for '{current_selection}': {e}")
    
    def _show_rename_dialog(self, mask_index, current_name):
        """Show a dialog to rename the mask"""
        # Delete existing dialog if it exists
        if dpg.does_item_exist("rename_mask_window"):
            dpg.delete_item("rename_mask_window")
        
        # Create a modal window for renaming
        with dpg.window(label="Rename Mask", modal=True, tag="rename_mask_window", 
                       width=300, height=120, pos=[400, 300]):
            dpg.add_text(f"Rename: {current_name}")
            dpg.add_input_text(label="New name", tag="mask_rename_input", 
                              default_value=current_name, width=200)
            dpg.add_spacer(height=5)
            with dpg.group(horizontal=True):
                dpg.add_button(label="OK", callback=lambda: self._apply_rename(mask_index), width=80)
                dpg.add_button(label="Cancel", callback=lambda: dpg.delete_item("rename_mask_window"), width=80)
    
    def _apply_rename(self, mask_index):
        """Apply the rename operation"""
        if not dpg.does_item_exist("mask_rename_input"):
            return
            
        new_name = dpg.get_value("mask_rename_input")
        if new_name and new_name.strip() and self.main_window:
            self.main_window.rename_mask(mask_index, new_name.strip())
        
        # Close the window
        if dpg.does_item_exist("rename_mask_window"):
            dpg.delete_item("rename_mask_window")
    
    def _toggle_mask_overlay(self, sender, app_data, user_data):
        """Toggle the visibility of mask overlays"""
        show_overlay = dpg.get_value("show_mask_overlay")
        
        if self.main_window and hasattr(self.main_window, 'layer_masks'):
            # Get currently selected mask index
            current_selection = None
            selected_index = -1
            
            if dpg.does_item_exist("mask_list"):
                current_selection = dpg.get_value("mask_list")
                if current_selection:
                    current_items = dpg.get_item_configuration("mask_list")["items"]
                    try:
                        selected_index = current_items.index(current_selection)
                    except ValueError:
                        selected_index = -1
            
            if show_overlay:
                # Show the selected mask overlay if there's a selection
                if selected_index >= 0 and selected_index < len(self.main_window.layer_masks):
                    self.main_window.show_selected_mask(selected_index)
                    print(f"Showing mask overlay for: {current_selection}")
                elif len(self.main_window.layer_masks) > 0:
                    # Show first mask if no specific selection
                    self.main_window.show_selected_mask(0)
                    print("Showing first mask overlay")
            else:
                # Hide all mask overlays
                for idx in range(len(self.main_window.layer_masks)):
                    mask_tag = f"mask_series_{idx}"
                    if dpg.does_item_exist(mask_tag):
                        dpg.configure_item(mask_tag, show=False)
                print("Hidden all mask overlays")

    def toggle_mask_section(self, sender, app_data, user_data):
        """Toggle the visibility of the mask section"""
        current = dpg.get_value("mask_section_toggle")
        dpg.configure_item("mask_panel", show=current)
        
        # If mask section is hidden, also hide all mask overlays
        if not current and self.main_window and hasattr(self.main_window, 'layer_masks'):
            # Hide all mask overlays
            for idx in range(len(self.main_window.layer_masks)):
                mask_tag = f"mask_series_{idx}"
                if dpg.does_item_exist(mask_tag):
                    dpg.configure_item(mask_tag, show=False)
            print("Hidden all mask overlays (mask section toggled off)")
            
            # Disable mask editing when section is hidden
            if (hasattr(self.main_window, 'crop_rotate_ui') and 
                self.main_window.crop_rotate_ui and 
                hasattr(self.main_window.crop_rotate_ui, 'image_processor')):
                
                self.main_window.crop_rotate_ui.image_processor.set_mask_editing(False, None)
                print("Mask editing disabled (mask section toggled off)")
        
        elif current and self.main_window and hasattr(self.main_window, 'layer_masks'):
            # When mask section is shown again, respect the overlay toggle setting
            if dpg.does_item_exist("show_mask_overlay") and dpg.get_value("show_mask_overlay"):
                # Get currently selected mask and show it
                if dpg.does_item_exist("mask_list"):
                    current_selection = dpg.get_value("mask_list")
                    if current_selection:
                        current_items = dpg.get_item_configuration("mask_list")["items"]
                        try:
                            selected_index = current_items.index(current_selection)
                            if selected_index < len(self.main_window.layer_masks):
                                self.main_window.show_selected_mask(selected_index)
                                print(f"Restored mask overlay for: {current_selection}")
                        except ValueError:
                            pass
                    elif len(self.main_window.layer_masks) > 0:
                        # Show first mask if no selection
                        self.main_window.show_selected_mask(0)
                        print("Restored first mask overlay")

    def _auto_segment(self, sender, app_data, user_data):
        """Trigger automatic segmentation through main window"""
        if self.main_window:
            self.main_window.segment_current_image()
        else:
            print("Main window reference not available for auto segmentation")
    
    def _toggle_box_selection(self, sender, app_data, user_data):
        """Toggle box selection mode through main window"""
        if self.main_window:
            self.main_window.toggle_box_selection_mode(sender, app_data)
        else:
            print("Main window reference not available for box selection")
    
    def _clear_all_masks(self, sender, app_data, user_data):
        """Clear all masks through main window"""
        if self.main_window:
            self.main_window.clear_all_masks()
        else:
            print("Main window reference not available for clearing masks")