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
        
        # Mask editing state management
        self.mask_editing_enabled = False
        self.current_mask_index = -1
        self.global_params = None  # Store global parameters when switching to mask mode
        self.mask_params = {}  # Store parameters for each mask {mask_index: params}

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
            dpg.add_separator()
            dpg.add_spacer(height=2)
            dpg.add_checkbox(label="Masks", tag="mask_section_toggle", default_value=True, callback=self.toggle_mask_section)
            with dpg.child_window(tag="mask_panel", height=240, autosize_x=True, show=True, border=True):
                # Segmentation options
                dpg.add_text("Segmentation", color=[200, 200, 200])
                with dpg.group(horizontal=True):
                    dpg.add_button(label="Auto Segment", tag="auto_segment_btn", callback=self._auto_segment, width=85, height=20)
                    dpg.add_button(label="Clear Masks", tag="clear_all_masks_btn", callback=self._clear_all_masks, width=85, height=20)
                dpg.add_checkbox(label="Box Selection Mode", tag="box_selection_mode", 
                               default_value=False, callback=self._toggle_box_selection)
                
                dpg.add_separator()
                dpg.add_spacer(height=5)
                
                # Show/hide mask overlay control
                dpg.add_checkbox(label="Show Mask Overlay", tag="show_mask_overlay", 
                               default_value=True, callback=self._toggle_mask_overlay)
                
                dpg.add_spacer(height=5)
                
                dpg.add_spacer(height=5)
                
                # Mask management buttons
                with dpg.group(horizontal=True):
                    dpg.add_button(label="Delete Mask", tag="delete_mask_btn", callback=self._delete_selected_mask, width=82, height=20)
                    dpg.add_button(label="Rename Mask", tag="rename_mask_btn", callback=self._rename_selected_mask, width=82, height=20)
                dpg.add_listbox(items=[], tag="mask_list", callback=self._mask_selected, num_items=5)
            
            dpg.add_separator()
            dpg.add_spacer(height=2)
    
    def _param_changed(self, sender, app_data, user_data):
        # Sync curves from curves panel if it exists
        if self.curves_panel:
            self.curves = self.curves_panel.get_curves()
        
        # Save mask parameters if in mask editing mode
        if self.mask_editing_enabled and self.current_mask_index >= 0:
            self._save_mask_parameters()
        
        if self.callback:
            self.callback()

    def toggle_crop_mode(self, sender, app_data, user_data):
        current = dpg.get_value("crop_mode")
        
        # If crop mode is being enabled, disable masks
        if current:
            # Check if masks are currently enabled
            if dpg.does_item_exist("mask_section_toggle") and dpg.get_value("mask_section_toggle"):
                print("Disabling masks to enable crop & rotate")
                dpg.set_value("mask_section_toggle", False)
                # Trigger the mask section toggle to handle all the mask disabling logic
                self.toggle_mask_section(None, None, None)
        
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
        # When a mask is selected, update visibility and apply mask editing based on main masks checkbox
        if not app_data:
            # No mask selected - disable mask editing and switch to global mode
            if self.mask_editing_enabled:
                self._disable_mask_editing()
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
                    
                    # Update current mask index for potential mask editing
                    self.current_mask_index = selected_index
                    
                    # Check if masks are enabled (main checkbox) to apply mask editing
                    masks_enabled = True
                    if dpg.does_item_exist("mask_section_toggle"):
                        masks_enabled = dpg.get_value("mask_section_toggle")
                    
                    if masks_enabled:
                        # Masks are enabled - apply mask editing to selected mask
                        self._apply_mask_editing(selected_index)
                    else:
                        # Masks are disabled - ensure we're in global editing mode
                        if self.mask_editing_enabled:
                            self._disable_mask_editing()
                    
                    # Trigger image update
                    self._param_changed(sender, app_data, user_data)
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
        # Disable mask editing mode
        self._disable_mask_editing()
        
        # Hide all mask overlays
        if dpg.does_item_exist("show_mask_overlay"):
            dpg.set_value("show_mask_overlay", False)
    
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
        """Toggle the visibility of the mask section and control editing mode"""
        current = dpg.get_value("mask_section_toggle")
        dpg.configure_item("mask_panel", show=current)
        
        # If masks are being enabled, disable crop mode
        if current:
            # Check if crop mode is currently enabled
            if dpg.does_item_exist("crop_mode") and dpg.get_value("crop_mode"):
                print("Disabling crop & rotate to enable masks")
                dpg.set_value("crop_mode", False)
                dpg.configure_item("crop_panel", show=False)
        
        # Control editing mode based on masks checkbox state
        if not current and self.main_window and hasattr(self.main_window, 'layer_masks'):
            # Hide all mask overlays when masks are disabled
            for idx in range(len(self.main_window.layer_masks)):
                mask_tag = f"mask_series_{idx}"
                if dpg.does_item_exist(mask_tag):
                    dpg.configure_item(mask_tag, show=False)
            print("Hidden all mask overlays (masks disabled)")
            
            # Disable box selection mode when masks are disabled
            if dpg.does_item_exist("box_selection_mode"):
                dpg.set_value("box_selection_mode", False)
                # Also disable it in the main window
                if hasattr(self.main_window, 'toggle_box_selection_mode'):
                    self.main_window.toggle_box_selection_mode(None, False)
                print("Disabled box selection mode (masks disabled)")
            
            # Disable mask-related UI controls
            mask_controls = ["auto_segment_btn", "clear_all_masks_btn", "box_selection_mode", 
                           "show_mask_overlay", "delete_mask_btn", "rename_mask_btn", "mask_list"]
            for control in mask_controls:
                if dpg.does_item_exist(control):
                    dpg.configure_item(control, enabled=False)
            
            # Switch to global editing mode when masks are disabled
            self._disable_mask_editing()
            print("Switched to global editing mode")
        
        elif current and self.main_window and hasattr(self.main_window, 'layer_masks'):
            # Re-enable mask-related UI controls
            mask_controls = ["auto_segment_btn", "clear_all_masks_btn", "box_selection_mode", 
                           "show_mask_overlay", "delete_mask_btn", "rename_mask_btn", "mask_list"]
            for control in mask_controls:
                if dpg.does_item_exist(control):
                    dpg.configure_item(control, enabled=True)
            
            # When mask section is shown again, check if a mask is selected for editing
            if dpg.does_item_exist("mask_list"):
                current_selection = dpg.get_value("mask_list")
                if current_selection:
                    # Apply mask editing to selected mask
                    current_items = dpg.get_item_configuration("mask_list")["items"]
                    try:
                        selected_index = current_items.index(current_selection)
                        if selected_index < len(self.main_window.layer_masks):
                            self._apply_mask_editing(selected_index)
                            print(f"Switched to mask editing mode for: {current_selection}")
                    except ValueError:
                        pass
            
            # Restore overlay visibility if enabled
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
        # Check if masks are enabled before allowing auto segmentation
        masks_enabled = True
        if dpg.does_item_exist("mask_section_toggle"):
            masks_enabled = dpg.get_value("mask_section_toggle")
        
        if not masks_enabled:
            print("Cannot perform auto segmentation when masks are disabled")
            return
        
        if self.main_window:
            self.main_window.segment_current_image()
        else:
            print("Main window reference not available for auto segmentation")
    
    def _toggle_box_selection(self, sender, app_data, user_data):
        """Toggle box selection mode through main window"""
        # Check if masks are enabled before allowing box selection
        masks_enabled = True
        if dpg.does_item_exist("mask_section_toggle"):
            masks_enabled = dpg.get_value("mask_section_toggle")
        
        if not masks_enabled:
            # If masks are disabled, prevent box selection from being activated
            if dpg.does_item_exist("box_selection_mode"):
                dpg.set_value("box_selection_mode", False)
            print("Cannot enable box selection when masks are disabled")
            return
        
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
    

    
    def _get_current_parameters(self):
        """Get current UI parameter values"""
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
        }
        
        # Add curves if available
        if self.curves_panel:
            params['curves'] = self.curves_panel.get_curves()
        else:
            params['curves'] = self.curves
            
        return params
    
    def _apply_parameters(self, params):
        """Apply parameter values to UI controls"""
        if not params:
            return
            
        # Apply basic parameters
        for param_name, value in params.items():
            if param_name != 'curves' and dpg.does_item_exist(param_name):
                dpg.set_value(param_name, value)
        
        # Apply curves
        if 'curves' in params and self.curves_panel:
            self.curves_panel.set_curves(params['curves'])
            self.curves = params['curves']
    
    def _save_mask_parameters(self):
        """Save current parameters for the active mask"""
        if self.mask_editing_enabled and self.current_mask_index >= 0:
            self.mask_params[self.current_mask_index] = self._get_current_parameters()
    
    def _load_mask_parameters(self, mask_index):
        """Load parameters for a specific mask"""
        if mask_index in self.mask_params:
            self._apply_parameters(self.mask_params[mask_index])
        else:
            # Initialize with default parameters if none exist
            default_params = {
                'exposure': 0,
                'illumination': 0.0,
                'contrast': 1.0,
                'shadow': 0,
                'whites': 0,
                'blacks': 0,
                'saturation': 1.0,
                'texture': 0,
                'grain': 0,
                'temperature': 0,
                'curves': {"r": [(0, 0), (128, 128), (255, 255)],
                          "g": [(0, 0), (128, 128), (255, 255)],
                          "b": [(0, 0), (128, 128), (255, 255)]}
            }
            self.mask_params[mask_index] = default_params
            self._apply_parameters(default_params)
    
    def _disable_mask_editing(self):
        """Disable mask editing and return to global editing mode"""
        # Save current mask parameters
        if self.current_mask_index >= 0:
            self._save_mask_parameters()
        
        # Commit current edits to base image before switching to global editing
        if (self.main_window and hasattr(self.main_window, 'crop_rotate_ui') and 
            self.main_window.crop_rotate_ui and 
            hasattr(self.main_window.crop_rotate_ui, 'image_processor')):
            
            self.main_window.crop_rotate_ui.image_processor.commit_edits_to_base()

        # Restore global parameters
        if self.global_params:
            self._apply_parameters(self.global_params)
        
        # Disable mask editing in processor
        if (self.main_window and hasattr(self.main_window, 'crop_rotate_ui') and 
            self.main_window.crop_rotate_ui and 
            hasattr(self.main_window.crop_rotate_ui, 'image_processor')):
            
            self.main_window.crop_rotate_ui.image_processor.set_mask_editing(False, None)
        
        # Update state
        self.mask_editing_enabled = False
        self.current_mask_index = -1
        
        print("Disabled mask editing, returned to global editing")
    
    def _apply_mask_editing(self, mask_index):
        """Apply mask editing to the specified mask"""
        if not self.main_window or not hasattr(self.main_window, 'layer_masks'):
            return
            
        if mask_index >= len(self.main_window.layer_masks):
            return
        
        # Store current global parameters if not in mask editing mode
        if not self.mask_editing_enabled:
            self.global_params = self._get_current_parameters()
            # Commit any global edits to the base image before switching to mask editing
            if (hasattr(self.main_window, 'crop_rotate_ui') and 
                self.main_window.crop_rotate_ui and 
                hasattr(self.main_window.crop_rotate_ui, 'image_processor')):
                
                self.main_window.crop_rotate_ui.image_processor.commit_edits_to_base()
        
        # Save current mask parameters and commit edits if switching masks
        if self.current_mask_index >= 0 and self.current_mask_index != mask_index:
            self._save_mask_parameters()
            # Commit the edits from the previous mask to the base image
            if (hasattr(self.main_window, 'crop_rotate_ui') and 
                self.main_window.crop_rotate_ui and 
                hasattr(self.main_window.crop_rotate_ui, 'image_processor')):
                
                self.main_window.crop_rotate_ui.image_processor.commit_edits_to_base()
        
        # Switch to new mask
        self.current_mask_index = mask_index
        self.mask_editing_enabled = True
        
        # Load mask-specific parameters
        self._load_mask_parameters(mask_index)
        
        # Enable mask editing in processor
        if (hasattr(self.main_window, 'crop_rotate_ui') and 
            self.main_window.crop_rotate_ui and 
            hasattr(self.main_window.crop_rotate_ui, 'image_processor')):
            
            selected_mask = self.main_window.layer_masks[mask_index]['segmentation']
            self.main_window.crop_rotate_ui.image_processor.set_mask_editing(True, selected_mask)
        
        print(f"Enabled mask editing for mask {mask_index}")