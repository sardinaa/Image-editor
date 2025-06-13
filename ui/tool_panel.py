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
        
        # Multiple mask selection state
        self.selected_mask_indices = set()  # Set of selected mask indices
        self.mask_checkboxes = {}  # Dictionary to track checkbox tags {mask_index: checkbox_tag}

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
                
                # Unified segmentation mode (replaces both segmentation mode and box selection mode)
                dpg.add_checkbox(label="Box Selection Mode", tag="segmentation_mode", 
                               default_value=False, callback=self._toggle_segmentation_mode)
                
                # Segmentation control buttons (initially hidden)
                with dpg.group(horizontal=True, tag="segmentation_controls", show=False):
                    dpg.add_button(label="Confirm", tag="confirm_segmentation_btn", callback=self._confirm_segmentation, 
                                 width=82, height=20)
                    dpg.add_button(label="Cancel", tag="cancel_segmentation_btn", callback=self._cancel_segmentation, 
                                 width=82, height=20)
                
                # Loading indicator for segmentation (initially hidden)
                with dpg.group(tag="segmentation_loading_group", show=False):
                    dpg.add_spacer(height=3)
                    with dpg.group(horizontal=True):
                        dpg.add_loading_indicator(tag="segmentation_loading_indicator", style=1, radius=2)
                        dpg.add_text("Processing...", color=[200, 200, 200], tag="segmentation_loading_text")
                    dpg.add_spacer(height=3)
                
                dpg.add_separator()
                dpg.add_spacer(height=5)
                
                # Show/hide mask overlay control
                dpg.add_checkbox(label="Show Mask Overlay", tag="show_mask_overlay", 
                               default_value=True, callback=self._toggle_mask_overlay)
                
                dpg.add_spacer(height=5)
                
                dpg.add_spacer(height=5)
                
                # Mask management buttons
                with dpg.group(horizontal=True):
                    dpg.add_button(label="Delete Selected", tag="delete_mask_btn", callback=self._delete_selected_masks, width=82, height=20)
                    dpg.add_button(label="Rename Mask", tag="rename_mask_btn", callback=self._rename_selected_mask, width=82, height=20)
                
                # Create table for mask selection with multiple selection support
                with dpg.table(tag="mask_table", header_row=True, borders_innerH=True, borders_outerH=True, 
                             borders_innerV=True, borders_outerV=True, row_background=True, 
                             policy=dpg.mvTable_SizingFixedFit, height=120):
                    dpg.add_table_column(label="Mask Name", width_stretch=True)
            
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

    def _create_row_callback(self, mask_index):
        """Create a proper callback for row selection with correct mask index"""
        return lambda s, a, u: self._mask_row_clicked(s, a, u, mask_index)
    

    
    def _mask_row_clicked(self, sender, app_data, user_data, mask_index):
        """Handle row clicks for single and multiple selection"""
        # Check if Ctrl is pressed for multiple selection
        is_ctrl_pressed = dpg.is_key_down(dpg.mvKey_LControl) or dpg.is_key_down(dpg.mvKey_RControl)
        
        if is_ctrl_pressed:
            # Multiple selection mode - toggle this mask's selection
            if mask_index in self.selected_mask_indices:
                # Deselect this mask
                self.selected_mask_indices.discard(mask_index)
                if mask_index in self.mask_checkboxes:
                    selectable_tag = self.mask_checkboxes[mask_index]
                    if dpg.does_item_exist(selectable_tag):
                        dpg.set_value(selectable_tag, False)
                print(f"Deselected mask {mask_index}, total selected: {len(self.selected_mask_indices)}")
            else:
                # Select this mask
                self.selected_mask_indices.add(mask_index)
                if mask_index in self.mask_checkboxes:
                    selectable_tag = self.mask_checkboxes[mask_index]
                    if dpg.does_item_exist(selectable_tag):
                        dpg.set_value(selectable_tag, True)
                print(f"Selected mask {mask_index}, total selected: {len(self.selected_mask_indices)}")
        else:
            # Single selection mode - clear all other selections and select only this mask
            for idx in list(self.selected_mask_indices):
                if idx != mask_index and idx in self.mask_checkboxes:
                    selectable_tag = self.mask_checkboxes[idx]
                    if dpg.does_item_exist(selectable_tag):
                        dpg.set_value(selectable_tag, False)
            
            # Clear selected indices and add only this one
            self.selected_mask_indices.clear()
            self.selected_mask_indices.add(mask_index)
            
            # Update the selectable for this mask
            if mask_index in self.mask_checkboxes:
                selectable_tag = self.mask_checkboxes[mask_index]
                if dpg.does_item_exist(selectable_tag):
                    dpg.set_value(selectable_tag, True)
            
            print(f"Quick selected mask {mask_index}")
        
        # Update overlay and apply editing logic
        self._update_mask_overlays_visibility()
        
        # Apply editing logic based on selection count
        if len(self.selected_mask_indices) == 1:
            selected_index = next(iter(self.selected_mask_indices))
            self._apply_single_mask_editing(selected_index)
        elif len(self.selected_mask_indices) == 0:
            # No masks selected - switch to global editing
            if self.mask_editing_enabled:
                self._disable_mask_editing()
        else:
            # Multiple masks selected - disable mask editing (can't edit multiple masks simultaneously)
            if self.mask_editing_enabled:
                self._disable_mask_editing()
            print(f"Multiple masks selected ({len(self.selected_mask_indices)}), mask editing disabled")
    
    def _apply_single_mask_editing(self, mask_index):
        """Apply mask editing to a single selected mask"""
        # Check if masks are enabled (main checkbox) to apply mask editing
        masks_enabled = True
        if dpg.does_item_exist("mask_section_toggle"):
            masks_enabled = dpg.get_value("mask_section_toggle")
        
        if masks_enabled:
            # Masks are enabled - apply mask editing to selected mask
            self._apply_mask_editing(mask_index)
        else:
            # Masks are disabled - ensure we're in global editing mode
            if self.mask_editing_enabled:
                self._disable_mask_editing()
    
    def _update_mask_overlays_visibility(self):
        """Update the visibility of mask overlays based on selection"""
        if not self.main_window or not hasattr(self.main_window, 'layer_masks'):
            return
        
        # Check if overlay should be shown
        show_overlay = True
        if dpg.does_item_exist("show_mask_overlay"):
            show_overlay = dpg.get_value("show_mask_overlay")
        
        if not show_overlay:
            return
        
        # Hide all masks first
        for idx in range(len(self.main_window.layer_masks)):
            mask_tag = f"mask_series_{idx}"
            if dpg.does_item_exist(mask_tag):
                dpg.configure_item(mask_tag, show=False)
        
        # Show selected masks
        for selected_index in self.selected_mask_indices:
            if selected_index < len(self.main_window.layer_masks):
                mask_tag = f"mask_series_{selected_index}"
                if dpg.does_item_exist(mask_tag):
                    dpg.configure_item(mask_tag, show=True)
                    print(f"Showing mask overlay {selected_index}")

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
                    
                    # Check if masks are enabled (main checkbox) to apply mask editing
                    masks_enabled = True
                    if dpg.does_item_exist("mask_section_toggle"):
                        masks_enabled = dpg.get_value("mask_section_toggle")
                    
                    if masks_enabled:
                        # Masks are enabled - apply mask editing to selected mask
                        # Note: _apply_mask_editing will handle updating current_mask_index internally
                        self._apply_mask_editing(selected_index)
                        # Note: _apply_mask_editing now handles the image update internally
                    else:
                        # Masks are disabled - ensure we're in global editing mode
                        if self.mask_editing_enabled:
                            self._disable_mask_editing()
                        # Note: _disable_mask_editing now handles the image update internally
                else:
                    print("Main window reference not available")
                    
            except ValueError as e:
                print(f"Error finding mask index for '{app_data}': {e}")
        else:
            print("Mask list widget does not exist")
    
    def update_masks(self, masks, mask_names=None):
        # Create mask entries based on custom names or default naming
        if mask_names and len(mask_names) >= len(masks):
            items = mask_names[:len(masks)]
        else:
            items = [f"Mask {idx+1}" for idx in range(len(masks))]
        
        # Clear existing table rows
        if dpg.does_item_exist("mask_table"):
            # Get all existing rows and delete them
            for mask_index in list(self.mask_checkboxes.keys()):
                row_tag = f"mask_row_{mask_index}"
                if dpg.does_item_exist(row_tag):
                    dpg.delete_item(row_tag)
            
            # Clear our tracking dictionaries
            self.mask_checkboxes.clear()
            self.selected_mask_indices.clear()
            
            # Add new rows for each mask
            for idx, mask_name in enumerate(items):
                with dpg.table_row(tag=f"mask_row_{idx}", parent="mask_table"):
                    # Mask name column (selectable for multiple selection)
                    dpg.add_selectable(label=mask_name, tag=f"mask_selectable_{idx}",
                                     callback=self._create_row_callback(idx),
                                     span_columns=True)
                
                # Track the selectable (no longer using checkboxes)
                self.mask_checkboxes[idx] = f"mask_selectable_{idx}"
        else:
            print("Mask table widget does not exist.")
            
        # Auto-disable mask editing if no masks are available
        if len(masks) == 0:
            self.reset_mask_editing()
    
    def get_selected_mask_indices(self):
        """Get the currently selected mask indices"""
        return list(self.selected_mask_indices)
    
    def select_mask(self, mask_index, add_to_selection=False):
        """Programmatically select a mask"""
        if not add_to_selection:
            # Clear existing selection
            for idx in list(self.selected_mask_indices):
                if idx in self.mask_checkboxes:
                    checkbox_tag = self.mask_checkboxes[idx]
                    if dpg.does_item_exist(checkbox_tag):
                        dpg.set_value(checkbox_tag, False)
            self.selected_mask_indices.clear()
        
        # Add new selection
        if mask_index in self.mask_checkboxes:
            checkbox_tag = self.mask_checkboxes[mask_index]
            if dpg.does_item_exist(checkbox_tag):
                dpg.set_value(checkbox_tag, True)
                self.selected_mask_indices.add(mask_index)
                self._update_mask_overlays_visibility()
                
                # Apply editing if single selection
                if len(self.selected_mask_indices) == 1:
                    self._apply_single_mask_editing(mask_index)

    def reset_mask_editing(self):
        """Reset mask editing mode and clear overlays"""
        # Disable mask editing mode
        self._disable_mask_editing()
        
        # Clear selection state
        self.selected_mask_indices.clear()
        
        # Uncheck all checkboxes
        for checkbox_tag in self.mask_checkboxes.values():
            if dpg.does_item_exist(checkbox_tag):
                dpg.set_value(checkbox_tag, False)
        
        # Hide all mask overlays
        if dpg.does_item_exist("show_mask_overlay"):
            dpg.set_value("show_mask_overlay", False)
    
    def update_histogram(self, image):
        """Update the histogram with new image data"""
        if self.histogram_panel:
            self.histogram_panel.update_histogram(image)
    
    def _delete_selected_masks(self, sender, app_data, user_data):
        """Delete all currently selected masks"""
        if not self.selected_mask_indices:
            print("No masks selected for deletion")
            return
        
        # Convert to sorted list (delete from highest index to lowest to avoid index shifting)
        indices_to_delete = sorted(self.selected_mask_indices, reverse=True)
        
        print(f"Deleting masks at indices: {indices_to_delete}")
        
        # Delete masks through main window
        if self.main_window:
            for mask_index in indices_to_delete:
                self.main_window.delete_mask(mask_index)
        else:
            print("Main window reference not available")
        
        # Clear selection since masks were deleted
        self.selected_mask_indices.clear()
    
    def _rename_selected_mask(self, sender, app_data, user_data):
        """Rename the currently selected mask (works only with single selection)"""
        if len(self.selected_mask_indices) == 0:
            print("No mask selected for renaming")
            return
        elif len(self.selected_mask_indices) > 1:
            print("Multiple masks selected. Please select only one mask for renaming.")
            return
        
        # Get the single selected mask index
        selected_index = next(iter(self.selected_mask_indices))
        
        # Get current mask name
        current_name = f"Mask {selected_index + 1}"
        if (self.main_window and hasattr(self.main_window, 'mask_names') and 
            self.main_window.mask_names and selected_index < len(self.main_window.mask_names)):
            current_name = self.main_window.mask_names[selected_index]
        
        print(f"Renaming mask at index {selected_index}, current name: {current_name}")
        
        # Show rename dialog
        self._show_rename_dialog(selected_index, current_name)
    
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
            if show_overlay:
                # Show overlays for selected masks
                self._update_mask_overlays_visibility()
                if self.selected_mask_indices:
                    selected_list = list(self.selected_mask_indices)
                    print(f"Showing mask overlays for selected masks: {selected_list}")
                elif len(self.main_window.layer_masks) > 0:
                    # Show first mask if none selected
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
            
            # Disable segmentation mode when masks are disabled
            if dpg.does_item_exist("segmentation_mode"):
                dpg.set_value("segmentation_mode", False)
                if hasattr(self.main_window, 'disable_segmentation_mode'):
                    self.main_window.disable_segmentation_mode()
                print("Disabled segmentation mode (masks disabled)")
            
            # Disable mask-related UI controls (only controls that support enabled property)
            mask_controls = ["auto_segment_btn", "clear_all_masks_btn", "segmentation_mode", 
                           "show_mask_overlay", "delete_mask_btn", "rename_mask_btn"]
            for control in mask_controls:
                if dpg.does_item_exist(control):
                    dpg.configure_item(control, enabled=False)
            
            # Hide container controls that don't support enabled property
            container_controls = ["segmentation_controls", "mask_table"]
            for control in container_controls:
                if dpg.does_item_exist(control):
                    dpg.configure_item(control, show=False)
            
            # Switch to global editing mode when masks are disabled
            self._disable_mask_editing()
            print("Switched to global editing mode")
        
        elif current and self.main_window and hasattr(self.main_window, 'layer_masks'):
            # Re-enable mask-related UI controls (only controls that support enabled property)
            mask_controls = ["auto_segment_btn", "clear_all_masks_btn", "segmentation_mode",
                           "show_mask_overlay", "delete_mask_btn", "rename_mask_btn"]
            for control in mask_controls:
                if dpg.does_item_exist(control):
                    dpg.configure_item(control, enabled=True)
            
            # Show container controls that don't support enabled property
            container_controls = ["segmentation_controls", "mask_table"]
            for control in container_controls:
                if dpg.does_item_exist(control):
                    dpg.configure_item(control, show=True)
            
            # When mask section is shown again, restore any previous selection
            # For now, we'll just ensure overlays are shown if enabled
            if dpg.does_item_exist("show_mask_overlay") and dpg.get_value("show_mask_overlay"):
                self._update_mask_overlays_visibility()

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

        image_processor = None
        if (self.main_window and hasattr(self.main_window, 'crop_rotate_ui') and 
            self.main_window.crop_rotate_ui and 
            hasattr(self.main_window.crop_rotate_ui, 'image_processor')):
            image_processor = self.main_window.crop_rotate_ui.image_processor

        # Commit current edits to base image before switching to global editing
        if image_processor:
            # Include curves in the commit to preserve them
            current_params = self._get_current_parameters()
            curves_data = current_params.get('curves', None)
            image_processor.commit_edits_to_base(curves_data)
            print("Committed mask edits to base image before switching to global mode")

        # Restore global parameters
        if self.global_params:
            self._apply_parameters(self.global_params)
            
            # Update image processor parameters with the restored global values
            if image_processor:
                params = self.global_params
                image_processor.exposure = params.get('exposure', 0)
                image_processor.illumination = params.get('illumination', 0.0)
                image_processor.contrast = params.get('contrast', 1.0)
                image_processor.shadow = params.get('shadow', 0)
                image_processor.whites = params.get('whites', 0)
                image_processor.blacks = params.get('blacks', 0)
                image_processor.saturation = params.get('saturation', 1.0)
                image_processor.texture = params.get('texture', 0)
                image_processor.grain = params.get('grain', 0)
                image_processor.temperature = params.get('temperature', 0)

        # Disable mask editing in processor
        if image_processor:
            image_processor.set_mask_editing(False, None)

        # Update state
        self.mask_editing_enabled = False
        self.current_mask_index = -1

        print("Disabled mask editing, returned to global editing")
        
        # Force an immediate image update to show the global edits
        if self.callback:
            self.callback()
    
    def _apply_mask_editing(self, mask_index):
        """Apply mask editing to the specified mask"""
        if not self.main_window or not hasattr(self.main_window, 'layer_masks'):
            return
            
        if mask_index >= len(self.main_window.layer_masks):
            return
        
        image_processor = None
        if (hasattr(self.main_window, 'crop_rotate_ui') and 
            self.main_window.crop_rotate_ui and 
            hasattr(self.main_window.crop_rotate_ui, 'image_processor')):
            image_processor = self.main_window.crop_rotate_ui.image_processor
        
        if not image_processor:
            print("Error: Could not access image processor")
            return
        
        # Store current global parameters if not in mask editing mode
        if not self.mask_editing_enabled:
            print(f"Switching from global to mask {mask_index} - storing global params")
            self.global_params = self._get_current_parameters()
            # Commit any global edits to the base image before switching to mask editing
            # Include curves in the commit to preserve them
            curves_data = self.global_params.get('curves', None)
            image_processor.commit_edits_to_base(curves_data)
        
        # Save current mask parameters and commit edits if switching masks
        if self.current_mask_index >= 0 and self.current_mask_index != mask_index:
            print(f"Switching from mask {self.current_mask_index} to mask {mask_index} - committing previous edits")
            self._save_mask_parameters()
            # Commit the edits from the previous mask to the base image
            # Include curves in the commit to preserve them
            current_params = self._get_current_parameters()
            curves_data = current_params.get('curves', None)
            image_processor.commit_edits_to_base(curves_data)
            print(f"Committed edits from mask {self.current_mask_index} to base image")
        
        # Switch to new mask
        self.current_mask_index = mask_index
        self.mask_editing_enabled = True
        
        # Load mask-specific parameters (this will update UI controls)
        self._load_mask_parameters(mask_index)
        
        # Update image processor parameters with the loaded values
        params = self._get_current_parameters()
        image_processor.exposure = params.get('exposure', 0)
        image_processor.illumination = params.get('illumination', 0.0)
        image_processor.contrast = params.get('contrast', 1.0)
        image_processor.shadow = params.get('shadow', 0)
        image_processor.whites = params.get('whites', 0)
        image_processor.blacks = params.get('blacks', 0)
        image_processor.saturation = params.get('saturation', 1.0)
        image_processor.texture = params.get('texture', 0)
        image_processor.grain = params.get('grain', 0)
        image_processor.temperature = params.get('temperature', 0)
        
        # Enable mask editing in processor
        selected_mask = self.main_window.layer_masks[mask_index]['segmentation']
        image_processor.set_mask_editing(True, selected_mask)
        
        print(f"Enabled mask editing for mask {mask_index} with parameters: {params}")
        
        # Force an immediate image update to show the new mask with its parameters
        if self.callback:
            self.callback()
    
    def _toggle_segmentation_mode(self, sender, app_data, user_data):
        """Toggle unified box selection/segmentation mode"""
        segmentation_mode_enabled = dpg.get_value("segmentation_mode")
        
        if segmentation_mode_enabled:
            # Try to enable modern real-time segmentation mode first
            if self.main_window and hasattr(self.main_window, 'enable_segmentation_mode'):
                success = self.main_window.enable_segmentation_mode()
                if success:
                    # Show confirm/cancel buttons for real-time mode
                    dpg.configure_item("segmentation_controls", show=True)
                    # Disable conflicting modes
                    if dpg.does_item_exist("crop_mode"):
                        dpg.set_value("crop_mode", False)
                    print("Real-time segmentation mode enabled")
                else:
                    # Fallback to legacy box selection mode
                    if hasattr(self.main_window, 'toggle_box_selection_mode'):
                        self.main_window.toggle_box_selection_mode("segmentation_mode", True)
                        print("Fallback to legacy box selection mode")
                    else:
                        # Failed to enable any mode, reset checkbox
                        dpg.set_value("segmentation_mode", False)
            else:
                # Fallback to legacy box selection if main window doesn't support new mode
                if self.main_window and hasattr(self.main_window, 'toggle_box_selection_mode'):
                    self.main_window.toggle_box_selection_mode("segmentation_mode", True)
                    print("Using legacy box selection mode")
                else:
                    # Failed to enable any mode, reset checkbox
                    dpg.set_value("segmentation_mode", False)
        else:
            # Disable both modes
            if self.main_window:
                # Disable real-time segmentation mode
                if hasattr(self.main_window, 'disable_segmentation_mode'):
                    self.main_window.disable_segmentation_mode()
                # Disable legacy box selection mode
                if hasattr(self.main_window, 'toggle_box_selection_mode'):
                    self.main_window.toggle_box_selection_mode("segmentation_mode", False)
            # Hide confirm/cancel buttons
            dpg.configure_item("segmentation_controls", show=False)
    
    def _confirm_segmentation(self, sender, app_data, user_data):
        """Confirm the current segmentation selection"""
        # Hide the buttons immediately when confirm is clicked
        dpg.configure_item("segmentation_controls", show=False)
        
        # Also update the checkbox state immediately
        if dpg.does_item_exist("segmentation_mode"):
            # Temporarily disable the callback to avoid circular calls
            original_callback = dpg.get_item_callback("segmentation_mode")
            dpg.set_item_callback("segmentation_mode", None)
            
            # Set the value without triggering callback
            dpg.set_value("segmentation_mode", False)
            
            # Restore the callback
            dpg.set_item_callback("segmentation_mode", original_callback)
        
        # Now proceed with the actual segmentation
        if self.main_window and hasattr(self.main_window, 'confirm_segmentation_selection'):
            self.main_window.confirm_segmentation_selection()
    
    def _cancel_segmentation(self, sender, app_data, user_data):
        """Cancel the current segmentation selection"""
        if self.main_window and hasattr(self.main_window, 'cancel_segmentation_selection'):
            self.main_window.cancel_segmentation_selection()
        # Reset UI state
        self.set_segmentation_mode(False)
    
    def set_segmentation_mode(self, enabled):
        """Set the segmentation mode state from external code"""
        if dpg.does_item_exist("segmentation_mode"):
            # Temporarily disable the callback to avoid circular calls
            original_callback = dpg.get_item_callback("segmentation_mode")
            dpg.set_item_callback("segmentation_mode", None)
            
            # Set the value without triggering callback
            dpg.set_value("segmentation_mode", enabled)
            
            # Restore the callback
            dpg.set_item_callback("segmentation_mode", original_callback)
            
            # Manually update the controls visibility
            dpg.configure_item("segmentation_controls", show=enabled)