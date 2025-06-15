"""
Mask management control panel.
Handles mask display, selection, editing, and segmentation operations.
"""
import dearpygui.dearpygui as dpg
from typing import Dict, Any, Set, List
from ui.components.base_panel import BasePanel
from utils.ui_helpers import UIStateManager, ControlGroupManager, MaskOverlayManager


class MasksPanel(BasePanel):
    """Panel for mask management controls."""
    
    def __init__(self, callback=None, main_window=None):
        super().__init__(callback, main_window)
        self.panel_tag = "masks_panel_container"
        
        # Mask management state
        self.selected_mask_indices: Set[int] = set()
        self.mask_checkboxes: Dict[int, str] = {}
        self.mask_editing_enabled = False
        self.current_mask_index = -1
        self.mask_params: Dict[int, Dict[str, Any]] = {}
        self.global_params: Optional[Dict[str, Any]] = None
        
        # Control group manager for mask-related controls
        self.control_groups = ControlGroupManager()
        self._setup_control_groups()
    
    def _setup_control_groups(self):
        """Set up control groups for managing UI state."""
        # Mask controls that support enabled property
        mask_controls = [
            "auto_segment_btn", "clear_all_masks_btn", "segmentation_mode",
            "show_mask_overlay", "delete_mask_btn", "rename_mask_btn"
        ]
        self.control_groups.register_group("mask_controls", mask_controls)
        
        # Container controls that only support show/hide
        container_controls = ["segmentation_controls", "mask_table"]
        self.control_groups.register_group("mask_containers", container_controls)
    
    def setup(self) -> None:
        """Setup the masks panel."""
        self.parameters = {
            'mask_section_toggle': True,
            'segmentation_mode': False,
            'show_mask_overlay': True
        }
    
    def draw(self) -> None:
        """Draw the masks panel UI."""
        # Masks section with toggle
        dpg.add_separator()
        dpg.add_spacer(height=2)
        
        self._create_checkbox(
            label="Masks",
            tag="mask_section_toggle",
            default=True
        )
        
        # Set specific callback for mask section toggle
        if UIStateManager.safe_item_exists("mask_section_toggle"):
            dpg.set_item_callback("mask_section_toggle", self.toggle_mask_section)
        
        # Mask panel content
        with dpg.child_window(
            tag="mask_panel",
            height=240,
            autosize_x=True,
            show=True,
            border=True
        ):
            self._draw_segmentation_controls()
            self._draw_mask_management_controls()
            self._draw_mask_table()
    
    def _draw_segmentation_controls(self):
        """Draw segmentation control section."""
        # Segmentation options
        self._create_section_header("Segmentation", [200, 200, 200])
        
        with dpg.group(horizontal=True):
            self._create_button(
                label="Auto Segment",
                callback=self._auto_segment,
                width=85,
                height=20
            )
            
            self._create_button(
                label="Clear Masks",
                callback=self._clear_all_masks,
                width=85,
                height=20
            )
        
        # Add tags to buttons for control management
        if dpg.does_item_exist(dpg.last_item()):
            dpg.configure_item(dpg.last_item(), tag="clear_all_masks_btn")
        
        # Unified segmentation mode
        self._create_checkbox(
            label="Box Selection Mode",
            tag="segmentation_mode",
            default=False
        )
        
        if UIStateManager.safe_item_exists("segmentation_mode"):
            dpg.set_item_callback("segmentation_mode", self._toggle_segmentation_mode)
        
        # Segmentation control buttons (initially hidden)
        with dpg.group(horizontal=True, tag="segmentation_controls", show=False):
            self._create_button(
                label="Confirm",
                callback=self._confirm_segmentation,
                width=82,
                height=20
            )
            
            self._create_button(
                label="Cancel",
                callback=self._cancel_segmentation,
                width=82,
                height=20
            )
        
        # Loading indicator for segmentation (initially hidden)
        with dpg.group(tag="segmentation_loading_group", show=False):
            dpg.add_spacer(height=3)
            with dpg.group(horizontal=True):
                dpg.add_loading_indicator(tag="segmentation_loading_indicator", style=1, radius=2)
                dpg.add_text("Processing...", color=[200, 200, 200], tag="segmentation_loading_text")
            dpg.add_spacer(height=3)
    
    def _draw_mask_management_controls(self):
        """Draw mask management controls."""
        dpg.add_separator()
        dpg.add_spacer(height=5)
        
        # Show/hide mask overlay control
        self._create_checkbox(
            label="Show Mask Overlay",
            tag="show_mask_overlay",
            default=True
        )
        
        if UIStateManager.safe_item_exists("show_mask_overlay"):
            dpg.set_item_callback("show_mask_overlay", self._toggle_mask_overlay)
        
        dpg.add_spacer(height=5)
        
        # Mask management buttons
        with dpg.group(horizontal=True):
            self._create_button(
                label="Delete Selected",
                callback=self._delete_selected_masks,
                width=82,
                height=20
            )
            
            self._create_button(
                label="Rename Mask",
                callback=self._rename_selected_mask,
                width=82,
                height=20
            )
        
        # Add tags for control management
        button_tags = ["delete_mask_btn", "rename_mask_btn"]
        for i, tag in enumerate(button_tags):
            # Note: This is a simplified approach - in practice you'd need to track button creation
            pass
    
    def _draw_mask_table(self):
        """Draw the mask selection table."""
        # Create table for mask selection with multiple selection support
        with dpg.table(
            tag="mask_table",
            header_row=True,
            borders_innerH=True,
            borders_outerH=True,
            borders_innerV=True,
            borders_outerV=True,
            row_background=True,
            policy=dpg.mvTable_SizingFixedFit,
            height=120
        ):
            dpg.add_table_column(label="Mask Name", width_stretch=True)
    
    def toggle_mask_section(self, sender, app_data, user_data):
        """Toggle the visibility of the mask section and control editing mode."""
        current = UIStateManager.safe_get_value("mask_section_toggle", True)
        UIStateManager.safe_configure_item("mask_panel", show=current)
        
        # Handle mode conflicts
        if current:
            # If masks are being enabled, disable crop mode
            if (UIStateManager.safe_item_exists("crop_mode") and 
                UIStateManager.safe_get_value("crop_mode", False)):
                print("Disabling crop & rotate to enable masks")
                UIStateManager.safe_set_value("crop_mode", False)
                UIStateManager.safe_configure_item("crop_panel", show=False)
        
        # Control editing mode based on masks checkbox state
        if not current and self.main_window and hasattr(self.main_window, 'layer_masks'):
            self._disable_masks()
        elif current and self.main_window and hasattr(self.main_window, 'layer_masks'):
            self._enable_masks()
    
    def _disable_masks(self):
        """Disable mask functionality."""
        # Hide all mask overlays
        if self.main_window and hasattr(self.main_window, 'layer_masks'):
            MaskOverlayManager.hide_all_overlays(len(self.main_window.layer_masks))
            print("Hidden all mask overlays (masks disabled)")
        
        # Disable segmentation mode
        if UIStateManager.safe_item_exists("segmentation_mode"):
            UIStateManager.safe_set_value("segmentation_mode", False)
            if hasattr(self.main_window, 'disable_segmentation_mode'):
                self.main_window.disable_segmentation_mode()
            print("Disabled segmentation mode (masks disabled)")
        
        # Disable mask-related UI controls
        self.control_groups.disable_group("mask_controls")
        self.control_groups.hide_group("mask_containers")
        
        # Switch to global editing mode
        self._disable_mask_editing()
        print("Switched to global editing mode")
    
    def _enable_masks(self):
        """Enable mask functionality."""
        # Re-enable mask-related UI controls
        self.control_groups.enable_group("mask_controls")
        self.control_groups.show_group("mask_containers")
        
        # Restore mask overlay visibility if enabled
        if UIStateManager.safe_get_value("show_mask_overlay", True):
            self._update_mask_overlays_visibility()
    
    def _auto_segment(self, sender, app_data, user_data):
        """Trigger automatic segmentation through main window."""
        # Check if masks are enabled
        masks_enabled = UIStateManager.safe_get_value("mask_section_toggle", True)
        
        if not masks_enabled:
            print("Cannot perform auto segmentation when masks are disabled")
            return
        
        if self.main_window and hasattr(self.main_window, 'segment_current_image'):
            self.main_window.segment_current_image()
        else:
            print("Main window reference not available for auto segmentation")
    
    def _clear_all_masks(self, sender, app_data, user_data):
        """Clear all masks through main window."""
        if self.main_window and hasattr(self.main_window, 'clear_all_masks'):
            self.main_window.clear_all_masks()
        else:
            print("Main window reference not available for clearing masks")
    
    def _toggle_segmentation_mode(self, sender, app_data, user_data):
        """Toggle unified box selection/segmentation mode."""
        segmentation_mode_enabled = UIStateManager.safe_get_value("segmentation_mode", False)
        
        if segmentation_mode_enabled:
            # Try to enable modern real-time segmentation mode first
            if self.main_window and hasattr(self.main_window, 'enable_segmentation_mode'):
                success = self.main_window.enable_segmentation_mode()
                if success:
                    UIStateManager.safe_configure_item("segmentation_controls", show=True)
                    # Disable conflicting modes
                    if UIStateManager.safe_item_exists("crop_mode"):
                        UIStateManager.safe_set_value("crop_mode", False)
                    print("Real-time segmentation mode enabled")
                else:
                    # Fallback to legacy box selection mode
                    if hasattr(self.main_window, 'toggle_box_selection_mode'):
                        self.main_window.toggle_box_selection_mode("segmentation_mode", True)
                        print("Fallback to legacy box selection mode")
                    else:
                        UIStateManager.safe_set_value("segmentation_mode", False)
        else:
            # Disable both modes
            if self.main_window:
                if hasattr(self.main_window, 'disable_segmentation_mode'):
                    self.main_window.disable_segmentation_mode()
                if hasattr(self.main_window, 'toggle_box_selection_mode'):
                    self.main_window.toggle_box_selection_mode("segmentation_mode", False)
            UIStateManager.safe_configure_item("segmentation_controls", show=False)
    
    def _confirm_segmentation(self, sender, app_data, user_data):
        """Confirm the current segmentation selection."""
        UIStateManager.safe_configure_item("segmentation_controls", show=False)
        
        # Update checkbox state
        UIStateManager.temporarily_disable_callback(
            "segmentation_mode",
            lambda: UIStateManager.safe_set_value("segmentation_mode", False)
        )
        
        if self.main_window and hasattr(self.main_window, 'confirm_segmentation_selection'):
            self.main_window.confirm_segmentation_selection()
    
    def _cancel_segmentation(self, sender, app_data, user_data):
        """Cancel the current segmentation selection."""
        if self.main_window and hasattr(self.main_window, 'cancel_segmentation_selection'):
            self.main_window.cancel_segmentation_selection()
        self.set_segmentation_mode(False)
    
    def _toggle_mask_overlay(self, sender, app_data, user_data):
        """Toggle the visibility of mask overlays."""
        show_overlay = UIStateManager.safe_get_value("show_mask_overlay", True)
        
        if self.main_window and hasattr(self.main_window, 'layer_masks'):
            if show_overlay:
                self._update_mask_overlays_visibility()
                if self.selected_mask_indices:
                    selected_list = list(self.selected_mask_indices)
                    print(f"Showing mask overlays for selected masks: {selected_list}")
                elif len(self.main_window.layer_masks) > 0:
                    self.main_window.show_selected_mask(0)
                    print("Showing first mask overlay")
            else:
                MaskOverlayManager.hide_all_overlays(len(self.main_window.layer_masks))
                print("Hidden all mask overlays")
    
    def _delete_selected_masks(self, sender, app_data, user_data):
        """Delete selected masks."""
        if not self.selected_mask_indices:
            print("No masks selected for deletion")
            return
        
        # Delete in reverse order to maintain indices
        for mask_index in sorted(self.selected_mask_indices, reverse=True):
            if self.main_window and hasattr(self.main_window, 'delete_mask'):
                self.main_window.delete_mask(mask_index)
        
        # Clear selection
        self.selected_mask_indices.clear()
    
    def _rename_selected_mask(self, sender, app_data, user_data):
        """Rename the selected mask (single selection only)."""
        if len(self.selected_mask_indices) != 1:
            print("Please select exactly one mask to rename")
            return
        
        mask_index = next(iter(self.selected_mask_indices))
        current_name = f"Mask {mask_index + 1}"
        
        # Get current name from main window if available
        if (self.main_window and hasattr(self.main_window, 'mask_names') and 
            mask_index < len(self.main_window.mask_names)):
            current_name = self.main_window.mask_names[mask_index]
        
        self._show_rename_dialog(mask_index, current_name)
    
    def _show_rename_dialog(self, mask_index: int, current_name: str):
        """Show a dialog to rename the mask."""
        # Delete existing dialog if it exists
        if UIStateManager.safe_item_exists("rename_mask_window"):
            dpg.delete_item("rename_mask_window")
        
        # Create a modal window for renaming
        with dpg.window(
            label="Rename Mask",
            modal=True,
            tag="rename_mask_window",
            width=300,
            height=120,
            pos=[400, 300]
        ):
            dpg.add_text(f"Rename: {current_name}")
            dpg.add_input_text(
                label="New name",
                tag="mask_rename_input",
                default_value=current_name,
                width=200
            )
            dpg.add_spacer(height=5)
            with dpg.group(horizontal=True):
                dpg.add_button(
                    label="OK",
                    callback=lambda: self._apply_rename(mask_index),
                    width=80
                )
                dpg.add_button(
                    label="Cancel",
                    callback=lambda: dpg.delete_item("rename_mask_window"),
                    width=80
                )
    
    def _apply_rename(self, mask_index: int):
        """Apply the rename operation."""
        if not UIStateManager.safe_item_exists("mask_rename_input"):
            return
        
        new_name = UIStateManager.safe_get_value("mask_rename_input", "")
        if new_name and new_name.strip() and self.main_window:
            if hasattr(self.main_window, 'rename_mask'):
                self.main_window.rename_mask(mask_index, new_name.strip())
        
        # Close the window
        if UIStateManager.safe_item_exists("rename_mask_window"):
            dpg.delete_item("rename_mask_window")
    
    def _update_mask_overlays_visibility(self):
        """Update the visibility of mask overlays based on selection."""
        if not self.main_window or not hasattr(self.main_window, 'layer_masks'):
            return
        
        show_overlay = UIStateManager.safe_get_value("show_mask_overlay", True)
        if not show_overlay:
            return
        
        # Hide all masks first
        MaskOverlayManager.hide_all_overlays(len(self.main_window.layer_masks))
        
        # Show selected masks
        MaskOverlayManager.show_selected_overlays(list(self.selected_mask_indices))
    
    def _disable_mask_editing(self):
        """Disable mask editing and return to global editing mode."""
        # Implementation would go here - this is complex logic from the original
        # For now, just reset the state
        self.mask_editing_enabled = False
        self.current_mask_index = -1
        
        # Force an immediate image update
        if self.callback:
            self.callback(None, None, None)
    
    def set_segmentation_mode(self, enabled: bool):
        """Set the segmentation mode state from external code."""
        UIStateManager.temporarily_disable_callback(
            "segmentation_mode",
            lambda: UIStateManager.safe_set_value("segmentation_mode", enabled)
        )
        UIStateManager.safe_configure_item("segmentation_controls", show=enabled)
    
    def update_masks(self, masks: List[Dict[str, Any]], mask_names: List[str] = None):
        """Update the mask table with new masks."""
        # Create mask entries
        if mask_names and len(mask_names) >= len(masks):
            items = mask_names[:len(masks)]
        else:
            items = [f"Mask {idx+1}" for idx in range(len(masks))]
        
        # Clear existing table rows
        if UIStateManager.safe_item_exists("mask_table"):
            # Clear tracking dictionaries
            self.mask_checkboxes.clear()
            self.selected_mask_indices.clear()
            
            # Add new rows for each mask
            for idx, mask_name in enumerate(items):
                with dpg.table_row(tag=f"mask_row_{idx}", parent="mask_table"):
                    dpg.add_selectable(
                        label=mask_name,
                        tag=f"mask_selectable_{idx}",
                        callback=self._create_row_callback(idx),
                        span_columns=True
                    )
                
                # Track the selectable
                self.mask_checkboxes[idx] = f"mask_selectable_{idx}"
    
    def _create_row_callback(self, mask_index: int):
        """Create a proper callback for row selection with correct mask index."""
        return lambda s, a, u: self._mask_row_clicked(s, a, u, mask_index)
    
    def _mask_row_clicked(self, sender, app_data, user_data, mask_index: int):
        """Handle row clicks for single and multiple selection."""
        # Check if Ctrl is pressed for multiple selection
        is_ctrl_pressed = dpg.is_key_down(dpg.mvKey_LControl) or dpg.is_key_down(dpg.mvKey_RControl)
        
        if is_ctrl_pressed:
            # Multiple selection mode - toggle this mask's selection
            if mask_index in self.selected_mask_indices:
                self.selected_mask_indices.discard(mask_index)
                if mask_index in self.mask_checkboxes:
                    selectable_tag = self.mask_checkboxes[mask_index]
                    UIStateManager.safe_set_value(selectable_tag, False)
                print(f"Deselected mask {mask_index}, total selected: {len(self.selected_mask_indices)}")
            else:
                self.selected_mask_indices.add(mask_index)
                if mask_index in self.mask_checkboxes:
                    selectable_tag = self.mask_checkboxes[mask_index]
                    UIStateManager.safe_set_value(selectable_tag, True)
                print(f"Selected mask {mask_index}, total selected: {len(self.selected_mask_indices)}")
        else:
            # Single selection mode - clear all other selections
            for idx in list(self.selected_mask_indices):
                if idx != mask_index and idx in self.mask_checkboxes:
                    selectable_tag = self.mask_checkboxes[idx]
                    UIStateManager.safe_set_value(selectable_tag, False)
            
            self.selected_mask_indices.clear()
            self.selected_mask_indices.add(mask_index)
            
            if mask_index in self.mask_checkboxes:
                selectable_tag = self.mask_checkboxes[mask_index]
                UIStateManager.safe_set_value(selectable_tag, True)
            print(f"Quick selected mask {mask_index}")
        
        # Update overlay visibility and editing logic
        self._update_mask_overlays_visibility()
        self._apply_editing_logic()
    
    def _apply_editing_logic(self):
        """Apply editing logic based on selection count."""
        if len(self.selected_mask_indices) == 1:
            selected_index = next(iter(self.selected_mask_indices))
            self._apply_single_mask_editing(selected_index)
        elif len(self.selected_mask_indices) == 0:
            if self.mask_editing_enabled:
                self._disable_mask_editing()
        else:
            # Multiple masks selected - disable mask editing
            if self.mask_editing_enabled:
                self._disable_mask_editing()
            print(f"Multiple masks selected ({len(self.selected_mask_indices)}), mask editing disabled")
    
    def _apply_single_mask_editing(self, mask_index: int):
        """Apply mask editing to a single selected mask."""
        masks_enabled = UIStateManager.safe_get_value("mask_section_toggle", True)
        
        if masks_enabled:
            # Enable mask editing for selected mask
            # This would integrate with the mask editing system
            print(f"Enabling mask editing for mask {mask_index}")
        else:
            if self.mask_editing_enabled:
                self._disable_mask_editing()
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get current mask parameters."""
        return {
            'mask_section_toggle': UIStateManager.safe_get_value("mask_section_toggle", True),
            'segmentation_mode': UIStateManager.safe_get_value("segmentation_mode", False),
            'show_mask_overlay': UIStateManager.safe_get_value("show_mask_overlay", True)
        }
