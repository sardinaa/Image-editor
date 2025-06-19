"""
Mask management panel.
Handles mask display, selection, editing, and segmentation operations.
"""
import dearpygui.dearpygui as dpg
from typing import Dict, Any, Set, List, Optional
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
        
        # Parameter tracking for persistent mask editing
        self.mask_params: Dict[int, Dict[str, Any]] = {}  # Current parameter values per mask
        self.mask_committed_params: Dict[int, Dict[str, Any]] = {}  # Already committed values per mask (baseline)
        self.mask_base_image_states: Dict[int, Any] = {}  # Base image state before each mask was edited
        self.global_params: Optional[Dict[str, Any]] = None
        
        # Mask grouping functionality
        self.mask_groups: Dict[str, Set[int]] = {}  # group_id -> set of mask indices
        self.mask_to_group: Dict[int, str] = {}  # mask_index -> group_id
        self.next_group_id = 0
        
        # Control group manager for mask-related controls
        self.control_groups = ControlGroupManager()
        self._setup_control_groups()
    
    def _setup_control_groups(self):
        """Set up control groups for managing UI state."""
        # Mask controls that support enabled property
        mask_controls = [
            "auto_segment_btn", "clear_all_masks_btn", "segmentation_mode",
            "show_mask_overlay", "delete_mask_btn", "rename_mask_btn", "group_ungroup_btn"
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
                height=20,
                tag="delete_mask_btn"
            )
            
            self._create_button(
                label="Rename Mask",
                callback=self._rename_selected_mask,
                width=82,
                height=20,
                tag="rename_mask_btn"
            )
            
            self._create_button(
                label="Group",
                callback=self._toggle_group_selected_masks,
                width=82,
                height=20,
                tag="group_ungroup_btn"
            )
        
        # Add tags for control management
        button_tags = ["delete_mask_btn", "rename_mask_btn", "group_ungroup_btn"]
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
            # When disabling masks, commit any current edits and disable mask editing
            if self.mask_editing_enabled:
                self._commit_current_edits_to_base()
            self._disable_masks()
        elif current and self.main_window and hasattr(self.main_window, 'layer_masks'):
            self._enable_masks()
    
    def _disable_masks(self):
        """Disable mask functionality."""
        # Commit current edits before disabling masks
        if self.mask_editing_enabled:
            self._commit_current_edits_to_base()
        
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
        
        # Restore mask overlay visibility if enabled and crop mode is not active
        if (UIStateManager.safe_get_value("show_mask_overlay", True) and 
            not (UIStateManager.safe_item_exists("crop_mode") and 
                 UIStateManager.safe_get_value("crop_mode", False))):
            self._update_mask_overlays_visibility()
            print("Restored mask overlays (masks enabled, crop mode not active)")
        else:
            print("Not restoring mask overlays - crop mode is active")
    
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
        
        # Clean up group data for deleted masks
        for mask_index in sorted(self.selected_mask_indices, reverse=True):
            self._cleanup_group_data_for_deleted_mask(mask_index)
            
            if self.main_window and hasattr(self.main_window, 'delete_mask'):
                self.main_window.delete_mask(mask_index)
        
        # Clear selection
        self.selected_mask_indices.clear()
        
        # Update group button label
        self._update_group_button_label()
    
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
        
        # Don't show overlays if crop mode is active
        if (UIStateManager.safe_item_exists("crop_mode") and 
            UIStateManager.safe_get_value("crop_mode", False)):
            print("Not showing mask overlays - crop mode is active")
            return
        
        # Hide all masks first
        MaskOverlayManager.hide_all_overlays(len(self.main_window.layer_masks))
        
        # Show selected masks
        MaskOverlayManager.show_selected_overlays(list(self.selected_mask_indices))
    
    def _disable_mask_editing(self):
        """Disable mask editing and return to global editing mode."""
        # Commit current mask edits to base image before disabling
        self._commit_current_edits_to_base()
        
        # Reset UI parameters to defaults for global editing
        self._reset_ui_parameters_to_defaults()
        
        # Disable mask editing in the processor
        if self.main_window and self.main_window.app_service and self.main_window.app_service.image_service:
            processor = self.main_window.app_service.image_service.image_processor
            if processor:
                processor.set_mask_editing(False)
                print("Disabled mask editing in processor")
        
        # Reset mask editing state
        self.mask_editing_enabled = False
        self.current_mask_index = -1
        
        # Force an immediate image update to show global editing mode
        if self.callback:
            self.callback(None, None, None)
        
        print("🌐 Switched to global editing mode")
    
    def _commit_current_edits_to_base(self):
        """Commit current edits to the base image for cumulative editing."""
        if not self.main_window or not self.main_window.app_service:
            return
            
        try:
            # Get current parameters including curves
            current_params = self._get_current_parameters()
            curves_data = self._get_current_curves_data()
            
            # Store current parameters as committed and keep them for persistence
            if self.mask_editing_enabled and self.current_mask_index >= 0:
                # Store what was committed (for reference)
                self.mask_committed_params[self.current_mask_index] = {
                    'parameters': current_params.copy(),
                    'curves': curves_data
                }
                
                # Keep the parameters persistent so user can readjust them
                self.mask_params[self.current_mask_index] = {
                    'parameters': current_params.copy(),
                    'curves': curves_data
                }
                
                print(f"💾 Committed parameters for mask {self.current_mask_index} (keeping persistent for readjustment)")
            
            # Get the image processor
            image_service = self.main_window.app_service.image_service
            if image_service and image_service.image_processor:
                processor = image_service.image_processor
                
                # Commit current edits to base image
                processor.commit_edits_to_base(curves_data)
                
                # Reset processor parameters to defaults for next editing session
                processor.exposure = 0
                processor.illumination = 0.0
                processor.contrast = 1.0
                processor.shadow = 0
                processor.highlights = 0
                processor.whites = 0
                processor.blacks = 0
                processor.saturation = 1.0
                processor.texture = 0
                processor.grain = 0
                processor.temperature = 0
                
                print("✓ Committed current edits to base image and reset processor parameters")
                
        except Exception as e:
            print(f"Error committing edits to base: {e}")
            import traceback
            traceback.print_exc()
    
    def _get_current_parameters(self):
        """Get current parameter values from UI controls."""
        params = {}
        try:
            # Use the same parameter collection logic as the main window
            if self.main_window and hasattr(self.main_window, '_collect_current_parameters'):
                params = self.main_window._collect_current_parameters()
        except Exception as e:
            print(f"Error collecting current parameters: {e}")
        return params
    
    def _get_current_curves_data(self):
        """Get current curves data from the tool panel."""
        curves_data = None
        try:
            if self.main_window and hasattr(self.main_window, 'tool_panel'):
                tool_panel = self.main_window.tool_panel
                if tool_panel and hasattr(tool_panel, 'curves_panel') and tool_panel.curves_panel:
                    curves_data = tool_panel.curves_panel.get_curves()
        except Exception as e:
            print(f"Error getting curves data: {e}")
        return curves_data
    
    def _reset_ui_parameters_to_defaults(self):
        """Reset all UI parameters to their default values without triggering callbacks."""
        if not self.main_window or not self.main_window.tool_panel:
            return
        
        try:
            print("🔄 Resetting UI parameters to defaults...")
            
            # Temporarily disable callbacks to prevent triggering while resetting
            tool_panel = self.main_window.tool_panel
            tool_panel.disable_parameter_callbacks()
            
            # Define default parameter values
            default_params = {
                # Exposure parameters
                'exposure': 0,
                'illumination': 0.0,
                'contrast': 1.0,
                'shadow': 0,
                'highlights': 0,
                'whites': 0,
                'blacks': 0,
                
                # Color effects parameters
                'saturation': 1.0,
                'texture': 0,
                'grain': 0,
                'temperature': 0
            }
            
            # Reset UI controls to default values
            for param_name, default_value in default_params.items():
                if UIStateManager.safe_item_exists(param_name):
                    UIStateManager.safe_set_value(param_name, default_value)
            
            # Reset curves to default (linear)
            if tool_panel.curves_panel:
                tool_panel.curves_panel.curves = {
                    "r": [(0, 0), (128, 128), (255, 255)],
                    "g": [(0, 0), (128, 128), (255, 255)],
                    "b": [(0, 0), (128, 128), (255, 255)]
                }
                tool_panel.curves_panel.update_plot()
            
            # Re-enable callbacks
            tool_panel.enable_parameter_callbacks()
            
            print("✅ UI parameters reset to defaults")
            
        except Exception as e:
            print(f"Error resetting UI parameters: {e}")
            # Make sure to re-enable callbacks even if there's an error
            if tool_panel:
                tool_panel.enable_parameter_callbacks()
    
    def _save_mask_parameters(self, mask_index: int):
        """Save current parameters for a specific mask."""
        try:
            # Get current parameters
            current_params = self._get_current_parameters()
            curves_data = self._get_current_curves_data()
            
            # Store in mask_params dictionary
            self.mask_params[mask_index] = {
                'parameters': current_params,
                'curves': curves_data
            }
            
            print(f"💾 Saved parameters for mask {mask_index}: {len(current_params)} params, curves: {curves_data is not None}")
            
        except Exception as e:
            print(f"Error saving parameters for mask {mask_index}: {e}")

    def _clear_mask_parameters(self, mask_index: int):
        """Clear saved parameters for a mask when they are committed to base image."""
        try:
            if mask_index in self.mask_params:
                del self.mask_params[mask_index]
                print(f"🗑️ Cleared saved parameters for mask {mask_index} (committed to base image)")
        except Exception as e:
            print(f"Error clearing parameters for mask {mask_index}: {e}")

    def _load_mask_parameters(self, mask_index: int):
        """Load saved parameters for a specific mask for persistent editing."""
        try:
            # Check if we have saved parameters for this mask
            if mask_index in self.mask_params:
                saved_data = self.mask_params[mask_index]
                saved_params = saved_data.get('parameters', {})
                saved_curves = saved_data.get('curves')
                
                if saved_params or saved_curves:
                    print(f"📂 Loading persistent parameters for mask {mask_index}...")
                    
                    # Temporarily disable callbacks to prevent triggering during load
                    tool_panel = self.main_window.tool_panel
                    tool_panel.disable_parameter_callbacks()
                    
                    # If this mask was previously edited, restore base image to pre-edit state
                    # This prevents double-application of the saved parameters
                    if (mask_index in self.mask_base_image_states and 
                        self.main_window.app_service and self.main_window.app_service.image_service):
                        processor = self.main_window.app_service.image_service.image_processor
                        if processor:
                            # Restore base image to the state before this mask was first edited
                            processor.base_image = self.mask_base_image_states[mask_index].copy()
                            print(f"🔄 Restored base image to pre-edit state for mask {mask_index}")
                    
                    # Always reset processor to defaults first to prevent double-application
                    # This ensures that the UI values represent the actual change being applied
                    if self.main_window.app_service and self.main_window.app_service.image_service:
                        processor = self.main_window.app_service.image_service.image_processor
                        if processor:
                            processor.exposure = 0
                            processor.illumination = 0.0
                            processor.contrast = 1.0
                            processor.shadow = 0
                            processor.highlights = 0
                            processor.whites = 0
                            processor.blacks = 0
                            processor.saturation = 1.0
                            processor.texture = 0
                            processor.grain = 0
                            processor.temperature = 0
                            print(f"🔄 Reset processor to defaults before loading UI parameters")
                    
                    # Apply saved parameters to UI controls
                    for param_name, value in saved_params.items():
                        if UIStateManager.safe_item_exists(param_name):
                            UIStateManager.safe_set_value(param_name, value)
                    
                    # Apply saved curves
                    if saved_curves and tool_panel.curves_panel:
                        tool_panel.curves_panel.set_curves(saved_curves)
                    
                    # Re-enable callbacks
                    tool_panel.enable_parameter_callbacks()
                    
                    print(f"✅ Loaded persistent parameters for mask {mask_index} (base image restored to prevent double-application)")
                    return
            
            # No saved parameters, reset to defaults
            print(f"📝 No saved parameters for mask {mask_index}, using defaults")
            self._reset_ui_parameters_to_defaults()
            
        except Exception as e:
            print(f"Error loading parameters for mask {mask_index}: {e}")
            # Make sure to re-enable callbacks even if there's an error
            if self.main_window and self.main_window.tool_panel:
                self.main_window.tool_panel.enable_parameter_callbacks()
    
    def _save_global_parameters(self):
        """Save current global editing parameters."""
        try:
            # Get current parameters
            current_params = self._get_current_parameters()
            curves_data = self._get_current_curves_data()
            
            # Store in global_params
            self.global_params = {
                'parameters': current_params,
                'curves': curves_data
            }
            
            print(f"💾 Saved global parameters: {len(current_params)} params, curves: {curves_data is not None}")
            
        except Exception as e:
            print(f"Error saving global parameters: {e}")
    
    def _load_global_parameters(self):
        """Load saved global editing parameters."""
        try:
            if self.global_params:
                saved_params = self.global_params.get('parameters', {})
                saved_curves = self.global_params.get('curves')
                
                if saved_params or saved_curves:
                    print(f"📂 Loading saved global parameters...")
                    
                    # Temporarily disable callbacks
                    tool_panel = self.main_window.tool_panel
                    tool_panel.disable_parameter_callbacks()
                    
                    # Apply saved parameters to UI controls
                    for param_name, value in saved_params.items():
                        if UIStateManager.safe_item_exists(param_name):
                            UIStateManager.safe_set_value(param_name, value)
                    
                    # Apply saved curves
                    if saved_curves and tool_panel.curves_panel:
                        tool_panel.curves_panel.set_curves(saved_curves)
                    
                    # Re-enable callbacks
                    tool_panel.enable_parameter_callbacks()
                    
                    print(f"✅ Loaded global parameters")
                    return
            
            print(f"📝 No saved global parameters, keeping current values")
            
        except Exception as e:
            print(f"Error loading global parameters: {e}")
            # Make sure to re-enable callbacks even if there's an error
            if self.main_window and self.main_window.tool_panel:
                self.main_window.tool_panel.enable_parameter_callbacks()
    
    def set_segmentation_mode(self, enabled: bool):
        """Set the segmentation mode state from external code."""
        UIStateManager.temporarily_disable_callback(
            "segmentation_mode",
            lambda: UIStateManager.safe_set_value("segmentation_mode", enabled)
        )
        UIStateManager.safe_configure_item("segmentation_controls", show=enabled)
    
    def update_masks(self, masks: List[Dict[str, Any]], mask_names: List[str] = None):
        """Update the mask table with new masks."""
        try:
            print(f"MasksPanel: Updating mask table with {len(masks)} masks")
            
            # Create mask entries with proper names
            if mask_names and len(mask_names) >= len(masks):
                base_names = mask_names[:len(masks)]
            else:
                base_names = [f"Mask {idx+1}" for idx in range(len(masks))]
            
            # Create display names that include group information
            display_names = []
            for idx, base_name in enumerate(base_names):
                if idx in self.mask_to_group:
                    group_id = self.mask_to_group[idx]
                    group_num = group_id.split('_')[-1] if '_' in group_id else group_id
                    display_name = f"[G{group_num}] {base_name}"
                else:
                    display_name = base_name
                display_names.append(display_name)
            
            # Check if mask table exists
            if not UIStateManager.safe_item_exists("mask_table"):
                print("Mask table does not exist, cannot update masks")
                return
                
            # First, delete all existing rows
            # Need to create a list first because we'll be modifying as we iterate
            rows_to_delete = []
            
            try:
                # Get all child items in the mask table
                children = dpg.get_item_children("mask_table", slot=1)
                if children:
                    for item_id in children:
                        # Safely check if item exists and has an alias
                        if dpg.does_item_exist(item_id):
                            try:
                                # Try to get the alias if it exists
                                tag = None
                                if dpg.does_alias_exist(item_id):
                                    tag = dpg.get_item_alias(item_id)
                                    
                                # If we found a mask row tag, add it to deletion list
                                if tag and tag.startswith("mask_row_"):
                                    rows_to_delete.append(item_id)
                            except Exception as e:
                                print(f"Error checking item alias: {e}")
                                # If there's an error with alias, use a different approach
                                # Just delete the item directly if it's a table row
                                if dpg.get_item_type(item_id) == "mvAppItemType::mvTableRow":
                                    rows_to_delete.append(item_id)
            except Exception as e:
                print(f"Error identifying rows to delete: {e}")
                # Fallback: delete all children
                if dpg.does_item_exist("mask_table"):
                    children = dpg.get_item_children("mask_table", slot=1)
                    if children:
                        rows_to_delete = children
            
            # Now delete all rows
            for row_id in rows_to_delete:
                try:
                    if dpg.does_item_exist(row_id):
                        dpg.delete_item(row_id)
                except Exception as e:
                    print(f"Error deleting row {row_id}: {e}")
                
            # Clear our tracking collections
            self.mask_checkboxes.clear()
            self.selected_mask_indices.clear()
            
            # Add new rows for each mask
            for idx, display_name in enumerate(display_names):
                with dpg.table_row(tag=f"mask_row_{idx}", parent="mask_table"):
                    # Add the selectable item for this row
                    dpg.add_selectable(
                        label=display_name,
                        tag=f"mask_selectable_{idx}",
                        callback=self._create_row_callback(idx),
                        span_columns=True
                    )
                
                # Track the selectable item
                self.mask_checkboxes[idx] = f"mask_selectable_{idx}"
            
            # Update group button label
            self._update_group_button_label()
                
            print(f"Successfully updated mask table with {len(display_names)} items")
            
        except Exception as e:
            print(f"Error in MasksPanel.update_masks: {e}")
            import traceback
            traceback.print_exc()
    
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
                # If mask is grouped, deselect entire group
                if self._is_mask_grouped(mask_index):
                    group_id = self.mask_to_group[mask_index]
                    group_masks = self.mask_groups[group_id].copy()
                    for idx in group_masks:
                        self.selected_mask_indices.discard(idx)
                        if idx in self.mask_checkboxes:
                            selectable_tag = self.mask_checkboxes[idx]
                            UIStateManager.safe_set_value(selectable_tag, False)
                    print(f"Deselected group {group_id} with masks {group_masks}")
                else:
                    # Single mask deselection
                    self.selected_mask_indices.discard(mask_index)
                    if mask_index in self.mask_checkboxes:
                        selectable_tag = self.mask_checkboxes[mask_index]
                        UIStateManager.safe_set_value(selectable_tag, False)
                    print(f"Deselected mask {mask_index}")
            else:
                # If mask is grouped, select entire group
                if self._is_mask_grouped(mask_index):
                    self._expand_group_selection(mask_index)
                else:
                    # Single mask selection
                    self.selected_mask_indices.add(mask_index)
                    if mask_index in self.mask_checkboxes:
                        selectable_tag = self.mask_checkboxes[mask_index]
                        UIStateManager.safe_set_value(selectable_tag, True)
                    print(f"Selected mask {mask_index}")
                
            print(f"Total selected: {len(self.selected_mask_indices)}")
        else:
            # Single selection mode - clear all other selections
            for idx in list(self.selected_mask_indices):
                if idx in self.mask_checkboxes:
                    selectable_tag = self.mask_checkboxes[idx]
                    UIStateManager.safe_set_value(selectable_tag, False)
            
            self.selected_mask_indices.clear()
            
            # If clicked mask is grouped, select entire group
            if self._is_mask_grouped(mask_index):
                self._expand_group_selection(mask_index)
            else:
                # Single mask selection
                self.selected_mask_indices.add(mask_index)
                if mask_index in self.mask_checkboxes:
                    selectable_tag = self.mask_checkboxes[mask_index]
                    UIStateManager.safe_set_value(selectable_tag, True)
                print(f"Quick selected mask {mask_index}")
        
        # Update button label based on selection
        self._update_group_button_label()
        
        # Update overlay visibility and editing logic
        self._update_mask_overlays_visibility()
        self._apply_editing_logic()
    
    def _apply_editing_logic(self):
        """Apply editing logic based on selection count."""
        if len(self.selected_mask_indices) == 1:
            selected_index = next(iter(self.selected_mask_indices))
            # Save global parameters if we're switching from global editing to mask editing
            if not self.mask_editing_enabled:
                self._save_global_parameters()
            self._apply_single_mask_editing(selected_index)
        elif len(self.selected_mask_indices) == 0:
            if self.mask_editing_enabled:
                # Switching from mask editing to global editing
                self._disable_mask_editing()
                # Load global parameters if they exist
                self._load_global_parameters()
                print("Switched to global editing mode")
        else:
            # Multiple masks selected - check if they're all in the same group
            if self._are_selected_masks_grouped():
                # All selected masks are in the same group - enable group editing
                self._apply_group_editing()
            else:
                # Mixed selection or ungrouped multiple masks - disable mask editing
                if self.mask_editing_enabled:
                    self._disable_mask_editing()
                    self._load_global_parameters()
                print(f"Multiple masks selected ({len(self.selected_mask_indices)}), mask editing disabled")
    
    def _are_selected_masks_grouped(self) -> bool:
        """Check if all selected masks belong to the same group."""
        if len(self.selected_mask_indices) < 2:
            return False
        
        # Get the group of the first mask
        first_mask = next(iter(self.selected_mask_indices))
        if first_mask not in self.mask_to_group:
            return False
        
        first_group = self.mask_to_group[first_mask]
        
        # Check if all other masks are in the same group
        for mask_index in self.selected_mask_indices:
            if mask_index not in self.mask_to_group or self.mask_to_group[mask_index] != first_group:
                return False
        
        return True
    
    def _apply_group_editing(self):
        """Apply mask editing to a group of masks uniformly."""
        if not self.selected_mask_indices:
            return
        
        # Use the first mask in the group as the representative
        representative_mask = min(self.selected_mask_indices)
        group_id = self.mask_to_group.get(representative_mask)
        
        print(f"Enabling group editing for group {group_id} with {len(self.selected_mask_indices)} masks")
        
        # Save current mask parameters if we're switching from another mask
        if self.mask_editing_enabled and self.current_mask_index != representative_mask and self.current_mask_index >= 0:
            self._save_mask_parameters(self.current_mask_index)
            self._commit_current_edits_to_base()
            print(f"💾 Saved and committed parameters for mask {self.current_mask_index} when switching away")
        
        # Save global parameters if coming from global editing
        if not self.mask_editing_enabled:
            self._save_global_parameters()
            self._commit_current_edits_to_base()
            print(f"📝 Committed global edits before switching to group editing")
        
        # Enable group editing mode
        self.mask_editing_enabled = True
        self.current_mask_index = representative_mask
        
        # For group editing, we'll use a combined mask of all selected masks
        if self.main_window and hasattr(self.main_window, 'layer_masks'):
            combined_mask = self._create_combined_mask(self.selected_mask_indices)
            
            if combined_mask is not None and self.main_window.app_service and self.main_window.app_service.image_service:
                processor = self.main_window.app_service.image_service.image_processor
                if processor:
                    # Enable mask editing with the combined mask
                    processor.set_mask_editing(True, combined_mask)
                    print(f"Enabled group editing with combined mask for {len(self.selected_mask_indices)} masks")
                    
                    # Capture base image state for the representative mask
                    self._capture_base_image_state(representative_mask)
                    
                    # Load mask-specific parameters for the representative mask
                    self._load_mask_parameters(representative_mask)
                    
                    # Trigger parameter update to refresh the display
                    if self.callback:
                        self.callback(None, None, None)
    
    def _create_combined_mask(self, mask_indices: Set[int]):
        """Create a combined mask from multiple mask indices."""
        if not self.main_window or not hasattr(self.main_window, 'layer_masks'):
            return None
        
        try:
            import numpy as np
            
            combined_mask = None
            
            for mask_index in mask_indices:
                if mask_index < len(self.main_window.layer_masks):
                    mask_data = self.main_window.layer_masks[mask_index]
                    mask = mask_data.get("segmentation")
                    
                    if mask is not None:
                        if combined_mask is None:
                            combined_mask = mask.copy()
                        else:
                            # Combine masks using logical OR
                            combined_mask = np.logical_or(combined_mask, mask)
            
            return combined_mask
            
        except Exception as e:
            print(f"Error creating combined mask: {e}")
            return None
    
    def _apply_single_mask_editing(self, mask_index: int):
        """Apply mask editing to a single selected mask."""
        masks_enabled = UIStateManager.safe_get_value("mask_section_toggle", True)
        
        if masks_enabled:
            # Save current mask parameters if we're switching from another mask
            if self.mask_editing_enabled and self.current_mask_index != mask_index and self.current_mask_index >= 0:
                self._save_mask_parameters(self.current_mask_index)
                # Always commit when switching away from a mask to preserve changes
                self._commit_current_edits_to_base()
                print(f"💾 Saved and committed parameters for mask {self.current_mask_index} when switching away")
            
            # Check if this mask has been edited before
            mask_previously_edited = mask_index in self.mask_base_image_states
            is_switching_masks = self.current_mask_index != mask_index
            
            # Only commit current edits if coming from global editing
            # Don't commit when switching between masks (already handled above)
            if not self.mask_editing_enabled:
                # Coming from global editing - commit global changes
                self._commit_current_edits_to_base()
                print(f"📝 Committed global edits before switching to mask {mask_index}")
            
            # Capture base image state before editing this mask (for reset functionality)
            self._capture_base_image_state(mask_index)
            
            # Enable mask editing for selected mask
            if self.main_window and hasattr(self.main_window, 'layer_masks'):
                if 0 <= mask_index < len(self.main_window.layer_masks):
                    mask_data = self.main_window.layer_masks[mask_index]
                    mask = mask_data.get("segmentation")
                    
                    # Get image processor from app service
                    if self.main_window.app_service and self.main_window.app_service.image_service:
                        processor = self.main_window.app_service.image_service.image_processor
                        if processor and mask is not None:
                            # Enable mask editing with the selected mask
                            processor.set_mask_editing(True, mask)
                            self.mask_editing_enabled = True
                            self.current_mask_index = mask_index
                            
                            print(f"Enabled mask editing for mask {mask_index}")
                            
                            # Load mask-specific parameters (persistent values for readjustment)
                            # This loads the saved UI values so user can continue editing from where they left off
                            self._load_mask_parameters(mask_index)
                            
                            # Trigger parameter update to refresh the display
                            if self.callback:
                                self.callback(None, None, None)
        else:
            if self.mask_editing_enabled:
                self._disable_mask_editing()

    def _capture_base_image_state(self, mask_index: int):
        """Capture the current base image state before editing a mask for the first time."""
        try:
            # Only capture if we haven't already saved a state for this mask
            if mask_index in self.mask_base_image_states:
                return  # Already captured
            
            # Get the processor
            if not (self.main_window and self.main_window.app_service and 
                   self.main_window.app_service.image_service):
                return
                
            processor = self.main_window.app_service.image_service.image_processor
            if not processor:
                return
            
            # Save the current base image state
            self.mask_base_image_states[mask_index] = processor.base_image.copy()
            print(f"💾 Captured base image state for mask {mask_index} (for reset functionality)")
            
        except Exception as e:
            print(f"Error capturing base image state for mask {mask_index}: {e}")

    def get_parameters(self) -> Dict[str, Any]:
        """Get current mask parameters."""
        return {
            'mask_section_toggle': UIStateManager.safe_get_value("mask_section_toggle", True),
            'segmentation_mode': UIStateManager.safe_get_value("segmentation_mode", False),
            'show_mask_overlay': UIStateManager.safe_get_value("show_mask_overlay", True)
        }
    
    def _toggle_group_selected_masks(self, sender, app_data, user_data):
        """Toggle group/ungroup for selected masks."""
        if not self.selected_mask_indices:
            print("No masks selected for grouping operation")
            return
        
        # Check if any selected masks are already grouped
        grouped_masks = {idx for idx in self.selected_mask_indices if idx in self.mask_to_group}
        ungrouped_masks = self.selected_mask_indices - grouped_masks
        
        if grouped_masks and not ungrouped_masks:
            # All selected masks are grouped - ungroup them
            self._ungroup_masks(grouped_masks)
        else:
            # Either no masks are grouped, or mixed selection - create/update group
            self._group_masks(self.selected_mask_indices)
        
        # Update button label
        self._update_group_button_label()
    
    def _group_masks(self, mask_indices: Set[int]):
        """Group the specified masks together."""
        if len(mask_indices) < 2:
            print("Need at least 2 masks to create a group")
            return
        
        # Check if any masks are already in groups
        existing_groups = set()
        for idx in mask_indices:
            if idx in self.mask_to_group:
                existing_groups.add(self.mask_to_group[idx])
        
        if existing_groups:
            # Merge into existing group(s) - use the first group found
            group_id = next(iter(existing_groups))
            
            # Remove masks from other groups if they exist
            for other_group_id in existing_groups:
                if other_group_id != group_id:
                    self._merge_groups(group_id, other_group_id)
            
            # Add new masks to the group
            self.mask_groups[group_id].update(mask_indices)
            for idx in mask_indices:
                self.mask_to_group[idx] = group_id
            
            print(f"Added masks {mask_indices} to existing group {group_id}")
        else:
            # Create new group
            group_id = f"group_{self.next_group_id}"
            self.next_group_id += 1
            
            self.mask_groups[group_id] = set(mask_indices)
            for idx in mask_indices:
                self.mask_to_group[idx] = group_id
            
            print(f"Created new group {group_id} with masks {mask_indices}")
    
    def _ungroup_masks(self, mask_indices: Set[int]):
        """Ungroup the specified masks."""
        groups_to_remove = set()
        
        for idx in mask_indices:
            if idx in self.mask_to_group:
                group_id = self.mask_to_group[idx]
                
                # Remove mask from group
                if group_id in self.mask_groups:
                    self.mask_groups[group_id].discard(idx)
                    
                    # If group is empty, mark for removal
                    if not self.mask_groups[group_id]:
                        groups_to_remove.add(group_id)
                
                # Remove mask from mapping
                del self.mask_to_group[idx]
        
        # Clean up empty groups
        for group_id in groups_to_remove:
            del self.mask_groups[group_id]
        
        print(f"Ungrouped masks {mask_indices}")
    
    def _merge_groups(self, target_group_id: str, source_group_id: str):
        """Merge source group into target group."""
        if source_group_id not in self.mask_groups or target_group_id not in self.mask_groups:
            return
        
        # Move all masks from source to target
        source_masks = self.mask_groups[source_group_id].copy()
        self.mask_groups[target_group_id].update(source_masks)
        
        # Update mask-to-group mapping
        for idx in source_masks:
            self.mask_to_group[idx] = target_group_id
        
        # Remove source group
        del self.mask_groups[source_group_id]
        
        print(f"Merged group {source_group_id} into {target_group_id}")
    
    def _update_group_button_label(self):
        """Update the group/ungroup button label based on current selection."""
        if not UIStateManager.safe_item_exists("group_ungroup_btn"):
            return
        
        if not self.selected_mask_indices:
            UIStateManager.safe_configure_item("group_ungroup_btn", label="Group")
            return
        
        # Check if all selected masks are in the same group
        groups_in_selection = set()
        for idx in self.selected_mask_indices:
            if idx in self.mask_to_group:
                groups_in_selection.add(self.mask_to_group[idx])
        
        if len(groups_in_selection) == 1 and len(self.selected_mask_indices) > 1:
            # All selected masks are in the same group
            UIStateManager.safe_configure_item("group_ungroup_btn", label="Ungroup")
        else:
            # Mixed or no grouping
            UIStateManager.safe_configure_item("group_ungroup_btn", label="Group")
    
    def _expand_group_selection(self, mask_index: int):
        """Expand selection to include all masks in the same group."""
        if mask_index not in self.mask_to_group:
            return
        
        group_id = self.mask_to_group[mask_index]
        if group_id not in self.mask_groups:
            return
        
        # Select all masks in the group
        group_masks = self.mask_groups[group_id]
        self.selected_mask_indices.update(group_masks)
        
        # Update UI selection state
        for idx in group_masks:
            if idx in self.mask_checkboxes:
                selectable_tag = self.mask_checkboxes[idx]
                UIStateManager.safe_set_value(selectable_tag, True)
        
        print(f"Expanded selection to include entire group {group_id}: {group_masks}")
    
    def _is_mask_grouped(self, mask_index: int) -> bool:
        """Check if a mask is part of a group."""
        return mask_index in self.mask_to_group
    
    def _save_group_parameters(self, group_masks: Set[int]):
        """Save parameters for all masks in a group."""
        try:
            current_params = self._get_current_parameters()
            curves_data = self._get_current_curves_data()
            
            # Save the same parameters for all masks in the group
            for mask_index in group_masks:
                self.mask_params[mask_index] = {
                    'parameters': current_params.copy(),
                    'curves': curves_data
                }
            
            print(f"💾 Saved group parameters for masks {group_masks}")
            
        except Exception as e:
            print(f"Error saving group parameters: {e}")
    
    def _update_mask_display_names(self):
        """Update mask display names to show group information."""
        if not self.main_window or not hasattr(self.main_window, 'mask_names'):
            return
        
        # Create updated display names that show group information
        display_names = []
        for idx in range(len(self.main_window.mask_names)):
            base_name = self.main_window.mask_names[idx]
            
            if idx in self.mask_to_group:
                group_id = self.mask_to_group[idx]
                # Extract group number from group_id (e.g., "group_0" -> "0")
                group_num = group_id.split('_')[-1] if '_' in group_id else group_id
                display_name = f"[G{group_num}] {base_name}"
            else:
                display_name = base_name
            
            display_names.append(display_name)
        
        # Update the UI if the mask table exists
        if UIStateManager.safe_item_exists("mask_table"):
            self._refresh_mask_table_display(display_names)
    
    def _refresh_mask_table_display(self, display_names: List[str]):
        """Refresh the mask table display with updated names."""
        try:
            # Update existing selectable items with new labels
            for idx, display_name in enumerate(display_names):
                selectable_tag = f"mask_selectable_{idx}"
                if UIStateManager.safe_item_exists(selectable_tag):
                    UIStateManager.safe_configure_item(selectable_tag, label=display_name)
            
            print(f"Updated mask table display with group indicators")
            
        except Exception as e:
            print(f"Error refreshing mask table display: {e}")
    
    def _cleanup_group_data_for_deleted_mask(self, mask_index: int):
        """Clean up group data when a mask is deleted."""
        if mask_index not in self.mask_to_group:
            return
        
        group_id = self.mask_to_group[mask_index]
        
        # Remove mask from group
        if group_id in self.mask_groups:
            self.mask_groups[group_id].discard(mask_index)
            
            # If group becomes empty or has only one mask, dissolve it
            if len(self.mask_groups[group_id]) <= 1:
                remaining_masks = list(self.mask_groups[group_id])
                for remaining_mask in remaining_masks:
                    if remaining_mask in self.mask_to_group:
                        del self.mask_to_group[remaining_mask]
                del self.mask_groups[group_id]
                print(f"Dissolved group {group_id} (insufficient masks)")
        
        # Remove from mask-to-group mapping
        if mask_index in self.mask_to_group:
            del self.mask_to_group[mask_index]
        
        # Adjust indices for masks after the deleted one
        self._adjust_group_indices_after_deletion(mask_index)
    
    def _adjust_group_indices_after_deletion(self, deleted_index: int):
        """Adjust group data indices after a mask deletion."""
        # Create new mappings with adjusted indices
        new_mask_to_group = {}
        new_mask_groups = {}
        
        for group_id, mask_set in self.mask_groups.items():
            adjusted_mask_set = set()
            for mask_idx in mask_set:
                if mask_idx > deleted_index:
                    adjusted_idx = mask_idx - 1
                    adjusted_mask_set.add(adjusted_idx)
                    new_mask_to_group[adjusted_idx] = group_id
                elif mask_idx < deleted_index:
                    adjusted_mask_set.add(mask_idx)
                    new_mask_to_group[mask_idx] = group_id
                # Skip the deleted index
            
            if adjusted_mask_set:
                new_mask_groups[group_id] = adjusted_mask_set
        
        self.mask_to_group = new_mask_to_group
        self.mask_groups = new_mask_groups
