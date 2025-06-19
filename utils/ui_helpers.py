"""
UI Helper utilities for DearPyGUI components.
Centralizes common UI patterns to reduce code duplication.
"""
import dearpygui.dearpygui as dpg
from typing import List, Dict, Any, Optional, Callable


class UIStateManager:
    """Centralized UI state management to reduce duplication."""
    
    @staticmethod
    def safe_item_exists(tag: str) -> bool:
        """Safely check if a DearPyGUI item exists."""
        return dpg.does_item_exist(tag)
    
    @staticmethod
    def safe_configure_item(tag: str, **kwargs) -> bool:
        """Safely configure a DearPyGUI item if it exists."""
        if dpg.does_item_exist(tag):
            try:
                dpg.configure_item(tag, **kwargs)
                return True
            except Exception as e:
                print(f"Error configuring item {tag}: {e}")
        return False
    
    @staticmethod
    def safe_set_value(tag: str, value: Any) -> bool:
        """Safely set value for a DearPyGUI item if it exists."""
        if dpg.does_item_exist(tag):
            try:
                dpg.set_value(tag, value)
                return True
            except Exception as e:
                print(f"Error setting value for {tag}: {e}")
        return False
    
    @staticmethod
    def safe_get_value(tag: str, default: Any = None) -> Any:
        """Safely get value from a DearPyGUI item."""
        if dpg.does_item_exist(tag):
            try:
                return dpg.get_value(tag)
            except Exception as e:
                print(f"Error getting value for {tag}: {e}")
        return default
    
    @staticmethod
    def toggle_controls_visibility(control_tags: List[str], show: bool) -> None:
        """Toggle visibility for multiple controls."""
        for tag in control_tags:
            UIStateManager.safe_configure_item(tag, show=show)
    
    @staticmethod
    def toggle_controls_enabled(control_tags: List[str], enabled: bool) -> None:
        """Toggle enabled state for multiple controls."""
        for tag in control_tags:
            UIStateManager.safe_configure_item(tag, enabled=enabled)
    
    @staticmethod
    def temporarily_disable_callback(tag: str, action: Callable) -> None:
        """Temporarily disable a callback, perform action, then restore it."""
        if not dpg.does_item_exist(tag):
            return
        
        original_callback = dpg.get_item_callback(tag)
        dpg.set_item_callback(tag, None)
        try:
            action()
        finally:
            dpg.set_item_callback(tag, original_callback)


class ControlGroupManager:
    """Manages groups of related UI controls."""
    
    def __init__(self):
        self.control_groups: Dict[str, List[str]] = {}
    
    def register_group(self, group_name: str, control_tags: List[str]) -> None:
        """Register a group of controls."""
        self.control_groups[group_name] = control_tags
    
    def show_group(self, group_name: str) -> None:
        """Show all controls in a group."""
        if group_name in self.control_groups:
            UIStateManager.toggle_controls_visibility(
                self.control_groups[group_name], True
            )
    
    def hide_group(self, group_name: str) -> None:
        """Hide all controls in a group."""
        if group_name in self.control_groups:
            UIStateManager.toggle_controls_visibility(
                self.control_groups[group_name], False
            )
    
    def enable_group(self, group_name: str) -> None:
        """Enable all controls in a group."""
        if group_name in self.control_groups:
            UIStateManager.toggle_controls_enabled(
                self.control_groups[group_name], True
            )
    
    def disable_group(self, group_name: str) -> None:
        """Disable all controls in a group."""
        if group_name in self.control_groups:
            UIStateManager.toggle_controls_enabled(
                self.control_groups[group_name], False
            )


class MaskOverlayManager:
    """Centralized mask overlay management."""
    
    @staticmethod
    def hide_all_overlays(mask_count: int, max_masks: int = 100) -> None:
        """Hide all mask overlays."""
        hidden_count = 0
        for idx in range(max_masks):
            series_tag = f"mask_series_{idx}"
            if UIStateManager.safe_configure_item(series_tag, show=False):
                hidden_count += 1
        if hidden_count > 0:
            print(f"Hidden {hidden_count} mask overlays")
    
    @staticmethod
    def show_overlay(mask_index: int) -> bool:
        """Show a specific mask overlay."""
        series_tag = f"mask_series_{mask_index}"
        return UIStateManager.safe_configure_item(series_tag, show=True)
    
    @staticmethod
    def show_selected_overlays(selected_indices: List[int]) -> None:
        """Show overlays for selected mask indices."""
        for idx in selected_indices:
            MaskOverlayManager.show_overlay(idx)
    
    @staticmethod
    def cleanup_overlay(mask_index: int) -> bool:
        """Clean up a specific mask overlay."""
        series_tag = f"mask_series_{mask_index}"
        if dpg.does_item_exist(series_tag):
            try:
                dpg.delete_item(series_tag)
                print(f"Deleted mask overlay {mask_index}")
                return True
            except Exception as e:
                print(f"Error deleting mask overlay {mask_index}: {e}")
        return False


# Convenience function for backward compatibility
def safe_item_check(tag: str) -> bool:
    """Safely check if a DearPyGUI item exists."""
    return UIStateManager.safe_item_exists(tag)


def setup_ui_theme():
    """Setup a consistent UI theme for the application."""
    try:
        # Create a dark theme
        with dpg.theme() as global_theme:
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_color(dpg.mvThemeCol_WindowBg, [37, 37, 38, 255])
                dpg.add_theme_color(dpg.mvThemeCol_ChildBg, [43, 43, 43, 255])
                dpg.add_theme_color(dpg.mvThemeCol_PopupBg, [50, 50, 50, 255])
                dpg.add_theme_color(dpg.mvThemeCol_FrameBg, [66, 66, 66, 255])
                dpg.add_theme_color(dpg.mvThemeCol_FrameBgHovered, [76, 76, 76, 255])
                dpg.add_theme_color(dpg.mvThemeCol_FrameBgActive, [86, 86, 86, 255])
                dpg.add_theme_color(dpg.mvThemeCol_Button, [70, 70, 70, 255])
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, [80, 80, 80, 255])
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, [90, 90, 90, 255])
                dpg.add_theme_color(dpg.mvThemeCol_Header, [50, 100, 150, 255])
                dpg.add_theme_color(dpg.mvThemeCol_HeaderHovered, [70, 120, 170, 255])
                dpg.add_theme_color(dpg.mvThemeCol_HeaderActive, [90, 140, 190, 255])
                dpg.add_theme_color(dpg.mvThemeCol_Text, [255, 255, 255, 255])
                dpg.add_theme_color(dpg.mvThemeCol_TextDisabled, [128, 128, 128, 255])
                
                # Style settings
                dpg.add_theme_style(dpg.mvStyleVar_WindowRounding, 5)
                dpg.add_theme_style(dpg.mvStyleVar_ChildRounding, 3)
                dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 3)
                dpg.add_theme_style(dpg.mvStyleVar_PopupRounding, 3)
                dpg.add_theme_style(dpg.mvStyleVar_ScrollbarRounding, 3)
                dpg.add_theme_style(dpg.mvStyleVar_GrabRounding, 3)
                dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 8, 8)
                dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 4, 3)
                dpg.add_theme_style(dpg.mvStyleVar_ItemSpacing, 4, 4)
        
        dpg.bind_theme(global_theme)
        
    except Exception as e:
        print(f"Error setting up theme: {e}")
