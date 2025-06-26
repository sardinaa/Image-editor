"""
ExportService - Centralized Export Functionality Service
Extracts export dialog and file handling logic from ProductionMainWindow to complete service-oriented architecture.
"""

import dearpygui.dearpygui as dpg
import cv2
import numpy as np
import os
import threading
from typing import Optional, Dict, Any

from utils.ui_helpers import safe_item_check


class ExportService:
    """
    Centralized service for handling image export operations and UI.
    
    Extracts export dialog creation and management logic from the UI layer,
    providing a clean separation between UI and export business logic.
    """
    
    def __init__(self, app_service):
        """Initialize the ExportService with service dependencies."""
        self.app_service = app_service
        
        # Export dialog state
        self._export_modal_created = False
        self._export_settings = {
            'filename': 'exported_image',
            'format': 'PNG',
            'quality': 95,
            'path': os.path.expanduser("~")
        }
        
    def show_export_dialog(self):
        """Show the export image modal dialog."""
        if safe_item_check("export_modal"):
            dpg.show_item("export_modal")
        else:
            self._create_export_modal()
            dpg.show_item("export_modal")
    
    def _create_export_modal(self):
        """Create the comprehensive export modal dialog."""
        if self._export_modal_created:
            return
            
        # Create directory selection dialog at top level (not inside modal)
        with dpg.file_dialog(
            directory_selector=True,
            show=False,
            callback=self._export_path_callback,
            cancel_callback=self._export_path_cancel_callback,
            tag="export_path_dialog",
            width=600,
            height=400,
            modal=True
        ):
            pass
        
        with dpg.window(
            label="Export Image",
            tag="export_modal",
            modal=True,
            show=False,
            width=500,
            height=400,
            no_resize=True
        ):
            dpg.add_text("Export Settings", color=[176, 204, 255])
            dpg.add_separator()
            dpg.add_spacer(height=10)
            
            # File name section
            with dpg.group(horizontal=True):
                dpg.add_text("File Name:")
                dpg.add_spacer(width=20)
                dpg.add_input_text(
                    tag="export_filename",
                    default_value=self._export_settings['filename'],
                    width=300,
                    hint="Enter filename without extension",
                    callback=self._update_export_preview
                )
            
            dpg.add_spacer(height=10)
            
            # File type section
            with dpg.group(horizontal=True):
                dpg.add_text("Format:")
                dpg.add_spacer(width=43)
                dpg.add_combo(
                    tag="export_format",
                    items=["PNG", "JPEG", "BMP", "TIFF"],
                    default_value=self._export_settings['format'],
                    width=150,
                    callback=self._on_format_change
                )
            
            dpg.add_spacer(height=10)
            
            # Quality section (initially hidden for PNG)
            with dpg.group(tag="quality_group", show=False):
                with dpg.group(horizontal=True):
                    dpg.add_text("Quality:")
                    dpg.add_spacer(width=34)
                    dpg.add_slider_int(
                        tag="export_quality",
                        default_value=self._export_settings['quality'],
                        min_value=1,
                        max_value=100,
                        width=200,
                        format="%d%%",
                        callback=self._update_export_preview
                    )
                dpg.add_spacer(height=10)
            
            # Path selection section
            with dpg.group(horizontal=True):
                dpg.add_text("Save to:")
                dpg.add_spacer(width=30)
                dpg.add_input_text(
                    tag="export_path",
                    default_value=self._export_settings['path'],
                    width=250,
                    readonly=True
                )
                dpg.add_button(
                    label="Browse...",
                    width=70,
                    callback=self._browse_export_path
                )
            
            dpg.add_spacer(height=20)
            dpg.add_separator()
            dpg.add_spacer(height=10)
            
            # Preview information
            dpg.add_text("Preview:", color=[200, 200, 200])
            dpg.add_text("", tag="export_preview", wrap=400)
            
            dpg.add_spacer(height=20)
            
            # Buttons
            with dpg.group(horizontal=True):
                dpg.add_spacer(width=250)
                dpg.add_button(
                    label="Cancel",
                    width=70,
                    callback=lambda: dpg.hide_item("export_modal")
                )
                dpg.add_spacer(width=10)
                dpg.add_button(
                    label="Export",
                    width=70,
                    callback=self._perform_export
                )
        
        # Update preview initially
        self._update_export_preview()
        self._export_modal_created = True
    
    def _on_format_change(self, sender, app_data, user_data):
        """Handle format change in export dialog."""
        format_type = dpg.get_value("export_format")
        
        # Show/hide quality slider based on format
        if format_type == "JPEG":
            dpg.configure_item("quality_group", show=True)
        else:
            dpg.configure_item("quality_group", show=False)
        
        self._export_settings['format'] = format_type
        self._update_export_preview()
    
    def _browse_export_path(self):
        """Open path browser for export location."""
        print("📂 Browse export path clicked")
        
        # Temporarily hide the export modal to avoid layering issues
        if dpg.does_item_exist("export_modal"):
            dpg.hide_item("export_modal")
            print("🔄 Hid export modal")
        
        # Show the path dialog
        if dpg.does_item_exist("export_path_dialog"):
            dpg.show_item("export_path_dialog")
            print("🔄 Showed path dialog")
        else:
            print("❌ Export path dialog does not exist")
    
    def _export_path_callback(self, sender, app_data, user_data):
        """Handle export path selection."""
        print(f"📁 Export path callback triggered")
        print(f"   Sender: {sender}")
        print(f"   App data: {app_data}")
        print(f"   User data: {user_data}")
        
        # Hide the path dialog first
        if dpg.does_item_exist("export_path_dialog"):
            dpg.hide_item("export_path_dialog")
        
        # Try to extract path from different possible data structures
        selected_path = ""
        if isinstance(app_data, dict):
            # Try different keys that DearPyGUI might use
            selected_path = app_data.get("file_path_name", "")
            if not selected_path:
                selected_path = app_data.get("current_path", "")
            if not selected_path:
                selected_path = app_data.get("file_name", "")
        elif isinstance(app_data, str):
            # Direct string path
            selected_path = app_data
        
        if selected_path:
            print(f"✅ Selected path: {selected_path}")
            if dpg.does_item_exist("export_path"):
                dpg.set_value("export_path", selected_path)
            self._export_settings['path'] = selected_path
            self._update_export_preview()
        else:
            print("❌ No path selected or path extraction failed")
        
        # Use a small delay to ensure proper order of operations
        def show_modal_after_delay():
            if dpg.does_item_exist("export_modal"):
                dpg.show_item("export_modal")
                print("🔄 Showed export modal again (delayed)")
            else:
                print("❌ Export modal does not exist (delayed check)")
        
        # Schedule the modal to show after a small delay
        timer = threading.Timer(0.1, show_modal_after_delay)
        timer.start()
    
    def _export_path_cancel_callback(self, sender, app_data, user_data):
        """Handle export path dialog cancellation."""
        print("🚫 Export path dialog cancelled")
        
        # Hide the path dialog
        if dpg.does_item_exist("export_path_dialog"):
            dpg.hide_item("export_path_dialog")
        
        # Use a small delay to ensure proper order of operations  
        def show_modal_after_delay():
            if dpg.does_item_exist("export_modal"):
                dpg.show_item("export_modal")
                print("🔄 Showed export modal again after cancel (delayed)")
            else:
                print("❌ Export modal does not exist (delayed check)")
        
        # Schedule the modal to show after a small delay
        timer = threading.Timer(0.1, show_modal_after_delay)
        timer.start()
    
    def _update_export_preview(self):
        """Update the export preview text."""
        try:
            filename = dpg.get_value("export_filename")
            format_type = dpg.get_value("export_format")
            path = dpg.get_value("export_path")
            
            if not filename:
                filename = self._export_settings['filename']
            
            extension = {
                "PNG": ".png",
                "JPEG": ".jpg", 
                "BMP": ".bmp",
                "TIFF": ".tif"
            }.get(format_type, ".png")
            
            full_path = os.path.join(path, filename + extension)
            
            preview_text = f"File: {filename}{extension}\nLocation: {path}\nFull path: {full_path}"
            
            if format_type == "JPEG":
                quality = dpg.get_value("export_quality")
                preview_text += f"\nQuality: {quality}%"
            
            dpg.set_value("export_preview", preview_text)
            
            # Update internal settings
            self._export_settings.update({
                'filename': filename,
                'format': format_type,
                'path': path
            })
            
        except Exception as e:
            dpg.set_value("export_preview", f"Preview error: {e}")
    
    def _perform_export(self):
        """Perform the actual image export operation."""
        try:
            # Get current processed image from application service
            current_image = self.app_service.image_service.get_processed_image()
            if current_image is None:
                dpg.set_value("export_preview", "Error: No image to export")
                return
            
            # Get export settings
            filename = dpg.get_value("export_filename").strip()
            format_type = dpg.get_value("export_format")
            path = dpg.get_value("export_path")
            
            if not filename:
                dpg.set_value("export_preview", "Error: Please enter a filename")
                return
            
            # Create file extension mapping
            extension = {
                "PNG": ".png",
                "JPEG": ".jpg",
                "BMP": ".bmp", 
                "TIFF": ".tif"
            }.get(format_type, ".png")
            
            # Build full file path
            full_path = os.path.join(path, filename + extension)
            
            # Check if directory exists
            if not os.path.exists(path):
                dpg.set_value("export_preview", f"Error: Directory does not exist: {path}")
                return
            
            # Prepare image for saving
            save_image = current_image.copy()
            
            # Convert color format for OpenCV (RGB to BGR)
            if len(save_image.shape) == 3:
                if save_image.shape[2] == 3:
                    save_image = cv2.cvtColor(save_image, cv2.COLOR_RGB2BGR)
                elif save_image.shape[2] == 4:
                    save_image = cv2.cvtColor(save_image, cv2.COLOR_RGBA2BGRA)
            
            # Set up save parameters based on format
            save_params = []
            if format_type == "JPEG":
                quality = dpg.get_value("export_quality")
                save_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
            elif format_type == "PNG":
                save_params = [cv2.IMWRITE_PNG_COMPRESSION, 6]  # Good compression
            
            # Save the image
            success = cv2.imwrite(full_path, save_image, save_params)
            
            if success:
                print(f"✓ Image exported successfully to: {full_path}")
                dpg.set_value("export_preview", f"✓ Successfully exported to:\n{full_path}")
                
                # Close modal after successful export
                self._close_export_modal()
                
            else:
                dpg.set_value("export_preview", f"✗ Failed to export image to:\n{full_path}")
                print(f"✗ Failed to export image to: {full_path}")
                
        except Exception as e:
            error_msg = f"Export error: {str(e)}"
            dpg.set_value("export_preview", error_msg)
            print(f"Export error: {e}")
            import traceback
            traceback.print_exc()
    
    def _close_export_modal(self):
        """Close the export modal dialog."""
        if dpg.does_item_exist("export_modal"):
            dpg.hide_item("export_modal")
    
    def get_export_settings(self) -> Dict[str, Any]:
        """Get current export settings."""
        return self._export_settings.copy()
    
    def set_export_settings(self, settings: Dict[str, Any]):
        """Set export settings programmatically."""
        self._export_settings.update(settings)
        
        # Update UI if modal exists
        if dpg.does_item_exist("export_filename"):
            dpg.set_value("export_filename", self._export_settings['filename'])
        if dpg.does_item_exist("export_format"):
            dpg.set_value("export_format", self._export_settings['format'])
        if dpg.does_item_exist("export_quality"):
            dpg.set_value("export_quality", self._export_settings['quality'])
        if dpg.does_item_exist("export_path"):
            dpg.set_value("export_path", self._export_settings['path'])
    
    def export_current_image(self, filename: str = None, format_type: str = None, 
                           path: str = None, quality: int = None) -> bool:
        """
        Export current image programmatically without showing dialog.
        
        Args:
            filename: Target filename (without extension)
            format_type: Image format ("PNG", "JPEG", "BMP", "TIFF")
            path: Target directory path
            quality: JPEG quality (1-100)
            
        Returns:
            bool: True if export successful, False otherwise
        """
        try:
            # Get current processed image
            current_image = self.app_service.image_service.get_processed_image()
            if current_image is None:
                print("Error: No image to export")
                return False
            
            # Use provided values or defaults
            filename = filename or self._export_settings['filename']
            format_type = format_type or self._export_settings['format']
            path = path or self._export_settings['path']
            quality = quality or self._export_settings['quality']
            
            # Create file extension mapping
            extension = {
                "PNG": ".png",
                "JPEG": ".jpg",
                "BMP": ".bmp", 
                "TIFF": ".tif"
            }.get(format_type, ".png")
            
            # Build full file path
            full_path = os.path.join(path, filename + extension)
            
            # Check if directory exists
            if not os.path.exists(path):
                print(f"Error: Directory does not exist: {path}")
                return False
            
            # Prepare image for saving
            save_image = current_image.copy()
            
            # Convert color format for OpenCV (RGB to BGR)
            if len(save_image.shape) == 3:
                if save_image.shape[2] == 3:
                    save_image = cv2.cvtColor(save_image, cv2.COLOR_RGB2BGR)
                elif save_image.shape[2] == 4:
                    save_image = cv2.cvtColor(save_image, cv2.COLOR_RGBA2BGRA)
            
            # Set up save parameters based on format
            save_params = []
            if format_type == "JPEG":
                save_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
            elif format_type == "PNG":
                save_params = [cv2.IMWRITE_PNG_COMPRESSION, 6]
            
            # Save the image
            success = cv2.imwrite(full_path, save_image, save_params)
            
            if success:
                print(f"✓ Image exported successfully to: {full_path}")
                return True
            else:
                print(f"✗ Failed to export image to: {full_path}")
                return False
                
        except Exception as e:
            print(f"Export error: {e}")
            return False
    
    def cleanup(self):
        """Cleanup ExportService resources."""
        # Clean up any dialogs
        if dpg.does_item_exist("export_modal"):
            dpg.delete_item("export_modal")
        if dpg.does_item_exist("export_path_dialog"):
            dpg.delete_item("export_path_dialog")
            
        self._export_modal_created = False
        self.app_service = None
