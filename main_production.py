#!/usr/bin/env python3
"""
Production Main Entry Point - Fully Refactored Image Editor
Uses service layer architecture with modular UI components.
"""

import dearpygui.dearpygui as dpg
import sys
import os
from pathlib import Path

# Add the project root to the path
sys.path.append(str(Path(__file__).parent))

from core.application import ApplicationService
from ui.main_window_production import ProductionMainWindow
from utils.memory_utils import setup_memory_optimization, MemoryManager
from utils.ui_helpers import setup_ui_theme


class ProductionImageEditor:
    """
    Production image editor using fully refactored architecture.
    
    This represents the completion of the refactoring project, replacing
    the original monolithic implementation with a clean, maintainable solution.
    """
    
    def __init__(self):
        # Initialize service layer
        self.app_service = ApplicationService()
        self.main_window = None
        self.memory_manager = MemoryManager()
        
        # Setup memory optimization
        setup_memory_optimization()
    
    def initialize(self):
        """Initialize the application."""
        print("🚀 Initializing Production Image Editor")
        print("✓ Service layer architecture")
        print("✓ Modular UI components")
        print("✓ Memory optimization enabled")
        
        # Create DearPyGUI context
        dpg.create_context()
        
        # Setup UI theme
        setup_ui_theme()
        
        # Create main window
        self.main_window = ProductionMainWindow(self.app_service)
        self.main_window.setup()
        
        # Important: Set main window reference on app service for EventHandlers
        self.app_service.main_window = self.main_window
        
        # Setup file dialogs
        self._setup_file_dialogs()
        
        # Configure viewport
        dpg.create_viewport(
            title='Image Editor - Production Version',
            width=1400,
            height=900,
            resizable=True,
            vsync=True
        )
        
        # Set viewport resize callback
        dpg.set_viewport_resize_callback(self._on_viewport_resize)
        
        dpg.setup_dearpygui()
        dpg.set_primary_window("main_window", True)
        
        print("✓ Initialization complete")
    
    def _setup_file_dialogs(self):
        """Setup file dialogs for open/save operations."""
        # Open dialog
        with dpg.file_dialog(
            directory_selector=False,
            show=False,
            callback=self._file_open_callback,
            tag="file_open_dialog",
            width=700,
            height=400
        ):
            dpg.add_file_extension(".png")
            dpg.add_file_extension(".jpg")
            dpg.add_file_extension(".jpeg")
            dpg.add_file_extension(".bmp")
            dpg.add_file_extension(".tiff")
            dpg.add_file_extension(".tif")
        
        # Save dialog
        with dpg.file_dialog(
            directory_selector=False,
            show=False,
            callback=self._file_save_callback,
            tag="file_save_dialog",
            width=700,
            height=400,
            default_filename="processed_image.jpg"
        ):
            dpg.add_file_extension(".jpg")
            dpg.add_file_extension(".png")
    
    def _file_open_callback(self, sender, app_data, user_data):
        """Handle file open dialog."""
        file_path = app_data.get("file_path_name")
        if file_path:
            print(f"Loading image: {file_path}")
            success = self._load_image_with_crop_ui(file_path)
            if success:
                print(f"✓ Image loaded successfully")
            else:
                print(f"✗ Failed to load image")
    
    def _load_image_with_crop_ui(self, file_path: str) -> bool:
        """Load image and create CropRotateUI following original pattern."""
        try:
            # Load image through service
            image = self.app_service.load_image(file_path)
            if image is None:
                return False
            
            # Create CropRotateUI like original main.py
            from ui.crop_rotate import CropRotateUI
            from processing.image_processor import ImageProcessor
            import cv2
            import numpy as np
            
            # Create processor and CropRotateUI
            processor = ImageProcessor(image.copy())
            crop_rotate_ui = CropRotateUI(image, processor)
            
            # Update CropRotateUI to use the production main window's axis tags
            crop_rotate_ui.x_axis_tag = "main_x_axis"
            crop_rotate_ui.y_axis_tag = "main_y_axis"
            crop_rotate_ui.panel_id = "central_panel"  # Use production panel tag
            
            # Create texture following original pattern
            with dpg.texture_registry():
                gray_background = np.full((crop_rotate_ui.texture_h, crop_rotate_ui.texture_w, 4), 
                                         [37,37,38,255], dtype=np.uint8)
                offset_x = (crop_rotate_ui.texture_w - crop_rotate_ui.orig_w) // 2
                offset_y = (crop_rotate_ui.texture_h - crop_rotate_ui.orig_h) // 2
                
                if image.shape[2] == 3:
                    image_rgba = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)
                else:
                    image_rgba = image
                gray_background[offset_y:offset_y + crop_rotate_ui.orig_h, 
                                offset_x:offset_x + crop_rotate_ui.orig_w] = image_rgba
            
            # Use raw texture like original (for performance)
            if dpg.does_item_exist(crop_rotate_ui.texture_tag):
                dpg.set_value(crop_rotate_ui.texture_tag, gray_background.flatten().astype(np.float32) / 255.0)
            else:
                with dpg.texture_registry():
                    dpg.add_raw_texture(crop_rotate_ui.texture_w, crop_rotate_ui.texture_h,
                                        gray_background.flatten().astype(np.float32) / 255.0,
                                        tag=crop_rotate_ui.texture_tag, format=dpg.mvFormat_Float_rgba)
            
            crop_rotate_ui.offset_x = offset_x
            crop_rotate_ui.offset_y = offset_y
            
            # Connect to main window following original pattern
            self.main_window.set_crop_rotate_ui(crop_rotate_ui)
            # Only reset axis limits for initial image load, allow free panning afterward
            if not hasattr(crop_rotate_ui, '_axis_limits_initialized'):
                crop_rotate_ui.update_axis_limits()  # Don't force, just fit data
            
            # Update histogram with the newly loaded image
            if self.main_window and self.main_window.tool_panel:
                # Update histogram if available
                try:
                    self.main_window.tool_panel.update_histogram(image)
                except AttributeError:
                    pass  # Histogram update not available in production version
            
            return True
            
        except Exception as e:
            print(f"Error loading image with crop UI: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _file_save_callback(self, sender, app_data, user_data):
        """Handle file save dialog."""
        file_path = app_data.get("file_path_name")
        if file_path:
            print(f"Saving image: {file_path}")
            success = self.app_service.save_current_image(file_path)
            if success:
                print(f"✓ Image saved successfully")
            else:
                print(f"✗ Failed to save image")
    
    def show_open_dialog(self):
        """Show the open file dialog."""
        dpg.show_item("file_open_dialog")
    
    def show_save_dialog(self):
        """Show the save file dialog."""
        dpg.show_item("file_save_dialog")
    
    def _on_viewport_resize(self):
        """Handle viewport resize events."""
        if self.main_window:
            self.main_window.handle_resize()
    
    def run(self):
        """Run the image editor."""
        try:
            print("🎯 Starting Production Image Editor")
            
            # Show viewport and start main loop
            dpg.show_viewport()
            # Trigger initial resize
            if self.main_window:
                self.main_window.handle_resize()
                
                # Additional resize for tool panel components
                if (hasattr(self.main_window, 'tool_panel') and 
                    self.main_window.tool_panel and 
                    hasattr(self.main_window.tool_panel, 'handle_resize')):
                    self.main_window.tool_panel.handle_resize()
            dpg.start_dearpygui()
            
        except KeyboardInterrupt:
            print("\n⚠️  Interrupted by user")
        except Exception as e:
            print(f"❌ Runtime error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Cleanup application resources."""
        try:
            print("🧹 Cleaning up...")
            
            if self.main_window:
                self.main_window.cleanup()
            
            self.app_service.cleanup()
            self.memory_manager.cleanup_gpu_memory()
            
            dpg.destroy_context()
            print("✓ Cleanup completed")
            
        except Exception as e:
            print(f"⚠️  Cleanup error: {e}")


def main():
    """Main entry point."""
    print("=" * 60)
    print("🎯 IMAGE EDITOR - PRODUCTION VERSION")
    print("   Fully Refactored Service Layer Architecture")
    print("=" * 60)
    
    # Create and run the editor
    editor = ProductionImageEditor()
    editor.initialize()
    editor.run()


if __name__ == "__main__":
    main()
