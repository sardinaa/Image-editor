"""
Display Service for Image Editor

Handles all display-related operations including texture management,
axis coordination, viewport calculations, and image series management.
Extracted from ProductionMainWindow as part of Phase 3 architectural cleanup.
"""

import numpy as np
import cv2
import dearpygui.dearpygui as dpg
from typing import Tuple, Optional
import traceback

from utils.ui_helpers import safe_item_check


class DisplayService:
    """
    Service responsible for managing image display, textures, and viewport coordination.
    
    Centralizes all display logic that was previously scattered in the UI layer,
    providing a clean interface for image rendering and viewport management.
    """
    
    def __init__(self):
        """Initialize the display service."""
        self.current_texture_tag: Optional[str] = None
        self.current_texture_width: int = 0
        self.current_texture_height: int = 0
        self.orig_w: int = 0
        self.orig_h: int = 0
        self.image_offset_x: int = 0
        self.image_offset_y: int = 0
        self._updating_display = False
        
    def update_image_display(self, processed_image: np.ndarray, crop_rotate_ui=None) -> bool:
        """
        Update the image display with current processed image.
        
        Args:
            processed_image: The processed image to display
            crop_rotate_ui: Optional CropRotateUI instance for crop mode handling
            
        Returns:
            bool: True if update successful, False otherwise
        """
        if self._updating_display:
            return False
        
        self._updating_display = True
        try:
            if processed_image is None:
                return False
            
            # Check if crop mode is active - if so, let CropRotateUI handle texture updates
            crop_mode_active = safe_item_check("crop_mode") and dpg.get_value("crop_mode")
            if crop_mode_active and crop_rotate_ui:
                # Update the CropRotateUI's original image first
                crop_rotate_ui.original_image = processed_image.copy()
                # Let CropRotateUI handle the texture update
                crop_rotate_ui.update_image(None, None, None)
                return True
            
            # Use our own texture update for non-crop modes
            return self.update_texture(processed_image)
            
        except Exception as e:
            print(f"Error updating image display: {e}")
            traceback.print_exc()
            return False
        finally:
            self._updating_display = False
    
    def update_texture(self, image: np.ndarray, x_axis_tag: str = "x_axis", y_axis_tag: str = "y_axis") -> bool:
        """
        Update image texture using optimized pattern from CropRotateUI.
        
        Args:
            image: Image array to create texture from
            x_axis_tag: Tag for X axis
            y_axis_tag: Tag for Y axis
            
        Returns:
            bool: True if successful, False otherwise
        """
        if self._updating_display:
            return False
            
        try:
            # Prepare image data following optimized pattern
            height, width = image.shape[:2]
            
            # Store original image dimensions for axis calculation
            self.orig_w = width
            self.orig_h = height
            
            # Use dynamic texture dimensions based on image size
            # Scale down large images to reasonable texture size while maintaining aspect ratio
            max_texture_dimension = 1024
            scale_factor = min(max_texture_dimension / width, max_texture_dimension / height, 1.0)
            
            texture_width = max(int(width * scale_factor), 400)
            texture_height = max(int(height * scale_factor), 300)
            
            # Ensure texture can contain the scaled image
            if scale_factor < 1.0:
                # Image was scaled down, use scaled dimensions
                scaled_width = int(width * scale_factor)
                scaled_height = int(height * scale_factor)
                image = cv2.resize(image, (scaled_width, scaled_height))
                width, height = scaled_width, scaled_height
            
            # Create gray background like original implementation
            gray_background = np.full((texture_height, texture_width, 4), 
                                     [37,37,38,255], dtype=np.uint8)
            
            # Calculate centering offset
            offset_x = (texture_width - width) // 2
            offset_y = (texture_height - height) // 2
            
            # Store offsets for axis calculation
            self.image_offset_x = offset_x
            self.image_offset_y = offset_y
            
            # Convert to RGBA if needed
            if len(image.shape) == 2:
                # Grayscale to RGB then RGBA
                image_rgba = cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_GRAY2RGB), cv2.COLOR_RGB2RGBA)
            elif image.shape[2] == 3:
                # RGB to RGBA
                image_rgba = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)
            else:
                # Already RGBA
                image_rgba = image
            
            # Place image on background (centered)
            end_y = min(offset_y + image_rgba.shape[0], texture_height)
            end_x = min(offset_x + image_rgba.shape[1], texture_width)
            img_end_y = end_y - offset_y
            img_end_x = end_x - offset_x
            
            if offset_y >= 0 and offset_x >= 0:
                gray_background[offset_y:end_y, offset_x:end_x] = image_rgba[:img_end_y, :img_end_x]
            
            # Convert to float32 and normalize
            texture_data = gray_background.flatten().astype(np.float32) / 255.0
            
            # Use original pattern: update existing texture or create new one
            texture_tag = "main_display_texture"
            if dpg.does_item_exist(texture_tag):
                # Update existing texture data
                dpg.set_value(texture_tag, texture_data)
            else:
                # Create new raw texture
                with dpg.texture_registry():
                    dpg.add_raw_texture(
                        texture_width, texture_height,
                        texture_data,
                        tag=texture_tag,
                        format=dpg.mvFormat_Float_rgba
                    )
                
                # Create image series
                if dpg.does_item_exist(y_axis_tag):
                    dpg.add_image_series(
                        texture_tag,
                        bounds_min=[0, 0],
                        bounds_max=[texture_width, texture_height],
                        parent=y_axis_tag,
                        tag="main_image_series"
                    )
            
            # Store current texture info
            self.current_texture_tag = texture_tag
            self.current_texture_width = texture_width
            self.current_texture_height = texture_height
            
            # Update axis limits to center on actual image
            self.update_axis_limits_to_image(x_axis_tag, y_axis_tag)
            
            return True
            
        except Exception as e:
            print(f"Error updating texture: {e}")
            traceback.print_exc()
            return False
    
    def update_axis_limits_to_image(self, x_axis_tag: str, y_axis_tag: str, 
                                   image_plot_tag: str = "image_plot") -> None:
        """
        Update axis limits to center on the actual image.
        
        Args:
            x_axis_tag: Tag for X axis
            y_axis_tag: Tag for Y axis  
            image_plot_tag: Tag for image plot
        """
        try:
            if not hasattr(self, 'orig_w') or not hasattr(self, 'orig_h'):
                # Fallback to simple axis limits if we don't have image dimensions
                if safe_item_check(x_axis_tag):
                    dpg.set_axis_limits(x_axis_tag, 0, self.current_texture_width)
                if safe_item_check(y_axis_tag):
                    dpg.set_axis_limits(y_axis_tag, 0, self.current_texture_height)
                return
            
            # Get image and texture dimensions
            orig_w = self.orig_w
            orig_h = self.orig_h
            texture_w = self.current_texture_width
            texture_h = self.current_texture_height
            image_offset_x = getattr(self, 'image_offset_x', 0)
            image_offset_y = getattr(self, 'image_offset_y', 0)
            
            # Get current plot dimensions
            plot_width = dpg.get_item_width(image_plot_tag) if dpg.does_item_exist(image_plot_tag) else 800
            plot_height = dpg.get_item_height(image_plot_tag) if dpg.does_item_exist(image_plot_tag) else 600
            
            if plot_width <= 0 or plot_height <= 0:
                plot_width, plot_height = 800, 600  # Fallback dimensions
            
            # Calculate aspect ratios
            plot_aspect = plot_width / plot_height
            image_aspect = orig_w / orig_h
            
            # Calculate how to fit the actual image within the plot while maintaining aspect ratio
            padding_factor = 1.05  # 5% padding around the image
            
            if image_aspect > plot_aspect:
                # Image is wider relative to plot - fit to plot width
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
                display_height = orig_h * padding_factor
                display_width = display_height * plot_aspect
                
                # Center the view on the actual image
                x_center = image_offset_x + orig_w / 2
                y_center = image_offset_y + orig_h / 2
                x_min = x_center - display_width / 2
                x_max = x_center + display_width / 2
                y_min = y_center - display_height / 2
                y_max = y_center + display_height / 2
            
            # Set the axis limits to center on the actual image
            if safe_item_check(x_axis_tag):
                dpg.fit_axis_data(x_axis_tag)
            
            if safe_item_check(y_axis_tag):
                dpg.fit_axis_data(y_axis_tag)
                
        except Exception as e:
            print(f"Error updating axis limits: {e}")
            traceback.print_exc()
    
    def create_image_series(self, crop_rotate_ui, image_plot_tag: str, y_axis_tag: str) -> bool:
        """
        Create the image series to display the loaded image.
        
        Args:
            crop_rotate_ui: CropRotateUI instance
            image_plot_tag: Tag for image plot
            y_axis_tag: Tag for Y axis
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not crop_rotate_ui or dpg.does_item_exist("central_image"):
            return False
            
        try:
            # Get the y_axis from the plot to add the image series
            plot_children = dpg.get_item_children(image_plot_tag, slot=1)
            if not plot_children or len(plot_children) < 2:
                print("Error: Could not find y_axis in plot")
                return False
                
            y_axis = plot_children[1]  # Second child is the y_axis
            
            # Create image series that shows the full texture
            dpg.add_image_series(
                crop_rotate_ui.texture_tag,
                bounds_min=[0, 0],
                bounds_max=[crop_rotate_ui.texture_w, crop_rotate_ui.texture_h],
                parent=y_axis,
                tag="central_image"
            )
            
            return True
            
        except Exception as e:
            print(f"Error creating image series: {e}")
            traceback.print_exc()
            return False
    
    def update_axis_limits(self, crop_rotate_ui, x_axis_tag: str, y_axis_tag: str, 
                          image_plot_tag: str, initial: bool = False) -> None:
        """
        Update axis limits to properly display the image.
        
        Args:
            crop_rotate_ui: CropRotateUI instance
            x_axis_tag: Tag for X axis
            y_axis_tag: Tag for Y axis
            image_plot_tag: Tag for image plot
            initial: Whether this is initial setup
        """
        if not crop_rotate_ui:
            return
            
        try:
            # Don't override user's pan/zoom unless this is initial setup
            if not initial and dpg.does_item_exist(image_plot_tag):
                return
                
            # Get image dimensions
            orig_w = crop_rotate_ui.orig_w
            orig_h = crop_rotate_ui.orig_h
            texture_w = crop_rotate_ui.texture_w
            texture_h = crop_rotate_ui.texture_h
            
            # Calculate where the image is positioned within the texture (centered)
            image_offset_x = (texture_w - orig_w) // 2
            image_offset_y = (texture_h - orig_h) // 2
            
            # Get plot dimensions for aspect ratio calculation
            plot_width = dpg.get_item_width(image_plot_tag) if dpg.does_item_exist(image_plot_tag) else 800
            plot_height = dpg.get_item_height(image_plot_tag) if dpg.does_item_exist(image_plot_tag) else 600
            
            if plot_width <= 0 or plot_height <= 0:
                plot_width, plot_height = 800, 600  # Fallback dimensions
            
            # Calculate aspect ratios
            plot_aspect = plot_width / plot_height
            image_aspect = orig_w / orig_h
            
            # Calculate display dimensions with padding
            padding_factor = 1.05  # 5% padding around the image
            
            if image_aspect > plot_aspect:
                # Image is wider - fit to plot width
                display_width = orig_w * padding_factor
                display_height = display_width / plot_aspect
            else:
                # Image is taller - fit to plot height
                display_height = orig_h * padding_factor
                display_width = display_height * plot_aspect
            
            # Center the view on the actual image
            x_center = image_offset_x + orig_w / 2
            y_center = image_offset_y + orig_h / 2
            x_min = x_center - display_width / 2
            x_max = x_center + display_width / 2
            y_min = y_center - display_height / 2
            y_max = y_center + display_height / 2
            
            # Set axis limits
            dpg.fit_axis_data(x_axis_tag)
            dpg.fit_axis_data(y_axis_tag)
            
        except Exception as e:
            print(f"Error updating axis limits: {e}")
    
    def get_texture_info(self) -> dict:
        """
        Get current texture information.
        
        Returns:
            dict: Current texture information
        """
        return {
            'texture_tag': self.current_texture_tag,
            'texture_width': self.current_texture_width,
            'texture_height': self.current_texture_height,
            'orig_w': self.orig_w,
            'orig_h': self.orig_h,
            'image_offset_x': self.image_offset_x,
            'image_offset_y': self.image_offset_y
        }
    
    def cleanup(self) -> None:
        """Clean up display service resources."""
        try:
            # Clean up textures
            if self.current_texture_tag and dpg.does_item_exist(self.current_texture_tag):
                dpg.delete_item(self.current_texture_tag)
            
            # Clean up image series
            if dpg.does_item_exist("main_image_series"):
                dpg.delete_item("main_image_series")
            if dpg.does_item_exist("central_image"):
                dpg.delete_item("central_image")
            
            # Reset state
            self.current_texture_tag = None
            self.current_texture_width = 0
            self.current_texture_height = 0
            self.orig_w = 0
            self.orig_h = 0
            self.image_offset_x = 0
            self.image_offset_y = 0
            self._updating_display = False
            
            print("✓ DisplayService cleanup completed")
            
        except Exception as e:
            print(f"⚠️  Error during DisplayService cleanup: {e}")
