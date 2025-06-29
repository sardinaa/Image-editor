import dearpygui.dearpygui as dpg
import numpy as np
import cv2
import time
import traceback
from typing import List, Dict, Any, Optional


class MaskOverlayRenderer:
    """
    Handles rendering mask overlays on the image display.
    
    This class encapsulates all mask visualization logic that was previously
    scattered throughout the UI layer, providing clean separation of concerns.
    """
    
    def __init__(self, x_axis_tag: str, y_axis_tag: str):
        """
        Initialize the mask overlay renderer.
        
        Args:
            x_axis_tag: Tag for the x-axis of the plot
            y_axis_tag: Tag for the y-axis of the plot
        """
        self.x_axis_tag = x_axis_tag
        self.y_axis_tag = y_axis_tag
        
        # Color palette for different masks (RGBA format)
        self.colors = [
            [255, 0, 0, 100],     # Red
            [0, 255, 0, 100],     # Green  
            [0, 0, 255, 100],     # Blue
            [255, 255, 0, 100],   # Yellow
            [255, 0, 255, 100],   # Magenta
            [0, 255, 255, 100],   # Cyan
            [255, 128, 0, 100],   # Orange
            [128, 0, 255, 100],   # Purple
            [255, 192, 203, 100], # Pink
            [0, 128, 128, 100],   # Teal
        ]
    
    def update_mask_overlays(self, masks: List[Dict[str, Any]], crop_rotate_ui) -> None:
        """
        Update the visual mask overlays on the image.
        
        Args:
            masks: List of mask data dictionaries
            crop_rotate_ui: Reference to crop/rotate UI for dimensions and rotation
        """
        if not masks or not crop_rotate_ui:
            return
        
        # Get texture dimensions and rotation state
        render_context = self._prepare_render_context(crop_rotate_ui)
        
        # Clean up old overlays
        self._cleanup_old_overlays()
        
        # Create new overlays
        successful_masks = self._create_mask_overlays(masks, render_context)
        
        # Hide unused overlays and show first mask if appropriate
        self._finalize_overlay_display(successful_masks, len(masks))
    
    def show_selected_mask(self, selected_index: int, total_masks: int) -> None:
        """
        Show only the selected mask and hide others.
        
        Args:
            selected_index: Index of the mask to show (-1 to hide all)
            total_masks: Total number of masks available
        """
        for idx in range(total_masks):
            series_tag = f"mask_series_{idx}"
            if dpg.does_item_exist(series_tag):
                try:
                    # Show only the selected mask, hide all if selected_index is -1
                    should_show = (selected_index >= 0 and idx == selected_index)
                    dpg.configure_item(series_tag, show=should_show)
                except Exception as e:
                    print(f"Error configuring mask {idx}: {e}")
        
        # Make sure the axis is properly fit if showing a mask
        if 0 <= selected_index < total_masks:
            try:
                dpg.fit_axis_data(self.x_axis_tag)
                dpg.fit_axis_data(self.y_axis_tag)
            except Exception as e:
                print(f"Error fitting axis data: {e}")
    
    def cleanup_all_mask_overlays(self) -> None:
        """Clean up all mask overlays."""
        # Clean up series for up to 100 masks to handle large accumulations
        for idx in range(100):
            series_tag = f"mask_series_{idx}"
            if dpg.does_item_exist(series_tag):
                try:
                    dpg.configure_item(series_tag, show=False)
                    dpg.delete_item(series_tag)
                except Exception as e:
                    print(f"Error cleaning up mask overlay {idx}: {e}")
    
    def cleanup_mask_overlay(self, mask_index: int) -> None:
        """Clean up the visual overlay for a specific mask."""
        series_tag = f"mask_series_{mask_index}"
        if dpg.does_item_exist(series_tag):
            try:
                dpg.configure_item(series_tag, show=False)
                dpg.delete_item(series_tag)
            except Exception as e:
                print(f"Error cleaning up mask overlay {mask_index}: {e}")
    
    def _prepare_render_context(self, crop_rotate_ui) -> Dict[str, Any]:
        """Prepare rendering context with dimensions, rotation, and flip info."""
        context = {
            'texture_w': crop_rotate_ui.texture_w,
            'texture_h': crop_rotate_ui.texture_h,
            'orig_w': crop_rotate_ui.orig_w,
            'orig_h': crop_rotate_ui.orig_h,
            'current_angle': 0,
            'flip_horizontal': False,
            'flip_vertical': False,
            'offset_x': 0,
            'offset_y': 0
        }
        
        # Get current rotation angle if available
        if dpg.does_item_exist("rotation_slider"):
            context['current_angle'] = dpg.get_value("rotation_slider")
        
        # Get current flip states from CropRotateUI
        if hasattr(crop_rotate_ui, 'get_flip_states'):
            flip_states = crop_rotate_ui.get_flip_states()
            context['flip_horizontal'] = flip_states.get('flip_horizontal', False)
            context['flip_vertical'] = flip_states.get('flip_vertical', False)
        
        # Calculate offset based on rotation
        if (context['current_angle'] != 0 and 
            hasattr(crop_rotate_ui, 'rotated_image') and 
            crop_rotate_ui.rotated_image is not None):
            # Use rotated image dimensions and offset
            if hasattr(crop_rotate_ui, 'rot_h') and hasattr(crop_rotate_ui, 'rot_w'):
                context['offset_x'] = (context['texture_w'] - crop_rotate_ui.rot_w) // 2
                context['offset_y'] = (context['texture_h'] - crop_rotate_ui.rot_h) // 2
                context['rot_w'] = crop_rotate_ui.rot_w
                context['rot_h'] = crop_rotate_ui.rot_h
            else:
                # Fallback to original calculation
                context['offset_x'] = (context['texture_w'] - context['orig_w']) // 2
                context['offset_y'] = (context['texture_h'] - context['orig_h']) // 2
        else:
            # No rotation, use original calculation
            context['offset_x'] = (context['texture_w'] - context['orig_w']) // 2
            context['offset_y'] = (context['texture_h'] - context['orig_h']) // 2
        
        return context
    
    def _cleanup_old_overlays(self) -> None:
        """Remove existing series to avoid conflicts."""
        for idx in range(100):
            old_series_tag = f"mask_series_{idx}"
            if dpg.does_item_exist(old_series_tag):
                try:
                    dpg.delete_item(old_series_tag)
                except Exception as e:
                    print(f"Error deleting old series {old_series_tag}: {e}")
    
    def _create_mask_overlays(self, masks: List[Dict[str, Any]], context: Dict[str, Any]) -> int:
        """Create overlay textures and series for all masks."""
        successful_masks = 0
        timestamp = int(time.time() * 1000000)  # Microsecond timestamp for uniqueness
        
        for idx, mask in enumerate(masks):
            try:
                binary_mask = mask.get("segmentation")
                if binary_mask is None:
                    continue
                
                # Create overlay texture for this mask
                overlay = self._create_mask_overlay_texture(
                    binary_mask, idx, context
                )
                
                if overlay is not None:
                    # Create texture and series in DearPyGUI
                    texture_tag = f"mask_overlay_{idx}_{timestamp}"
                    series_tag = f"mask_series_{idx}"
                    
                    # Convert to texture format
                    texture_data = overlay.flatten().astype(np.float32) / 255.0
                    
                    # Create new texture with unique tag
                    with dpg.texture_registry():
                        dpg.add_raw_texture(
                            context['texture_w'], 
                            context['texture_h'], 
                            texture_data, 
                            tag=texture_tag, 
                            format=dpg.mvFormat_Float_rgba
                        )
                    
                    # Create new image series
                    dpg.add_image_series(
                        texture_tag,
                        bounds_min=[0, 0],
                        bounds_max=[context['texture_w'], context['texture_h']],
                        parent=self.y_axis_tag,
                        tag=series_tag,
                        show=False  # Start hidden
                    )
                    
                    successful_masks += 1
                
            except Exception as e:
                print(f"Error creating mask overlay {idx}: {e}")
                traceback.print_exc()
                continue
        
        return successful_masks
    
    def _create_mask_overlay_texture(self, binary_mask: np.ndarray, mask_idx: int, context: Dict[str, Any]) -> Optional[np.ndarray]:
        """Create an overlay texture for a single mask."""
        try:
            # Apply transformations (rotation and flips) to the mask
            transformed_mask = self._transform_mask(
                binary_mask, 
                context['current_angle'],
                context['flip_horizontal'], 
                context['flip_vertical'],
                context['orig_w'], 
                context['orig_h']
            )
            
            # Create overlay texture
            overlay = np.zeros((context['texture_h'], context['texture_w'], 4), dtype=np.uint8)
            color = self.colors[mask_idx % len(self.colors)]
            
            # Apply mask with color based on transformation state
            if (context['current_angle'] != 0 and 
                'rot_h' in context and 'rot_w' in context):
                # Use rotated image dimensions
                mask_h = min(transformed_mask.shape[0], context['rot_h'])
                mask_w = min(transformed_mask.shape[1], context['rot_w'])
                
                # Apply transformed mask with color
                for channel in range(4):
                    overlay[
                        context['offset_y']:context['offset_y'] + mask_h, 
                        context['offset_x']:context['offset_x'] + mask_w, 
                        channel
                    ] = np.where(transformed_mask[:mask_h, :mask_w] == 1, color[channel], 0)
            else:
                # Apply original mask with color (no rotation)
                for channel in range(4):
                    overlay[
                        context['offset_y']:context['offset_y'] + context['orig_h'], 
                        context['offset_x']:context['offset_x'] + context['orig_w'], 
                        channel
                    ] = np.where(transformed_mask == 1, color[channel], 0)
            
            return overlay
            
        except Exception as e:
            print(f"Error creating overlay texture for mask {mask_idx}: {e}")
            return None
    
    def _transform_mask(self, mask: np.ndarray, angle: float, flip_horizontal: bool, flip_vertical: bool, orig_w: int, orig_h: int) -> np.ndarray:
        """Apply rotation and flip transformations to a mask."""       
        try:
            transformed_mask = mask.copy()
            
            # Apply flips first (before rotation)
            if flip_horizontal:
                transformed_mask = np.fliplr(transformed_mask)
            
            if flip_vertical:
                transformed_mask = np.flipud(transformed_mask)
            
            # Apply rotation if needed
            if angle != 0:
                transformed_mask = self._rotate_mask(transformed_mask, angle, orig_w, orig_h)
            
            return transformed_mask
            
        except Exception as e:
            print(f"Error transforming mask: {e}")
            # Return original mask if transformation fails
            return mask
    
    def _rotate_mask(self, mask: np.ndarray, angle: float, orig_w: int, orig_h: int) -> np.ndarray:
        """Rotate a mask by the specified angle."""        
        try:
            # Convert boolean mask to uint8 for OpenCV
            mask_uint8 = (mask * 255).astype(np.uint8)
            
            # Calculate rotation matrix
            center = (orig_w / 2, orig_h / 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            
            # Calculate new dimensions
            cos = abs(M[0, 0])
            sin = abs(M[0, 1])
            new_w = int((orig_h * sin) + (orig_w * cos))
            new_h = int((orig_h * cos) + (orig_w * sin))
            
            # Adjust the rotation matrix for the new image center
            M[0, 2] += (new_w / 2) - center[0]
            M[1, 2] += (new_h / 2) - center[1]
            
            # Apply rotation to mask
            rotated_mask = cv2.warpAffine(
                mask_uint8, M, (new_w, new_h), 
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT, 
                borderValue=0
            )
            
            # Convert back to boolean
            return (rotated_mask > 127).astype(bool)
            
        except Exception as e:
            print(f"Error rotating mask: {e}")
            # Return original mask if rotation fails
            return mask
    
    def _finalize_overlay_display(self, successful_masks: int, total_masks: int) -> None:
        """Hide unused overlays and show first mask if appropriate."""
        # Hide any unused existing overlays (beyond current mask count)
        for idx in range(total_masks, 100):
            series_tag = f"mask_series_{idx}"
            if dpg.does_item_exist(series_tag):
                try:
                    dpg.configure_item(series_tag, show=False)
                except Exception as e:
                    print(f"Error hiding overlay {idx}: {e}")
        
        # Show first mask if any were created successfully and masks should be visible
        if successful_masks > 0:
            try:
                # Only show masks if conditions are met
                masks_enabled = (dpg.does_item_exist("mask_section_toggle") and 
                               dpg.get_value("mask_section_toggle"))
                crop_mode_active = (dpg.does_item_exist("crop_mode") and 
                                  dpg.get_value("crop_mode"))
                show_overlay = (dpg.does_item_exist("show_mask_overlay") and 
                              dpg.get_value("show_mask_overlay"))
                
                if masks_enabled and not crop_mode_active and show_overlay:
                    self.show_selected_mask(0, total_masks)

            except Exception as e:
                print(f"Error showing selected mask: {e}")
