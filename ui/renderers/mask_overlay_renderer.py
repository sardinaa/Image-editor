import dearpygui.dearpygui as dpg
import numpy as np
import cv2
import time
import traceback
from typing import List, Dict, Any, Optional, Set


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
        
        # Performance settings - Increased limits for better mask support
        self.max_visible_overlays = 50  # Increased from 10 to 50
        self.max_total_overlays = 200   # Maximum number of overlays to support
        self.progressive_loading = True
        self.last_update_time = 0
        self.update_throttle_ms = 50  # Throttle overlay updates
        
        # Cache for overlay data
        self.overlay_cache: Dict[str, Any] = {}
        self.visible_overlay_indices: Set[int] = set()
        
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
    
    def update_mask_overlays(self, masks: List[Dict[str, Any]], crop_rotate_ui, auto_show_first: bool = False) -> None:
        """
        Update the visual mask overlays on the image with performance optimizations.
        
        Args:
            masks: List of mask data dictionaries
            crop_rotate_ui: Reference to crop/rotate UI for dimensions and rotation
            auto_show_first: Whether to automatically show the first mask when new masks are created
        """
        if not masks or not crop_rotate_ui:
            return
        
        # Throttle updates to prevent overwhelming the renderer
        current_time = time.time() * 1000
        if current_time - self.last_update_time < self.update_throttle_ms:
            return
        
        self.last_update_time = current_time
        
        # Limit the number of masks to render for performance
        limited_masks = masks[:self.max_visible_overlays] if len(masks) > self.max_visible_overlays else masks
        
        # Get texture dimensions and rotation state
        render_context = self._prepare_render_context(crop_rotate_ui)
        
        # Clean up old overlays beyond the current count
        self._cleanup_excess_overlays(len(limited_masks))
        
        # Create new overlays for limited mask set
        if self.progressive_loading:
            successful_masks = self._create_mask_overlays_progressive(limited_masks, render_context)
        else:
            successful_masks = self._create_mask_overlays(limited_masks, render_context)
        
        # Hide unused overlays and manage mask visibility based on auto_show_first setting
        self._finalize_overlay_display(successful_masks, len(limited_masks), auto_show_first)
    
    def set_performance_settings(self, max_visible: int = 50, throttle_ms: int = 50, progressive: bool = True):
        """Configure performance settings for the overlay renderer."""
        self.max_visible_overlays = max_visible
        self.update_throttle_ms = throttle_ms
        self.progressive_loading = progressive
        
        # Clear cache when settings change
        self.overlay_cache.clear()
    
    def _cleanup_excess_overlays(self, current_count: int):
        """Clean up overlays beyond the current mask count."""
        for idx in range(current_count, self.max_total_overlays):  # Use configurable limit instead of hardcoded 100
            series_tag = f"mask_series_{idx}"
            if dpg.does_item_exist(series_tag):
                try:
                    dpg.configure_item(series_tag, show=False)
                    # Remove from visible set
                    self.visible_overlay_indices.discard(idx)
                except Exception as e:
                    print(f"Error hiding excess overlay {idx}: {e}")
    
    def _create_mask_overlays_progressive(self, masks: List[Dict[str, Any]], render_context: Dict[str, Any]) -> int:
        """Create mask overlays progressively to reduce frame drops."""
        successful_masks = 0
        
        # Process masks in small batches
        batch_size = 3
        for i in range(0, len(masks), batch_size):
            batch = masks[i:i + batch_size]
            
            for j, mask_data in enumerate(batch):
                mask_index = i + j
                try:
                    if self._create_single_mask_overlay(mask_index, mask_data, render_context):
                        successful_masks += 1
                        self.visible_overlay_indices.add(mask_index)
                except Exception as e:
                    print(f"Error creating progressive overlay for mask {mask_index}: {e}")
            
            # Small yield to prevent blocking
            if i + batch_size < len(masks):
                time.sleep(0.001)
        
        return successful_masks
    
    def show_selected_mask(self, selected_index: int, total_masks: int) -> None:
        """
        Show only the selected mask and hide others.
        
        Args:
            selected_index: Index of the mask to show (-1 to hide all)
            total_masks: Total number of masks available
        """
        print(f"MaskOverlayRenderer: Showing mask {selected_index} out of {total_masks} total")
        
        for idx in range(total_masks):
            series_tag = f"mask_series_{idx}"
            if dpg.does_item_exist(series_tag):
                try:
                    # Show only the selected mask, hide all if selected_index is -1
                    should_show = (selected_index >= 0 and idx == selected_index)
                    dpg.configure_item(series_tag, show=should_show)
                    if should_show:
                        print(f"MaskOverlayRenderer: Showing mask series {idx}")
                    else:
                        print(f"MaskOverlayRenderer: Hiding mask series {idx}")
                except Exception as e:
                    print(f"Error configuring mask {idx}: {e}")
            else:
                print(f"MaskOverlayRenderer: Mask series {idx} does not exist")
        
        # Make sure the axis is properly fit if showing a mask
        if 0 <= selected_index < total_masks:
            try:
                dpg.fit_axis_data(self.x_axis_tag)
                dpg.fit_axis_data(self.y_axis_tag)
                print(f"MaskOverlayRenderer: Fitted axis data for mask {selected_index}")
            except Exception as e:
                print(f"Error fitting axis data: {e}")
    
    def show_selected_masks(self, selected_indices: List[int], total_masks: int) -> None:
        """
        Show multiple selected masks and hide others with performance optimization.
        
        Args:
            selected_indices: List of mask indices to show
            total_masks: Total number of masks available
        """
        # Limit the number of visible masks for performance
        limited_indices = selected_indices[:self.max_visible_overlays]
        
        for idx in range(total_masks):
            series_tag = f"mask_series_{idx}"
            if dpg.does_item_exist(series_tag):
                try:
                    # Show mask if it's in the limited selected indices
                    should_show = idx in limited_indices
                    dpg.configure_item(series_tag, show=should_show)
                    
                    if should_show:
                        self.visible_overlay_indices.add(idx)
                    else:
                        self.visible_overlay_indices.discard(idx)
                except Exception as e:
                    print(f"Error configuring mask {idx}: {e}")
        
        # Make sure the axis is properly fit if showing masks
        if limited_indices:
            try:
                dpg.fit_axis_data(self.x_axis_tag)
                dpg.fit_axis_data(self.y_axis_tag)
            except Exception as e:
                print(f"Error fitting axis data: {e}")
    
    def get_visible_overlay_count(self) -> int:
        """Get the number of currently visible overlays."""
        return len(self.visible_overlay_indices)
    
    def clear_overlay_cache(self):
        """Clear the overlay cache to free memory."""
        self.overlay_cache.clear()
        self.visible_overlay_indices.clear()
    
    def cleanup_all_mask_overlays(self) -> None:
        """Clean up all mask overlays."""
        # Clean up series using configurable limit instead of hardcoded 100
        for idx in range(self.max_total_overlays):
            series_tag = f"mask_series_{idx}"
            if dpg.does_item_exist(series_tag):
                try:
                    dpg.configure_item(series_tag, show=False)
                    dpg.delete_item(series_tag)
                except Exception as e:
                    print(f"Error cleaning up mask overlay {idx}: {e}")
        
        # Clear performance tracking
        self.visible_overlay_indices.clear()
        self.overlay_cache.clear()
    
    def _create_single_mask_overlay(self, mask_index: int, mask_data: Dict[str, Any], render_context: Dict[str, Any]) -> bool:
        """Create a single mask overlay with caching."""
        cache_key = f"mask_{mask_index}_{hash(str(mask_data.get('area', 0)))}"
        
        # Check cache first
        if cache_key in self.overlay_cache:
            cached_data = self.overlay_cache[cache_key]
            try:
                self._render_cached_overlay(mask_index, cached_data)
                return True
            except Exception as e:
                print(f"Error rendering cached overlay: {e}")
                # Remove bad cache entry
                del self.overlay_cache[cache_key]
        
        # Create new overlay
        try:
            success = self._create_mask_overlay_impl(mask_index, mask_data, render_context)
            if success:
                # Cache the result for future use
                self.overlay_cache[cache_key] = {
                    'mask_index': mask_index,
                    'timestamp': time.time()
                }
                # Limit cache size
                if len(self.overlay_cache) > 50:
                    self._cleanup_cache()
            return success
        except Exception as e:
            print(f"Error creating mask overlay {mask_index}: {e}")
            return False
    
    def _render_cached_overlay(self, mask_index: int, cached_data: Dict[str, Any]):
        """Render a cached overlay."""
        series_tag = f"mask_series_{mask_index}"
        if dpg.does_item_exist(series_tag):
            dpg.configure_item(series_tag, show=True)
    
    def _cleanup_cache(self):
        """Clean up old cache entries."""
        if len(self.overlay_cache) <= 25:
            return
        
        # Remove oldest entries
        sorted_items = sorted(self.overlay_cache.items(), 
                            key=lambda x: x[1].get('timestamp', 0))
        for key, _ in sorted_items[:len(sorted_items) // 2]:
            del self.overlay_cache[key]
    
    def _create_mask_overlay_impl(self, mask_index: int, mask_data: Dict[str, Any], render_context: Dict[str, Any]) -> bool:
        """Implementation of mask overlay creation."""
        mask = mask_data.get("segmentation")
        if mask is None or not isinstance(mask, np.ndarray):
            return False
        
        try:
            # Apply transformations and create overlay
            transformed_mask = self._apply_transformations(mask, render_context)
            if transformed_mask is None:
                return False
            
            # Create the overlay visualization
            self._create_overlay_visualization(mask_index, transformed_mask, render_context)
            return True
        except Exception as e:
            print(f"Error in mask overlay implementation: {e}")
            return False
    
    def _apply_transformations(self, mask: np.ndarray, context: Dict[str, Any]) -> Optional[np.ndarray]:
        """Apply transformations (rotation and flips) to the mask."""
        try:
            return self._transform_mask(
                mask,
                context.get('current_angle', 0),
                context.get('flip_horizontal', False),
                context.get('flip_vertical', False),
                context.get('orig_w', mask.shape[1]),
                context.get('orig_h', mask.shape[0])
            )
        except Exception as e:
            print(f"Error applying transformations: {e}")
            return mask  # Return original mask if transformation fails
    
    def _create_overlay_visualization(self, mask_index: int, transformed_mask: np.ndarray, context: Dict[str, Any]):
        """Create the visual overlay from a transformed mask."""
        try:
            # Create overlay texture
            overlay = self._create_mask_overlay_texture(transformed_mask, mask_index, context)
            if overlay is None:
                return
            
            # Create texture and series in DearPyGUI
            timestamp = int(time.time() * 1000000)
            texture_tag = f"mask_overlay_{mask_index}_{timestamp}"
            series_tag = f"mask_series_{mask_index}"
            
            # Convert to texture format
            texture_data = overlay.flatten().astype(np.float32) / 255.0
            
            # Remove existing series if it exists
            if dpg.does_item_exist(series_tag):
                dpg.delete_item(series_tag)
            
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
                show=True  # Show by default for clicked masks
            )
            
        except Exception as e:
            print(f"Error creating overlay visualization: {e}")
    
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
            'offset_y': 0,
            'is_cropped': False,
            'crop_rect': None
        }
        
        # Get current rotation angle if available
        if dpg.does_item_exist("rotation_slider"):
            context['current_angle'] = dpg.get_value("rotation_slider")
        
        # Get current flip states from CropRotateUI
        if hasattr(crop_rotate_ui, 'get_flip_states'):
            flip_states = crop_rotate_ui.get_flip_states()
            context['flip_horizontal'] = flip_states.get('flip_horizontal', False)
            context['flip_vertical'] = flip_states.get('flip_vertical', False)
        
        # Also try to get flip states directly from attributes if method doesn't exist
        if hasattr(crop_rotate_ui, 'flip_horizontal'):
            context['flip_horizontal'] = crop_rotate_ui.flip_horizontal
        if hasattr(crop_rotate_ui, 'flip_vertical'):
            context['flip_vertical'] = crop_rotate_ui.flip_vertical
        
        print(f"DEBUG: Mask overlay flip states - H: {context['flip_horizontal']}, V: {context['flip_vertical']}")
        
        # Check if we're in crop mode or result mode
        crop_mode = False
        if dpg.does_item_exist("crop_mode"):
            crop_mode = dpg.get_value("crop_mode")
        
        # Determine the correct offset and dimensions based on the current state
        if crop_mode and context['current_angle'] != 0:
            # Crop mode with rotation: use rotated image dimensions and offsets
            if (hasattr(crop_rotate_ui, 'rot_w') and hasattr(crop_rotate_ui, 'rot_h') and
                hasattr(crop_rotate_ui, 'offset_x') and hasattr(crop_rotate_ui, 'offset_y')):
                context['offset_x'] = crop_rotate_ui.offset_x
                context['offset_y'] = crop_rotate_ui.offset_y
                context['rot_w'] = crop_rotate_ui.rot_w
                context['rot_h'] = crop_rotate_ui.rot_h
                print(f"DEBUG: Mask overlay using crop mode rotated offsets: ({context['offset_x']}, {context['offset_y']})")
            else:
                # Fallback calculation
                context['offset_x'] = (context['texture_w'] - context['orig_w']) // 2
                context['offset_y'] = (context['texture_h'] - context['orig_h']) // 2
        elif not crop_mode and context['current_angle'] != 0:
            # Result mode with rotation: use rotated dimensions but check for crop
            if (hasattr(crop_rotate_ui, 'rot_w') and hasattr(crop_rotate_ui, 'rot_h') and
                hasattr(crop_rotate_ui, 'offset_x') and hasattr(crop_rotate_ui, 'offset_y')):
                # Check if there's a crop rectangle applied - if so, the displayed image is different
                if hasattr(crop_rotate_ui, 'user_rect') and crop_rotate_ui.user_rect:
                    context['is_cropped'] = True
                    context['crop_rect'] = crop_rotate_ui.user_rect
                    
                    # Calculate the actual cropped image dimensions that are displayed
                    crop_rect = crop_rotate_ui.user_rect
                    rx = int(crop_rect["x"] - crop_rotate_ui.offset_x)
                    ry = int(crop_rect["y"] - crop_rotate_ui.offset_y)
                    rw = int(crop_rect["w"])
                    rh = int(crop_rect["h"])
                    
                    # Clamp to rotated image bounds
                    rx = max(0, rx)
                    ry = max(0, ry)
                    rx_end = min(rx + rw, crop_rotate_ui.rot_w)
                    ry_end = min(ry + rh, crop_rotate_ui.rot_h)
                    actual_rw = rx_end - rx
                    actual_rh = ry_end - ry
                    
                    # The cropped image is displayed centered in the texture
                    context['display_offset_x'] = (context['texture_w'] - actual_rw) // 2
                    context['display_offset_y'] = (context['texture_h'] - actual_rh) // 2
                    context['cropped_display_w'] = actual_rw
                    context['cropped_display_h'] = actual_rh
                    
                    # Also store the crop region within the rotated image for proper mask transformation
                    context['crop_region_x'] = rx
                    context['crop_region_y'] = ry
                    context['crop_region_w'] = actual_rw
                    context['crop_region_h'] = actual_rh
                    
                    print(f"DEBUG: Mask overlay result mode crop - crop_region: ({rx},{ry},{actual_rw},{actual_rh}), display_offset: ({context['display_offset_x']}, {context['display_offset_y']})")
                else:
                    # No crop, just rotated image
                    context['offset_x'] = crop_rotate_ui.offset_x
                    context['offset_y'] = crop_rotate_ui.offset_y
                    context['rot_w'] = crop_rotate_ui.rot_w
                    context['rot_h'] = crop_rotate_ui.rot_h
                    print(f"DEBUG: Mask overlay using result mode rotated offsets: ({context['offset_x']}, {context['offset_y']})")
            else:
                context['offset_x'] = (context['texture_w'] - context['orig_w']) // 2
                context['offset_y'] = (context['texture_h'] - context['orig_h']) // 2
        else:
            # No rotation case
            context['offset_x'] = (context['texture_w'] - context['orig_w']) // 2
            context['offset_y'] = (context['texture_h'] - context['orig_h']) // 2
            
            # Check for crop even without rotation
            if hasattr(crop_rotate_ui, 'user_rect') and crop_rotate_ui.user_rect:
                context['is_cropped'] = True
                context['crop_rect'] = crop_rotate_ui.user_rect
                # For no rotation case, calculate simple crop display
                crop_rect = crop_rotate_ui.user_rect
                orig_h, orig_w = context['orig_h'], context['orig_w']
                
                # Calculate crop region relative to original image
                rx = max(0, min(orig_w, int(crop_rect["x"] - context['offset_x'])))
                ry = max(0, min(orig_h, int(crop_rect["y"] - context['offset_y'])))
                rw = min(int(crop_rect["w"]), orig_w - rx)
                rh = min(int(crop_rect["h"]), orig_h - ry)
                
                # Display position for cropped result
                context['display_offset_x'] = (context['texture_w'] - rw) // 2
                context['display_offset_y'] = (context['texture_h'] - rh) // 2
                context['cropped_display_w'] = rw
                context['cropped_display_h'] = rh
                context['crop_region_x'] = rx
                context['crop_region_y'] = ry
                context['crop_region_w'] = rw
                context['crop_region_h'] = rh
                
                print(f"DEBUG: Mask overlay detected crop rectangle (no rotation): crop_region: ({rx},{ry},{rw},{rh}), display: ({context['display_offset_x']},{context['display_offset_y']})")
        
        print(f"DEBUG: Mask overlay context - crop_mode: {crop_mode}, angle: {context['current_angle']}, offset: ({context.get('offset_x', 0)}, {context.get('offset_y', 0)}), cropped: {context.get('is_cropped', False)}")
        return context
    
    def _cleanup_old_overlays(self) -> None:
        """Remove existing series to avoid conflicts."""
        for idx in range(self.max_total_overlays):  # Use configurable limit
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
            # Create overlay texture
            overlay = np.zeros((context['texture_h'], context['texture_w'], 4), dtype=np.uint8)
            color = self.colors[mask_idx % len(self.colors)]
            
            # Handle different display scenarios
            if context.get('is_cropped', False) and 'display_offset_x' in context:
                # Image is cropped and displayed with specific offsets
                # The mask needs to be cropped to match the displayed region and positioned accordingly
                
                if 'crop_region_x' in context:
                    # Extract the part of the mask that corresponds to the crop region
                    crop_x = context['crop_region_x']
                    crop_y = context['crop_region_y']
                    crop_w = context['crop_region_w']
                    crop_h = context['crop_region_h']
                    
                    # Crop the mask to match the displayed image region
                    mask_h, mask_w = binary_mask.shape[:2]
                    
                    # Ensure crop region is within mask bounds
                    crop_x = max(0, min(crop_x, mask_w))
                    crop_y = max(0, min(crop_y, mask_h))
                    crop_end_x = min(crop_x + crop_w, mask_w)
                    crop_end_y = min(crop_y + crop_h, mask_h)
                    
                    if crop_end_x > crop_x and crop_end_y > crop_y:
                        # Extract the cropped portion of the mask
                        cropped_mask = binary_mask[crop_y:crop_end_y, crop_x:crop_end_x]
                        
                        # Position this cropped mask at the display offset
                        display_x = context['display_offset_x']
                        display_y = context['display_offset_y']
                        
                        cropped_h, cropped_w = cropped_mask.shape[:2]
                        end_y = min(display_y + cropped_h, context['texture_h'])
                        end_x = min(display_x + cropped_w, context['texture_w'])
                        
                        for channel in range(4):
                            overlay[
                                display_y:end_y,
                                display_x:end_x,
                                channel
                            ] = np.where(cropped_mask[:end_y-display_y, :end_x-display_x] == 1, color[channel], 0)
                        
                        print(f"DEBUG: Mask overlay cropped display - crop_region: ({crop_x},{crop_y},{crop_end_x-crop_x},{crop_end_y-crop_y}), display: ({display_x},{display_y}), cropped_mask_size: {cropped_w}x{cropped_h}")
                else:
                    # Fallback: simple centering
                    mask_h, mask_w = binary_mask.shape[:2]
                    display_offset_x = (context['texture_w'] - mask_w) // 2
                    display_offset_y = (context['texture_h'] - mask_h) // 2
                    
                    end_y = min(display_offset_y + mask_h, context['texture_h'])
                    end_x = min(display_offset_x + mask_w, context['texture_w'])
                    
                    for channel in range(4):
                        overlay[
                            display_offset_y:end_y,
                            display_offset_x:end_x,
                            channel
                        ] = np.where(binary_mask[:end_y-display_offset_y, :end_x-display_offset_x] == 1, color[channel], 0)
                        
                    print(f"DEBUG: Mask overlay fallback centering - mask_size: {mask_w}x{mask_h}, display_offset: ({display_offset_x}, {display_offset_y})")
                    
            elif (context.get('current_angle', 0) != 0 and 
                  'rot_h' in context and 'rot_w' in context):
                # Use rotated image dimensions and offsets
                mask_h = min(binary_mask.shape[0], context['rot_h'])
                mask_w = min(binary_mask.shape[1], context['rot_w'])
                
                print(f"DEBUG: Mask overlay for rotated image - rot_size: {context['rot_w']}x{context['rot_h']}, mask_size: {mask_w}x{mask_h}, offset: ({context['offset_x']}, {context['offset_y']})")
                
                # Apply transformed mask with color
                for channel in range(4):
                    overlay[
                        context['offset_y']:context['offset_y'] + mask_h, 
                        context['offset_x']:context['offset_x'] + mask_w, 
                        channel
                    ] = np.where(binary_mask[:mask_h, :mask_w] == 1, color[channel], 0)
            else:
                # Handle original image case (including flips without rotation)
                mask_h = min(binary_mask.shape[0], context['orig_h'])
                mask_w = min(binary_mask.shape[1], context['orig_w'])
                
                # Check if image has flips applied
                flip_h = context.get('flip_horizontal', False)
                flip_v = context.get('flip_vertical', False)
                
                # Apply any transformations (rotation and/or flips) to the mask
                transformed_mask = binary_mask
                if context.get('current_angle', 0) != 0 or flip_h or flip_v:
                    transformed_mask = self._transform_mask(
                        binary_mask,
                        context.get('current_angle', 0),
                        flip_h,
                        flip_v,
                        context['orig_w'],
                        context['orig_h']
                    )
                
                # Update mask dimensions after transformation
                mask_h = min(transformed_mask.shape[0], context['texture_h'] - context['offset_y'])
                mask_w = min(transformed_mask.shape[1], context['texture_w'] - context['offset_x'])
                
                if flip_h or flip_v:
                    print(f"DEBUG: Mask overlay for flipped image - flips: H{flip_h}/V{flip_v}, orig_size: {context['orig_w']}x{context['orig_h']}, transformed_mask_size: {mask_w}x{mask_h}, offset: ({context['offset_x']}, {context['offset_y']})")
                else:
                    print(f"DEBUG: Mask overlay for original image - orig_size: {context['orig_w']}x{context['orig_h']}, mask_size: {mask_w}x{mask_h}, offset: ({context['offset_x']}, {context['offset_y']})")
                
                # Apply the transformed mask at the offset position
                for channel in range(4):
                    overlay[
                        context['offset_y']:context['offset_y'] + mask_h, 
                        context['offset_x']:context['offset_x'] + mask_w, 
                        channel
                    ] = np.where(transformed_mask[:mask_h, :mask_w] == 1, color[channel], 0)
            
            return overlay
            
        except Exception as e:
            print(f"Error creating overlay texture for mask {mask_idx}: {e}")
            traceback.print_exc()
            return None
    
    def _transform_mask(self, mask: np.ndarray, angle: float, flip_horizontal: bool, flip_vertical: bool, orig_w: int, orig_h: int) -> np.ndarray:
        """Apply rotation and flip transformations to a mask to match image transformation order."""       
        try:
            transformed_mask = mask.copy()
            
            # Apply rotation first (same order as in CropRotateUI)
            if angle != 0:
                transformed_mask = self._rotate_mask(transformed_mask, angle, orig_w, orig_h)
            
            # Apply flips to the (possibly rotated) mask (same order as in CropRotateUI)
            if flip_horizontal:
                transformed_mask = np.fliplr(transformed_mask)
            
            if flip_vertical:
                transformed_mask = np.flipud(transformed_mask)
            
            print(f"DEBUG: Mask transformation applied - angle: {angle}, flip_h: {flip_horizontal}, flip_v: {flip_vertical}, input_shape: {mask.shape}, output_shape: {transformed_mask.shape}")
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
    
    def _finalize_overlay_display(self, successful_masks: int, total_masks: int, auto_show_first: bool = False) -> None:
        """Hide unused overlays and conditionally show first mask."""
        # Hide any unused existing overlays (beyond current mask count)
        for idx in range(total_masks, self.max_total_overlays):  # Use configurable limit instead of hardcoded 100
            series_tag = f"mask_series_{idx}"
            if dpg.does_item_exist(series_tag):
                try:
                    dpg.configure_item(series_tag, show=False)
                except Exception as e:
                    print(f"Error hiding overlay {idx}: {e}")
        
        # Show first mask only if auto_show_first is True and conditions are met
        if successful_masks > 0 and auto_show_first:
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
        elif successful_masks > 0:
            # Hide all masks by default when new masks are created without auto_show_first
            for idx in range(total_masks):
                series_tag = f"mask_series_{idx}"
                if dpg.does_item_exist(series_tag):
                    dpg.configure_item(series_tag, show=False)
                    self.visible_overlay_indices.discard(idx)
    
    def show_all_masks(self, total_masks: int) -> None:
        """
        Show all available masks without grouping.
        
        Args:
            total_masks: Total number of masks available
        """
        if total_masks <= 0:
            return
            
        # Create list of all mask indices
        all_indices = list(range(total_masks))
        self.show_selected_masks(all_indices, total_masks)
