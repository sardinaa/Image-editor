# Performance optimizations for autosegmentation
import asyncio
import time
from typing import List, Dict, Any, Optional, Set
from concurrent.futures import ThreadPoolExecutor
from utils.ui_helpers import UIStateManager

class PerformanceOptimizedMasksPanel:
    """Performance optimizations for the MasksPanel to handle heavy autosegmentation workloads."""
    
    def __init__(self, masks_panel):
        self.masks_panel = masks_panel
        
        # Performance settings
        self.MAX_MASKS_LIMIT = 50  # Configurable mask limit
        self.UI_UPDATE_THROTTLE_MS = 100  # Throttle UI updates
        self.BATCH_SIZE = 10  # Process masks in batches
        self.OVERLAY_UPDATE_DELAY = 50  # Delay between overlay updates
        
        # State tracking
        self.last_ui_update = 0
        self.pending_ui_update = False
        self.mask_processing_active = False
        self.current_batch = 0
        
        # Async processing
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.segmentation_task = None
        
    def set_max_masks_limit(self, limit: int):
        """Set the maximum number of masks allowed."""
        self.MAX_MASKS_LIMIT = max(1, min(limit, 200))  # Reasonable bounds
        
    async def perform_optimized_auto_segmentation(self):
        """Perform autosegmentation with performance optimizations."""
        if self.mask_processing_active:
            return
            
        self.mask_processing_active = True
        self.masks_panel._show_segmentation_loading()
        
        try:
            # Get current image
            current_image = self.masks_panel.main_window.app_service.image_service.get_current_image()
            if current_image is None:
                return
                
            # Use crop/rotate UI's processed image if available
            if (self.masks_panel.main_window.app_service.image_service.crop_rotate_ui and 
                hasattr(self.masks_panel.main_window.app_service.image_service.crop_rotate_ui, 'original_image')):
                image_to_segment = self.masks_panel.main_window.app_service.image_service.crop_rotate_ui.original_image
            else:
                image_to_segment = current_image
                
            # Perform segmentation in background thread with progress updates
            masks = await self._segment_with_progress(image_to_segment)
            
            # Apply mask limit
            if len(masks) > self.MAX_MASKS_LIMIT:
                # Sort by quality metrics and keep best masks
                masks = self._filter_best_masks(masks, self.MAX_MASKS_LIMIT)
                self.masks_panel.main_window._update_status(
                    f"Limited to {self.MAX_MASKS_LIMIT} best masks (from {len(masks)} total)"
                )
            
            # Update masks incrementally
            await self._update_masks_incrementally(masks)
            
        except Exception as e:
            self.masks_panel.main_window._update_status(f"Auto segmentation failed: {str(e)}")
        finally:
            self.mask_processing_active = False
            self.masks_panel._hide_segmentation_loading()
            
    async def _segment_with_progress(self, image):
        """Perform segmentation with progress reporting."""
        def run_segmentation():
            try:
                # Use the existing segmentation service but with progress monitoring
                segmentation_service = self.masks_panel.main_window.app_service.get_segmentation_service()
                return segmentation_service.segment_image(image)
            except Exception as e:
                print(f"Segmentation error: {e}")
                return []
                
        # Run segmentation in thread pool
        loop = asyncio.get_event_loop()
        masks = await loop.run_in_executor(self.executor, run_segmentation)
        return masks
        
    def _filter_best_masks(self, masks: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
        """Filter masks to keep only the best ones based on quality metrics."""
        # Sort by multiple criteria: area, stability score, predicted IoU
        def mask_quality_score(mask):
            area = mask.get('area', 0)
            stability = mask.get('stability_score', 0)
            iou = mask.get('predicted_iou', 0)
            
            # Normalize area (prefer medium-sized masks, not too small or too large)
            area_score = min(area / 10000, 1.0) if area < 50000 else max(0.5, 1.0 - (area - 50000) / 100000)
            
            # Combined score
            return (stability * 0.4) + (iou * 0.4) + (area_score * 0.2)
        
        # Sort by quality score (descending)
        sorted_masks = sorted(masks, key=mask_quality_score, reverse=True)
        return sorted_masks[:limit]
        
    async def _update_masks_incrementally(self, masks: List[Dict[str, Any]]):
        """Update masks incrementally to avoid UI freezing."""
        mask_service = self.masks_panel.main_window.app_service.get_mask_service()
        
        # Clear existing masks first
        mask_service.clear_all_masks()
        
        # Add masks in batches
        for i in range(0, len(masks), self.BATCH_SIZE):
            batch = masks[i:i + self.BATCH_SIZE]
            self.current_batch = i // self.BATCH_SIZE + 1
            total_batches = (len(masks) + self.BATCH_SIZE - 1) // self.BATCH_SIZE
            
            # Add batch to service
            for mask in batch:
                mask_service.add_mask(mask)
                
            # Update UI with current progress
            progress_msg = f"Processing masks: batch {self.current_batch}/{total_batches}"
            self.masks_panel.main_window._update_status(progress_msg)
            
            # Throttled UI update
            await self._throttled_ui_update()
            
            # Small delay to prevent UI freezing
            await asyncio.sleep(0.01)
            
        # Final complete update
        final_masks = mask_service.get_masks()
        mask_names = mask_service.get_mask_names()
        
        # Update UI and overlays
        self.masks_panel.update_masks(final_masks, mask_names)
        await self._update_overlays_progressively(final_masks)
        
        self.masks_panel.main_window._update_status(
            f"Auto segmentation completed: {len(final_masks)} masks created"
        )
        
    async def _throttled_ui_update(self):
        """Throttle UI updates to prevent overwhelming the interface."""
        current_time = time.time() * 1000  # Convert to milliseconds
        
        if current_time - self.last_ui_update >= self.UI_UPDATE_THROTTLE_MS:
            if not self.pending_ui_update:
                self.pending_ui_update = True
                
                # Get current masks for update
                mask_service = self.masks_panel.main_window.app_service.get_mask_service()
                current_masks = mask_service.get_masks()
                mask_names = mask_service.get_mask_names()
                
                # Update masks UI
                self.masks_panel.update_masks(current_masks, mask_names)
                
                self.last_ui_update = current_time
                self.pending_ui_update = False
                
                # Small delay to let UI process
                await asyncio.sleep(0.001)
                
    async def _update_overlays_progressively(self, masks: List[Dict[str, Any]]):
        """Update mask overlays progressively to avoid rendering bottlenecks."""
        if not hasattr(self.masks_panel.main_window, 'mask_overlay_renderer'):
            return
            
        renderer = self.masks_panel.main_window.mask_overlay_renderer
        crop_rotate_ui = self.masks_panel.main_window.crop_rotate_ui
        
        if not renderer or not crop_rotate_ui:
            return
            
        # Update overlays in smaller batches
        overlay_batch_size = 5
        for i in range(0, len(masks), overlay_batch_size):
            batch = masks[i:i + overlay_batch_size]
            
            # Update this batch of overlays
            renderer.update_mask_overlays(batch, crop_rotate_ui)
            
            # Small delay between batches
            await asyncio.sleep(self.OVERLAY_UPDATE_DELAY / 1000.0)
            
    def optimize_mask_table_updates(self):
        """Optimize mask table updates to avoid full recreation."""
        original_update_masks = self.masks_panel.update_masks
        
        def optimized_update_masks(masks: List[Dict[str, Any]], mask_names: List[str] = None):
            """Optimized version that tries incremental updates first."""
            try:
                # If the mask count is similar, try incremental update
                current_count = len(self.masks_panel.mask_checkboxes)
                new_count = len(masks)
                
                if abs(current_count - new_count) <= 3 and current_count > 0:
                    # Try incremental update
                    success = self._incremental_table_update(masks, mask_names)
                    if success:
                        return
                        
                # Fall back to full update
                original_update_masks(masks, mask_names)
                
            except Exception as e:
                print(f"Error in optimized mask update: {e}")
                # Always fall back to original implementation
                original_update_masks(masks, mask_names)
                
        # Replace the method
        self.masks_panel.update_masks = optimized_update_masks
        
    def _incremental_table_update(self, masks: List[Dict[str, Any]], mask_names: List[str] = None) -> bool:
        """Attempt incremental table update instead of full recreation."""
        try:
            # Create display names
            if mask_names and len(mask_names) >= len(masks):
                base_names = mask_names[:len(masks)]
            else:
                base_names = [f"Mask {idx+1}" for idx in range(len(masks))]
                
            display_names = []
            for idx, base_name in enumerate(base_names):
                if idx in self.masks_panel.mask_to_group:
                    group_id = self.masks_panel.mask_to_group[idx]
                    group_num = group_id.split('_')[-1] if '_' in group_id else group_id
                    display_name = f"[G{group_num}] {base_name}"
                else:
                    display_name = base_name
                display_names.append(display_name)
                
            # Update existing items if possible
            current_count = len(self.masks_panel.mask_checkboxes)
            new_count = len(display_names)
            
            # Update existing items
            for idx in range(min(current_count, new_count)):
                selectable_tag = f"mask_selectable_{idx}"
                if UIStateManager.safe_item_exists(selectable_tag):
                    UIStateManager.safe_configure_item(selectable_tag, label=display_names[idx])
                    
            # Add new items if needed
            if new_count > current_count:
                for idx in range(current_count, new_count):
                    with dpg.table_row(tag=f"mask_row_{idx}", parent="mask_table"):
                        dpg.add_selectable(
                            label=display_names[idx],
                            tag=f"mask_selectable_{idx}",
                            callback=self.masks_panel._create_row_callback(idx),
                            span_columns=True
                        )
                    self.masks_panel.mask_checkboxes[idx] = f"mask_selectable_{idx}"
                    
            # Remove extra items if needed
            elif new_count < current_count:
                for idx in range(new_count, current_count):
                    row_tag = f"mask_row_{idx}"
                    if UIStateManager.safe_item_exists(row_tag):
                        dpg.delete_item(row_tag)
                    if idx in self.masks_panel.mask_checkboxes:
                        del self.masks_panel.mask_checkboxes[idx]
                        
            return True
            
        except Exception as e:
            print(f"Incremental update failed: {e}")
            return False
            
    def add_mask_limit_ui(self):
        """Add UI controls for mask limit configuration."""
        # Add this after the auto segment button
        if UIStateManager.safe_item_exists("auto_segment_btn"):
            parent = dpg.get_item_parent("auto_segment_btn")
            
            with dpg.group(horizontal=True, parent=parent):
                dpg.add_text("Max Masks:")
                dpg.add_input_int(
                    tag="max_masks_input",
                    default_value=self.MAX_MASKS_LIMIT,
                    min_value=1,
                    max_value=200,
                    width=60,
                    callback=self._on_max_masks_changed
                )
                
    def _on_max_masks_changed(self, sender, app_data, user_data):
        """Handle max masks limit change."""
        new_limit = max(1, min(app_data, 200))
        self.set_max_masks_limit(new_limit)
        self.masks_panel.main_window._update_status(f"Max masks limit set to {new_limit}")
        
    def add_performance_monitoring(self):
        """Add performance monitoring to track segmentation times."""
        original_auto_segment = self.masks_panel._auto_segment
        
        def monitored_auto_segment(sender, app_data, user_data):
            start_time = time.time()
            
            def completion_callback():
                end_time = time.time()
                duration = end_time - start_time
                mask_count = len(self.masks_panel.mask_checkboxes)
                
                performance_msg = f"Segmentation completed in {duration:.2f}s, {mask_count} masks"
                self.masks_panel.main_window._update_status(performance_msg)
                
            # Run async version
            asyncio.create_task(self.perform_optimized_auto_segmentation())
            
        self.masks_panel._auto_segment = monitored_auto_segment
        
    def cleanup(self):
        """Clean up resources."""
        if self.segmentation_task and not self.segmentation_task.done():
            self.segmentation_task.cancel()
        self.executor.shutdown(wait=False)


class AsyncSegmentationMixin:
    """Mixin to add async segmentation capabilities to existing panels."""
    
    def __init__(self):
        self.performance_optimizer = None
        
    def enable_performance_optimizations(self):
        """Enable performance optimizations for this masks panel."""
        if not hasattr(self, 'performance_optimizer') or self.performance_optimizer is None:
            self.performance_optimizer = PerformanceOptimizedMasksPanel(self)
            
            # Apply optimizations
            self.performance_optimizer.optimize_mask_table_updates()
            self.performance_optimizer.add_performance_monitoring()
            
            # Add UI controls
            self.performance_optimizer.add_mask_limit_ui()
            
    def set_segmentation_limits(self, max_masks: int = 50, batch_size: int = 10):
        """Configure segmentation performance limits."""
        if self.performance_optimizer:
            self.performance_optimizer.set_max_masks_limit(max_masks)
            self.performance_optimizer.BATCH_SIZE = batch_size
            
    def cleanup_performance_optimizations(self):
        """Clean up performance optimization resources."""
        if self.performance_optimizer:
            self.performance_optimizer.cleanup()
            self.performance_optimizer = None


# Usage example integration pattern:
"""
To integrate these optimizations into the existing MasksPanel:

1. Add the mixin to MasksPanel class definition:
   class MasksPanel(BasePanel, AsyncSegmentationMixin):

2. In the __init__ method, call:
   AsyncSegmentationMixin.__init__(self)

3. In the draw() method, after creating UI elements:
   self.enable_performance_optimizations()

4. Optionally configure limits:
   self.set_segmentation_limits(max_masks=50, batch_size=10)

5. In cleanup/destructor:
   self.cleanup_performance_optimizations()
"""
