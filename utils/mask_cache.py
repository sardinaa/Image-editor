"""
Lightweight mask cache to improve performance with many masks.
"""
import weakref
from typing import Dict, Any, List, Optional, Set
import numpy as np
import time
from utils.performance_config import PerformanceConfig


class MaskCache:
    """Lightweight cache for mask data and overlays."""
    
    def __init__(self, max_size: int = None):
        self.max_size = max_size or PerformanceConfig.MAX_CACHED_MASKS
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._access_times: Dict[str, float] = {}
        self._overlay_cache: Dict[str, Any] = {}
        
    def _generate_key(self, mask_data: Dict[str, Any]) -> str:
        """Generate a cache key for mask data."""
        # Use area and bbox as a simple hash
        area = mask_data.get('area', 0)
        bbox = mask_data.get('bbox', [0, 0, 0, 0])
        return f"{area}_{bbox[0]}_{bbox[1]}_{bbox[2]}_{bbox[3]}"
    
    def get_mask(self, key: str) -> Optional[Dict[str, Any]]:
        """Get cached mask data."""
        if key in self._cache:
            self._access_times[key] = time.time()
            return self._cache[key]
        return None
    
    def store_mask(self, mask_data: Dict[str, Any]) -> str:
        """Store mask data in cache."""
        key = self._generate_key(mask_data)
        
        # Clean cache if needed
        if len(self._cache) >= self.max_size:
            self._cleanup_old_entries()
        
        self._cache[key] = mask_data.copy()
        self._access_times[key] = time.time()
        return key
    
    def get_overlay(self, key: str) -> Optional[Any]:
        """Get cached overlay data."""
        return self._overlay_cache.get(key)
    
    def store_overlay(self, key: str, overlay_data: Any):
        """Store overlay data in cache."""
        if len(self._overlay_cache) >= self.max_size // 2:
            # Keep overlay cache smaller
            oldest_keys = sorted(self._overlay_cache.keys())[:len(self._overlay_cache) // 4]
            for old_key in oldest_keys:
                del self._overlay_cache[old_key]
        
        self._overlay_cache[key] = overlay_data
    
    def _cleanup_old_entries(self):
        """Remove least recently used entries."""
        if not self._access_times:
            return
        
        # Remove oldest 25% of entries
        sorted_items = sorted(self._access_times.items(), key=lambda x: x[1])
        num_to_remove = max(1, len(sorted_items) // 4)
        
        for key, _ in sorted_items[:num_to_remove]:
            self._cache.pop(key, None)
            self._access_times.pop(key, None)
            self._overlay_cache.pop(key, None)
    
    def clear(self):
        """Clear all cached data."""
        self._cache.clear()
        self._access_times.clear()
        self._overlay_cache.clear()
    
    def get_size(self) -> int:
        """Get current cache size."""
        return len(self._cache)


class PerformanceMaskManager:
    """Enhanced mask manager with performance optimizations."""
    
    def __init__(self):
        self.cache = MaskCache()
        self.visible_mask_indices: Set[int] = set()
        self.config = PerformanceConfig.get_optimized_settings()
        self._last_cleanup = time.time()
        self._cleanup_interval = 30.0  # Cleanup every 30 seconds
    
    def filter_high_quality_masks(self, masks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter masks based on quality metrics to reduce load."""
        if not self.config['quality_filtering']:
            return masks[:self.config['max_masks']]
        
        filtered_masks = []
        for mask in masks:
            # Area filtering
            area = mask.get('area', 0)
            if area < self.config['min_area'] or area > self.config['max_area']:
                continue
            
            # Stability filtering
            stability = mask.get('stability_score', 0)
            if stability < self.config['min_stability']:
                continue
            
            # IoU filtering
            iou = mask.get('predicted_iou', 0)
            if iou < self.config['min_iou']:
                continue
            
            filtered_masks.append(mask)
            
            # Stop when we reach the limit
            if len(filtered_masks) >= self.config['max_masks']:
                break
        
        return filtered_masks
    
    def get_visible_masks(self, all_masks: List[Dict[str, Any]], 
                         selected_indices: Set[int]) -> List[Dict[str, Any]]:
        """Get only the masks that should be visible to reduce rendering load."""
        max_visible = self.config['max_visible_overlays']
        
        if len(selected_indices) == 0:
            # Show first few masks if none selected
            return all_masks[:min(max_visible, len(all_masks))]
        elif len(selected_indices) <= max_visible:
            # Show all selected if within limit
            return [all_masks[i] for i in selected_indices if i < len(all_masks)]
        else:
            # Show subset of selected masks
            selected_list = sorted(list(selected_indices))[:max_visible]
            return [all_masks[i] for i in selected_list if i < len(all_masks)]
    
    def should_cleanup(self) -> bool:
        """Check if cleanup is needed."""
        current_time = time.time()
        return (current_time - self._last_cleanup) > self._cleanup_interval
    
    def cleanup_if_needed(self):
        """Perform cleanup if needed."""
        if self.should_cleanup():
            self.cache.clear()
            self._last_cleanup = time.time()
    
    def update_visible_masks(self, indices: Set[int]):
        """Update which masks are currently visible."""
        self.visible_mask_indices = indices.copy()
        
        # Cleanup if too many masks are being managed
        if len(indices) > PerformanceConfig.CLEANUP_THRESHOLD:
            self.cleanup_if_needed()
