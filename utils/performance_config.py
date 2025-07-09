"""
Performance configuration and constants for mask operations.
"""
from typing import Dict, Any
import os


class PerformanceConfig:
    """Configuration class for performance-related settings."""
    
    # Mask limits
    DEFAULT_MAX_MASKS = 25  # Reduced from 50 for better performance
    PERFORMANCE_MAX_MASKS = 15  # Ultra-lightweight mode
    HIGH_PERFORMANCE_MAX_MASKS = 50  # For powerful systems
    
    # UI update settings
    UI_UPDATE_THROTTLE_MS = 150  # Increased throttling
    BATCH_SIZE = 5  # Smaller batches for smoother updates
    OVERLAY_UPDATE_DELAY = 100  # More delay between overlay updates
    
    # Memory management
    ENABLE_MASK_CACHING = True
    MAX_CACHED_MASKS = 100
    CLEANUP_THRESHOLD = 200  # Clean up when this many masks exist
    
    # Rendering optimizations
    MAX_VISIBLE_OVERLAYS = 50  # Increased to support more visible overlays
    PROGRESSIVE_LOADING = True
    LAZY_MASK_LOADING = True
    
    # Quality vs Performance trade-offs
    QUALITY_FILTERING_ENABLED = True
    MIN_MASK_AREA = 1000  # Filter out very small masks
    MAX_MASK_AREA = 500000  # Filter out very large masks
    MIN_STABILITY_SCORE = 0.7  # Only keep stable masks
    MIN_PREDICTED_IOU = 0.6  # Only keep high-quality masks
    
    @classmethod
    def get_performance_mode(cls) -> str:
        """Get current performance mode from environment or default."""
        return os.environ.get('IMAGE_EDITOR_PERFORMANCE_MODE', 'default')
    
    @classmethod
    def get_max_masks_for_mode(cls, mode: str = None) -> int:
        """Get maximum masks based on performance mode."""
        if mode is None:
            mode = cls.get_performance_mode()
            
        mode_limits = {
            'performance': cls.PERFORMANCE_MAX_MASKS,
            'default': cls.DEFAULT_MAX_MASKS,
            'high_performance': cls.HIGH_PERFORMANCE_MAX_MASKS
        }
        
        return mode_limits.get(mode, cls.DEFAULT_MAX_MASKS)
    
    @classmethod
    def get_optimized_settings(cls, mode: str = None) -> Dict[str, Any]:
        """Get optimized settings for a specific performance mode."""
        if mode is None:
            mode = cls.get_performance_mode()
            
        base_settings = {
            'max_masks': cls.get_max_masks_for_mode(mode),
            'ui_throttle': cls.UI_UPDATE_THROTTLE_MS,
            'batch_size': cls.BATCH_SIZE,
            'overlay_delay': cls.OVERLAY_UPDATE_DELAY,
            'enable_caching': cls.ENABLE_MASK_CACHING,
            'max_visible_overlays': cls.MAX_VISIBLE_OVERLAYS,
            'progressive_loading': cls.PROGRESSIVE_LOADING,
            'lazy_loading': cls.LAZY_MASK_LOADING,
            'quality_filtering': cls.QUALITY_FILTERING_ENABLED,
            'min_area': cls.MIN_MASK_AREA,
            'max_area': cls.MAX_MASK_AREA,
            'min_stability': cls.MIN_STABILITY_SCORE,
            'min_iou': cls.MIN_PREDICTED_IOU
        }
        
        # Mode-specific optimizations
        if mode == 'performance':
            base_settings.update({
                'ui_throttle': 200,  # More aggressive throttling
                'batch_size': 3,     # Even smaller batches
                'max_visible_overlays': 5,  # Fewer overlays
                'overlay_delay': 150,
                'min_stability': 0.8,  # Higher quality threshold
                'min_iou': 0.7
            })
        elif mode == 'high_performance':
            base_settings.update({
                'ui_throttle': 100,   # Less throttling
                'batch_size': 8,      # Larger batches
                'max_visible_overlays': 15,  # More overlays
                'overlay_delay': 50,
                'min_stability': 0.6,  # Lower quality threshold
                'min_iou': 0.5
            })
            
        return base_settings
