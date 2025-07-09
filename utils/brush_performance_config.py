"""
Brush Tool Performance Configuration

Centralized configuration for brush tool performance settings.
"""

class BrushPerformanceConfig:
    """Configuration for brush tool performance optimizations."""
    
    # Timing settings (in milliseconds)
    STROKE_UPDATE_THROTTLE_MS = 8      # ~120 FPS max for stroke updates
    DISPLAY_UPDATE_THROTTLE_MS = 25    # ~40 FPS for display updates  
    CURSOR_UPDATE_THROTTLE_MS = 16     # ~60 FPS for cursor movement
    
    # Distance settings
    MIN_STROKE_DISTANCE = 2.0          # Minimum pixel distance between stroke points
    
    # Cache settings
    BRUSH_CACHE_SIZE = 20              # Number of brush masks to cache
    COORDINATE_CACHE_TOLERANCE = 1.0   # Pixel tolerance for coordinate caching
    
    # Brush mask optimization
    MAX_INTERPOLATION_STEPS = 50       # Limit interpolation steps for performance
    BRUSH_STEP_SIZE_MULTIPLIER = 0.3   # Step size as fraction of brush size
    
    # Texture update settings
    DELAYED_UPDATE_FRAMES = 1          # Frames to delay for throttled updates
    
    @classmethod
    def get_performance_mode_settings(cls, mode: str = 'default'):
        """Get performance settings based on mode.
        
        Args:
            mode: 'fast' for maximum performance, 'default' for balanced, 'smooth' for quality
        """
        if mode == 'fast':
            return {
                'stroke_throttle': 16,      # ~60 FPS
                'display_throttle': 40,     # ~25 FPS  
                'cursor_throttle': 25,      # ~40 FPS
                'min_distance': 3.0,        # More aggressive filtering
                'cache_size': 15,           # Smaller cache
                'max_steps': 30             # Fewer interpolation steps
            }
        elif mode == 'smooth':
            return {
                'stroke_throttle': 4,       # ~250 FPS
                'display_throttle': 16,     # ~60 FPS
                'cursor_throttle': 8,       # ~120 FPS
                'min_distance': 1.0,        # Less filtering
                'cache_size': 30,           # Larger cache
                'max_steps': 100            # More interpolation steps
            }
        else:  # default
            return {
                'stroke_throttle': cls.STROKE_UPDATE_THROTTLE_MS,
                'display_throttle': cls.DISPLAY_UPDATE_THROTTLE_MS,
                'cursor_throttle': cls.CURSOR_UPDATE_THROTTLE_MS,
                'min_distance': cls.MIN_STROKE_DISTANCE,
                'cache_size': cls.BRUSH_CACHE_SIZE,
                'max_steps': cls.MAX_INTERPOLATION_STEPS
            }
    
    @classmethod
    def apply_settings_to_renderer(cls, renderer, mode: str = 'default'):
        """Apply performance settings to a brush renderer."""
        settings = cls.get_performance_mode_settings(mode)
        
        renderer.update_throttle_ms = settings['stroke_throttle']
        renderer.display_update_throttle_ms = settings['display_throttle']
        renderer._min_stroke_distance = settings['min_distance']
        renderer._max_cache_size = settings['cache_size']
        
        # Clear and resize cache if needed
        if len(renderer._brush_cache) > settings['cache_size']:
            renderer._brush_cache.clear()
