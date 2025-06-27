import cv2
import numpy as np
import math

class ImageProcessor:
    def __init__(self, image):
        """
        image: NumPy array in RGB format.
        """
        self.original = image.copy()
        self.current = image.copy()
        # Base image for cumulative editing - starts as original but gets updated
        # when we finish editing a mask or apply global edits
        self.base_image = image.copy()
        
        # Default parameters
        self.exposure = 0
        self.illumination = 0
        self.contrast = 1.0
        self.shadow = 0
        self.highlights = 0  # Add highlights parameter
        self.whites = 0
        self.blacks = 0
        self.saturation = 1.0
        self.texture = 0
        self.grain = 0
        self.temperature = 0
        self.rgb_curves = None  # Placeholder for custom curves
        self.curves_data = None  # RGB curves data from UI
        
        # Mask editing properties
        self.mask_editing_enabled = False
        self.current_mask = None
        
        # Optimization tracking
        self.cached_states = {}  # Cache intermediate processing states
        self.last_parameters = {
            'exposure': 0, 'illumination': 0, 'contrast': 1.0, 'shadow': 0,
            'highlights': 0, 'whites': 0, 'blacks': 0, 'saturation': 1.0,
            'texture': 0, 'grain': 0, 'temperature': 0, 'curves_data': None
        }
        self.enable_optimization = True  # Flag to enable/disable optimization

    def reset(self):
        self.current = self.original.copy()
        self.base_image = self.original.copy()
        
    def commit_edits_to_base(self, curves_data=None):
        """
        Commits current edits to the base image. This should be called
        when switching between masks or from mask editing to global editing
        to preserve the cumulative changes.
        
        Args:
            curves_data: Optional curves data to apply before committing
        """
        # Start with the current processed image
        final_image = self.current.copy()
        
        # Apply curves if provided (this ensures curves are included in the commit)
        if curves_data:
            if isinstance(curves_data, dict) and 'curves' in curves_data:
                # New format with interpolation mode
                curves = curves_data['curves']
                interpolation_mode = curves_data.get('interpolation_mode', 'Linear')
            else:
                # Old format (backward compatibility)
                curves = curves_data
                interpolation_mode = 'Linear'
            
            final_image = self.apply_rgb_curves(final_image, curves, interpolation_mode)
        
        # Commit the final image (including curves) to base
        self.base_image = final_image.copy()
        
        # Clear optimization cache when base image changes
        self.clear_optimization_cache()
        
        print(f"Committed current edits (including curves) to base image")
    
    def set_mask_editing(self, enabled, mask=None):
        """
        Enable or disable mask-based editing
        
        Args:
            enabled (bool): Whether to enable mask editing
            mask (numpy.ndarray): Binary mask array (boolean or 0-255) or None to disable
        """
        print(f"DEBUG: set_mask_editing called - enabled: {enabled}, mask provided: {mask is not None}")
        self.mask_editing_enabled = enabled
        if enabled and mask is not None:
            print(f"DEBUG: Original mask shape: {mask.shape}, dtype: {mask.dtype}, range: {mask.min()}-{mask.max()}")
            
            # Handle different mask formats
            if mask.dtype == bool:
                # Boolean mask - convert to float32 0-1 range
                self.current_mask = mask.astype(np.float32)
                print(f"DEBUG: Converted boolean mask to float32")
            else:
                # Ensure mask is binary and same dimensions as image
                if len(mask.shape) == 3:
                    mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
                    print(f"DEBUG: Converted to grayscale - shape: {mask.shape}")
                
                # Handle different numeric ranges
                if mask.max() > 1:
                    # 0-255 range mask - normalize to 0-1
                    self.current_mask = (mask > 127).astype(np.float32)
                    print(f"DEBUG: Converted 0-255 mask to 0-1 range")
                else:
                    # Already 0-1 range
                    self.current_mask = mask.astype(np.float32)
                    print(f"DEBUG: Used existing 0-1 range mask")
            
            print(f"DEBUG: Final mask shape: {self.current_mask.shape}, range: {self.current_mask.min()}-{self.current_mask.max()}")
            unique_values = np.unique(self.current_mask)
            print(f"DEBUG: Unique mask values: {unique_values}")
            non_zero_pixels = np.sum(self.current_mask > 0)
            total_pixels = self.current_mask.size
            print(f"DEBUG: Mask coverage: {non_zero_pixels}/{total_pixels} pixels ({100*non_zero_pixels/total_pixels:.1f}%)")
        else:
            self.current_mask = None
            self.mask_editing_enabled = False
            print(f"DEBUG: Mask editing disabled")

    def apply_all_edits(self):
        """
        Applies all the editing functions in sequence to the base image.
        If mask editing is enabled, applies edits only to masked areas.
        For cumulative editing, this starts from base_image instead of original.
        """
        img = self.base_image.copy()
        
        # Optimization: Check if we can use incremental processing
        if self.enable_optimization and not self.mask_editing_enabled:
            return self._apply_edits_optimized()
        
        if self.mask_editing_enabled and self.current_mask is not None:
            # Apply edits only to masked areas
            mask = self.current_mask
            
            # Create a copy of the masked area for editing
            masked_img = img.copy()
            masked_img = self.apply_exposure(masked_img, self.exposure)
            masked_img = self.apply_illumination(masked_img, self.illumination)
            masked_img = self.apply_contrast(masked_img, self.contrast)
            masked_img = self.apply_shadow(masked_img, self.shadow)
            masked_img = self.apply_highlights(masked_img, self.highlights)
            masked_img = self.apply_whites(masked_img, self.whites)
            masked_img = self.apply_blacks(masked_img, self.blacks)
            masked_img = self.apply_saturation(masked_img, self.saturation)
            masked_img = self.apply_texture(masked_img, self.texture)
            masked_img = self.apply_grain(masked_img, self.grain)
            masked_img = self.apply_temperature(masked_img, self.temperature)
            
            # Apply RGB curves if available
            if self.curves_data and 'curves' in self.curves_data:
                curves = self.curves_data['curves']
                interpolation_mode = self.curves_data.get('interpolation_mode', 'Linear')
                masked_img = self.apply_rgb_curves(masked_img, curves, interpolation_mode)
            
            # Ensure all images have the same number of channels
            if img.shape[2] != masked_img.shape[2]:
                if img.shape[2] == 4 and masked_img.shape[2] == 3:
                    # Add alpha channel to masked_img
                    alpha_channel = np.ones((masked_img.shape[0], masked_img.shape[1], 1), dtype=masked_img.dtype) * 255
                    masked_img = np.concatenate([masked_img, alpha_channel], axis=2)
                elif img.shape[2] == 3 and masked_img.shape[2] == 4:
                    # Remove alpha channel from masked_img
                    masked_img = masked_img[:, :, :3]
            
            # Blend the edited masked area back into the original image
            # Expand mask to match image channels
            if img.shape[2] == 4:
                mask_3d = np.stack([mask, mask, mask, mask], axis=2)
            else:
                mask_3d = np.stack([mask, mask, mask], axis=2)
                
            img = np.where(mask_3d, masked_img, img)
        else:
            # Apply edits to the entire image as before
            img = self.apply_exposure(img, self.exposure)
            img = self.apply_illumination(img, self.illumination)
            img = self.apply_contrast(img, self.contrast)
            img = self.apply_shadow(img, self.shadow)
            img = self.apply_highlights(img, self.highlights)
            img = self.apply_whites(img, self.whites)
            img = self.apply_blacks(img, self.blacks)
            img = self.apply_saturation(img, self.saturation)
            img = self.apply_texture(img, self.texture)
            img = self.apply_grain(img, self.grain)
            img = self.apply_temperature(img, self.temperature)
            
            # Apply RGB curves if available
            if self.curves_data and 'curves' in self.curves_data:
                curves = self.curves_data['curves']
                interpolation_mode = self.curves_data.get('interpolation_mode', 'Linear')
                img = self.apply_rgb_curves(img, curves, interpolation_mode)
        
        self.current = img
        return img
    
    def _apply_edits_optimized(self):
        """
        Optimized version that only recalculates changed parameters.
        Processing order: exposure → illumination → contrast → shadow → highlights → 
        whites → blacks → saturation → texture → grain → temperature → curves
        """
        current_params = {
            'exposure': self.exposure, 'illumination': self.illumination, 
            'contrast': self.contrast, 'shadow': self.shadow,
            'highlights': self.highlights, 'whites': self.whites, 
            'blacks': self.blacks, 'saturation': self.saturation,
            'texture': self.texture, 'grain': self.grain, 
            'temperature': self.temperature, 'curves_data': self.curves_data
        }
        
        # Find the first parameter that changed
        processing_order = ['exposure', 'illumination', 'contrast', 'shadow', 'highlights', 
                          'whites', 'blacks', 'saturation', 'texture', 'grain', 'temperature', 'curves_data']
        
        start_from = None
        for param in processing_order:
            if param == 'curves_data':
                # Special handling for curves_data - serialize for comparison
                if self._curves_data_changed(current_params[param], self.last_parameters[param]):
                    start_from = param
                    break
            else:
                if current_params[param] != self.last_parameters[param]:
                    start_from = param
                    break
        
        # If nothing changed, return cached result
        if start_from is None:
            return self.current
        
        # Find where to start processing from cache
        start_index = processing_order.index(start_from)
        
        # Get cached state before the changed parameter
        if start_index > 0:
            cache_key = processing_order[start_index - 1]
            if cache_key in self.cached_states:
                img = self.cached_states[cache_key].copy()
                print(f"✓ Using cached state from '{cache_key}' for processing '{start_from}'")
            else:
                # Instead of full processing fallback, try to find the most recent cache
                available_cache_keys = [k for k in processing_order[:start_index] if k in self.cached_states]
                if available_cache_keys:
                    # Use the most recent available cache
                    best_cache_key = available_cache_keys[-1]
                    best_cache_index = processing_order.index(best_cache_key)
                    img = self.cached_states[best_cache_key].copy()
                    print(f"✓ Cache miss for '{cache_key}', using '{best_cache_key}' instead")
                    # Update start_index to process from the available cache
                    start_index = best_cache_index + 1
                else:
                    # No cache available, start from base
                    img = self.base_image.copy()
                    start_index = 0
                    print(f"⚠ No cache available, starting from base image")
        else:
            img = self.base_image.copy()
        
        # Apply only the changed parameters and subsequent ones
        for i in range(start_index, len(processing_order)):
            param = processing_order[i]
            
            if param == 'exposure':
                img = self.apply_exposure(img, self.exposure)
            elif param == 'illumination':
                img = self.apply_illumination(img, self.illumination)
            elif param == 'contrast':
                img = self.apply_contrast(img, self.contrast)
            elif param == 'shadow':
                img = self.apply_shadow(img, self.shadow)
            elif param == 'highlights':
                img = self.apply_highlights(img, self.highlights)
            elif param == 'whites':
                img = self.apply_whites(img, self.whites)
            elif param == 'blacks':
                img = self.apply_blacks(img, self.blacks)
            elif param == 'saturation':
                img = self.apply_saturation(img, self.saturation)
            elif param == 'texture':
                img = self.apply_texture(img, self.texture)
            elif param == 'grain':
                img = self.apply_grain(img, self.grain)
            elif param == 'temperature':
                img = self.apply_temperature(img, self.temperature)
            elif param == 'curves_data':
                if self.curves_data and 'curves' in self.curves_data:
                    curves = self.curves_data['curves']
                    interpolation_mode = self.curves_data.get('interpolation_mode', 'Linear')
                    img = self.apply_rgb_curves(img, curves, interpolation_mode)
            
            # Cache intermediate state
            self.cached_states[param] = img.copy()
        
        self.current = img
        self._update_cache_and_params(current_params)
        return img
    
    def _apply_edits_full(self):
        """Full processing without optimization - fallback method"""
        img = self.base_image.copy()
        img = self.apply_exposure(img, self.exposure)
        img = self.apply_illumination(img, self.illumination)
        img = self.apply_contrast(img, self.contrast)
        img = self.apply_shadow(img, self.shadow)
        img = self.apply_highlights(img, self.highlights)
        img = self.apply_whites(img, self.whites)
        img = self.apply_blacks(img, self.blacks)
        img = self.apply_saturation(img, self.saturation)
        img = self.apply_texture(img, self.texture)
        img = self.apply_grain(img, self.grain)
        img = self.apply_temperature(img, self.temperature)
        
        if self.curves_data and 'curves' in self.curves_data:
            curves = self.curves_data['curves']
            interpolation_mode = self.curves_data.get('interpolation_mode', 'Linear')
            img = self.apply_rgb_curves(img, curves, interpolation_mode)
        
        return img
    
    def _update_cache_and_params(self, current_params):
        """Update the parameter tracking and limit cache size"""
        # Create a deep copy for curves_data to avoid reference issues
        params_copy = current_params.copy()
        if params_copy['curves_data'] is not None:
            import copy
            params_copy['curves_data'] = copy.deepcopy(params_copy['curves_data'])
        
        self.last_parameters = params_copy
        
        # Increase cache size to accommodate all parameters in the processing chain
        # We have 12 parameters total (exposure to curves_data), so cache should be at least 12
        max_cache_size = 15  # Allow some extra space for safety
        if len(self.cached_states) > max_cache_size:
            # Remove oldest entries but be more conservative about eviction
            # Only remove entries that are early in the processing chain
            processing_order = ['exposure', 'illumination', 'contrast', 'shadow', 'highlights', 
                              'whites', 'blacks', 'saturation', 'texture', 'grain', 'temperature', 'curves_data']
            
            # Sort keys by their order in processing chain, keeping later ones
            sorted_keys = []
            for param in processing_order:
                if param in self.cached_states:
                    sorted_keys.append(param)
            
            # Remove excess early entries first
            excess_count = len(self.cached_states) - max_cache_size
            if excess_count > 0:
                keys_to_remove = sorted_keys[:excess_count]
                for key in keys_to_remove:
                    del self.cached_states[key]
    
    def clear_optimization_cache(self):
        """Clear the optimization cache - call when base image changes"""
        self.cached_states.clear()
        self.last_parameters = {
            'exposure': 0, 'illumination': 0, 'contrast': 1.0, 'shadow': 0,
            'highlights': 0, 'whites': 0, 'blacks': 0, 'saturation': 1.0,
            'texture': 0, 'grain': 0, 'temperature': 0, 'curves_data': None
        }
        print("✓ Optimization cache cleared")
    
    def enable_mask_editing(self, mask=None):
        """
        Enable or disable mask editing.
        If a mask is provided, it will be used as the current mask.
        """
        self.mask_editing_enabled = True
        if mask is not None:
            self.current_mask = mask.astype(np.float32) / 255.0  # Normalize to [0,1] range

    def disable_mask_editing(self):
        """Disable mask editing."""
        self.mask_editing_enabled = False
        self.current_mask = None

    def apply_mask(self, image, mask, invert_mask=False):
        """
        Apply a mask to the image.
        If invert_mask is True, the mask will be inverted before application.
        """
        if invert_mask:
            mask = 1 - mask  # Invert the mask
        # Ensure the mask is the same size as the image
        if mask.shape[:2] != image.shape[:2]:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
        # Apply the mask: multiply image by mask (in float32), then convert back to uint8
        return cv2.convertScaleAbs(image.astype(np.float32) * mask[..., np.newaxis])
    
    def apply_highlights(self, image, value):
        """
        Adjusts mid-to-high highlight areas with smooth luminance-based masking.
        More aggressive than whites adjustment, better for highlight recovery.
        Positive values brighten highlights, negative values darken them.
        """
        if value == 0:
            return image
            
        # Convert image to float for precision
        img = image.astype(np.float32) / 255.0
        
        # Calculate proper luminance (ITU-R BT.709)
        luminance = 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]
        
        # Create smooth highlight mask with different threshold than whites
        # Highlights start affecting from 0.5 luminance (mid-to-high)
        highlight_threshold = 0.5
        mask = np.maximum(0, (luminance - highlight_threshold) / (1.0 - highlight_threshold))
        mask = mask ** 0.7  # Slightly different falloff than whites
        
        # Calculate adjustment factor (inverted: positive values brighten)
        adjustment = (value / 100.0) * 0.4  # Max 40% adjustment for highlights
        
        # Apply adjustment with smooth blending
        adjustment_3d = adjustment * mask[..., None]
        img = img + adjustment_3d
        
        # Clip and convert back
        img = np.clip(img * 255.0, 0, 255)
        return img.astype(np.uint8)

    def apply_exposure(self, image, value):
        """
        Proper exposure adjustment using exponential scaling.
        Value range: -100 to +100 (stops of light)
        """
        if value == 0:
            return image
        
        # Convert to float for precision
        img = image.astype(np.float32) / 255.0
        
        # Convert exposure value to exposure factor (2^(stops))
        # value of 100 = +1 stop, value of -100 = -1 stop
        exposure_factor = 2.0 ** (value / 100.0)
        
        # Apply exposure adjustment
        img = img * exposure_factor
        
        # Clip and convert back to uint8
        img = np.clip(img * 255.0, 0, 255)
        return img.astype(np.uint8)

    def apply_illumination(self, image, value):
        """
        Proper illumination adjustment using gamma correction with better range handling.
        Positive values brighten midtones, negative values darken them.
        Value range: -100 to +100
        """
        if value == 0:
            return image
            
        # Convert to float for precision
        img = image.astype(np.float32) / 255.0
        
        # Better gamma calculation for smoother transitions
        # Maps -100 to +100 range to approximately 0.3 to 3.0 gamma range
        if value > 0:
            gamma = 1.0 / (1.0 + value / 100.0)  # Brightening
        else:
            gamma = 1.0 - (value / 100.0)  # Darkening
        
        # Apply gamma correction
        img = np.power(img, gamma)
        
        # Clip and convert back
        img = np.clip(img * 255.0, 0, 255)
        return img.astype(np.uint8)

    def apply_contrast(self, image, value):
        """
        Professional contrast adjustment using midpoint-based scaling.
        Maintains proper midtone anchoring while adjusting contrast.
        Value range: 0.1 to 3.0 (1.0 = no change)
        """
        if value == 1.0:
            return image
            
        # Convert to float for precision
        img = image.astype(np.float32) / 255.0
        
        # Apply contrast adjustment anchored at midpoint (0.5)
        # This prevents overall brightness shifts
        img = ((img - 0.5) * value) + 0.5
        
        # Clip and convert back
        img = np.clip(img * 255.0, 0, 255)
        return img.astype(np.uint8)

    def apply_shadow(self, image, value):
        """
        Adjusts shadow areas with smooth luminance-based masking.
        Uses proper luminance calculation and smooth falloff.
        """
        if value == 0:
            return image
            
        # Convert image to float for precision
        img = image.astype(np.float32) / 255.0
        
        # Calculate proper luminance (ITU-R BT.709)
        luminance = 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]
        
        # Create smooth shadow mask with falloff
        # Shadows are typically below 0.3 luminance
        shadow_threshold = 0.3
        mask = np.maximum(0, (shadow_threshold - luminance) / shadow_threshold)
        mask = mask ** 0.5  # Smooth falloff
        
        # Calculate adjustment factor
        adjustment = (value / 100.0) * 0.3  # Max 30% adjustment
        
        # Apply adjustment with smooth blending
        adjustment_3d = adjustment * mask[..., None]
        img = img + adjustment_3d
        
        # Clip and convert back
        img = np.clip(img * 255.0, 0, 255)
        return img.astype(np.uint8)

    def apply_whites(self, image, value):
        """
        Adjusts highlight areas with smooth luminance-based masking.
        Positive values reduce highlight intensity, negative values increase it.
        """
        if value == 0:
            return image
            
        # Convert image to float for precision
        img = image.astype(np.float32) / 255.0
        
        # Calculate proper luminance (ITU-R BT.709)
        luminance = 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]
        
        # Create smooth highlight mask with falloff
        # Highlights are typically above 0.7 luminance
        highlight_threshold = 0.7
        mask = np.maximum(0, (luminance - highlight_threshold) / (1.0 - highlight_threshold))
        mask = mask ** 0.5  # Smooth falloff
        
        # Calculate adjustment factor (negative value reduces highlights)
        adjustment = -(value / 100.0) * 0.3  # Max 30% adjustment
        
        # Apply adjustment with smooth blending
        adjustment_3d = adjustment * mask[..., None]
        img = img + adjustment_3d
        
        # Clip and convert back
        img = np.clip(img * 255.0, 0, 255)
        return img.astype(np.uint8)

    def apply_blacks(self, image, value):
        """
        Adjusts very dark areas with smooth luminance-based masking.
        Positive values lighten dark areas, negative values darken them.
        """
        if value == 0:
            return image
            
        # Convert image to float for precision
        img = image.astype(np.float32) / 255.0
        
        # Calculate proper luminance (ITU-R BT.709)
        luminance = 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]
        
        # Create smooth black mask with falloff
        # Very dark areas are typically below 0.1 luminance
        black_threshold = 0.1
        mask = np.maximum(0, (black_threshold - luminance) / black_threshold)
        mask = mask ** 0.5  # Smooth falloff
        
        # Calculate adjustment factor
        adjustment = (value / 100.0) * 0.2  # Max 20% adjustment for blacks
        
        # Apply adjustment with smooth blending
        adjustment_3d = adjustment * mask[..., None]
        img = img + adjustment_3d
        
        # Clip and convert back
        img = np.clip(img * 255.0, 0, 255)
        return img.astype(np.uint8)

    def apply_saturation(self, image, value):
        # Convert to HSV, adjust the saturation, then convert back
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[..., 1] *= value
        hsv[..., 1] = np.clip(hsv[..., 1], 0, 255)
        hsv = hsv.astype(np.uint8)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    def apply_texture(self, image, value):
        """
        Enhanced texture control using unsharp masking for more natural results.
        Positive values increase texture/sharpness, negative values soften.
        """
        if value == 0:
            return image
            
        # Convert to float for precision
        img = image.astype(np.float32)
        
        if value > 0:
            # Sharpening using unsharp masking
            # Create a Gaussian blur
            blur_radius = 1.0
            blurred = cv2.GaussianBlur(img, (0, 0), blur_radius)
            
            # Calculate the difference (high-frequency details)
            detail = img - blurred
            
            # Scale the detail enhancement
            amount = value / 100.0  # 0 to 1 range
            sharpened = img + (detail * amount)
            
            # Clip and return
            result = np.clip(sharpened, 0, 255)
            return result.astype(np.uint8)
        else:
            # Softening using bilateral filter for edge-preserving smoothing
            sigma_color = abs(value) * 2  # Color similarity
            sigma_space = abs(value) * 2  # Spatial similarity
            
            # Apply bilateral filter
            result = cv2.bilateralFilter(
                image, 
                d=9,  # Diameter of pixel neighborhood
                sigmaColor=sigma_color,
                sigmaSpace=sigma_space
            )
            return result

    def apply_grain(self, image, value):
        """
        Adds realistic film grain using Gaussian noise with proper intensity scaling.
        Positive values add grain, value 0 = no grain.
        """
        if value == 0:
            return image
            
        # Convert to float for precision
        img = image.astype(np.float32)
        
        # Create realistic grain using Gaussian distribution
        grain_intensity = (value / 100.0) * 25  # Max 25 intensity
        noise = np.random.normal(0, grain_intensity, image.shape).astype(np.float32)
        
        # Add noise to image
        img = img + noise
        
        # Clip and convert back
        img = np.clip(img, 0, 255)
        return img.astype(np.uint8)

    def apply_temperature(self, image, value):
        """
        Realistic color temperature adjustment using proper color theory.
        Positive values = warmer (towards red/orange)
        Negative values = cooler (towards blue)
        """
        if value == 0:
            return image
            
        # Convert to float for precision
        img = image.astype(np.float32) / 255.0
        
        # Convert temperature adjustment to Kelvin scale
        # value range: -100 to +100 maps to roughly 2000K to 10000K
        temp_kelvin = 5500 + (value * 40)  # 5500K is neutral
        temp_kelvin = np.clip(temp_kelvin, 1000, 12000)
        
        # Calculate RGB multipliers for the given temperature
        # Based on Tanner Helland's algorithm
        temp = temp_kelvin / 100.0
        
        if temp <= 66:
            red = 255
            green = temp
            green = 99.4708025861 * np.log(green) - 161.1195681661
            if temp >= 19:
                blue = temp - 10
                blue = 138.5177312231 * np.log(blue) - 305.0447927307
            else:
                blue = 0
        else:
            red = temp - 60
            red = 329.698727446 * (red ** -0.1332047592)
            green = temp - 60
            green = 288.1221695283 * (green ** -0.0755148492)
            blue = 255
        
        # Normalize and create multipliers
        red = np.clip(red, 0, 255) / 255.0
        green = np.clip(green, 0, 255) / 255.0
        blue = np.clip(blue, 0, 255) / 255.0
        
        # Apply temperature adjustment
        img[..., 0] *= red    # Red channel
        img[..., 1] *= green  # Green channel
        img[..., 2] *= blue   # Blue channel
        
        # Clip and convert back
        img = np.clip(img * 255.0, 0, 255)
        return img.astype(np.uint8)

    def apply_rgb_curves(self, image, curves, interpolation_mode="Linear"):
        """
        Applies custom RGB curves to the image.
        curves: a dict with keys 'r', 'g', 'b'. Each value is a list of control points [(x, y), ...]
        where x and y are in [0,255]. If a channel is missing, that channel remains unchanged.
        interpolation_mode: "Linear" or "Spline"
        
        If mask editing is enabled, applies curves only to masked areas.
        """
        if curves is None:
            return image
        
        if self.mask_editing_enabled and self.current_mask is not None:
            # Apply curves only to masked areas
            result_image = image.copy()
            
            # Apply curves to the entire image first
            curves_applied = self._apply_curves_with_luminance(image, curves, interpolation_mode)
            
            # Ensure all images have the same number of channels
            if result_image.shape[2] != curves_applied.shape[2]:
                if result_image.shape[2] == 4 and curves_applied.shape[2] == 3:
                    # Add alpha channel to curves_applied
                    alpha_channel = np.ones((curves_applied.shape[0], curves_applied.shape[1], 1), dtype=curves_applied.dtype) * 255
                    curves_applied = np.concatenate([curves_applied, alpha_channel], axis=2)
                elif result_image.shape[2] == 3 and curves_applied.shape[2] == 4:
                    # Remove alpha channel from curves_applied
                    curves_applied = curves_applied[:, :, :3]
            
            # Blend the curves result back into the original using the mask
            if result_image.shape[2] == 4:
                mask_3d = np.stack([self.current_mask, self.current_mask, self.current_mask, self.current_mask], axis=2)
            else:
                mask_3d = np.stack([self.current_mask, self.current_mask, self.current_mask], axis=2)
                
            result_image = np.where(mask_3d, curves_applied, result_image)
            
            return result_image
        else:
            # Apply curves to the entire image
            return self._apply_curves_with_luminance(image, curves, interpolation_mode)
    
    def _apply_curves_to_image(self, image, curves, interpolation_mode="Linear"):
        """Helper method to apply curves to an image"""
        # Handle both RGB and RGBA images
        if image.shape[2] == 4:
            # RGBA image - split into RGB and alpha channels
            b, g, r, a = cv2.split(image)
            alpha_channel = a
        else:
            # RGB image - split normally (note: OpenCV uses BGR order)
            b, g, r = cv2.split(image)
            alpha_channel = None
        
        # Apply curves to RGB channels
        if "r" in curves and curves["r"]:
            lut_r = self.generate_lut_from_points(curves["r"], interpolation_mode)
            r = cv2.LUT(r, lut_r)
        if "g" in curves and curves["g"]:
            lut_g = self.generate_lut_from_points(curves["g"], interpolation_mode)
            g = cv2.LUT(g, lut_g)
        if "b" in curves and curves["b"]:
            lut_b = self.generate_lut_from_points(curves["b"], interpolation_mode)
            b = cv2.LUT(b, lut_b)
        
        # Merge channels back, including alpha if present
        if alpha_channel is not None:
            return cv2.merge([b, g, r, alpha_channel])
        else:
            return cv2.merge([b, g, r])
    
    def generate_lut_from_points(self, points, interpolation_mode="Linear"):
        """Generate a lookup table from control points using specified interpolation"""
        import numpy as np
        if len(points) < 2:
            return np.arange(256, dtype=np.uint8)
        
        points = sorted(points, key=lambda p: p[0])
        xs, ys = zip(*points)
        xs = np.array(xs)
        ys = np.array(ys)
        
        if interpolation_mode == "Spline" and len(points) >= 3:
            try:
                from scipy.interpolate import CubicSpline
                # Create cubic spline interpolation
                cs = CubicSpline(xs, ys, bc_type='natural')
                lut = cs(np.arange(256))
                # Clamp values to valid range
                lut = np.clip(lut, 0, 255)
            except ImportError:
                # Fallback to linear interpolation if scipy is not available
                lut = np.interp(np.arange(256), xs, ys)
            except Exception:
                # Fallback to linear interpolation on any error
                lut = np.interp(np.arange(256), xs, ys)
        else:
            # Linear interpolation (default)
            lut = np.interp(np.arange(256), xs, ys)
            
        return lut.astype(np.uint8)

    def _apply_curves_with_luminance(self, image, curves, interpolation_mode="Linear"):
        """Enhanced curves application with luminance support"""
        import cv2
        import numpy as np
        
        # Handle both RGB and RGBA images
        if image.shape[2] == 4:
            # RGBA image - split into RGB and alpha channels
            b, g, r, a = cv2.split(image)
            alpha_channel = a
        else:
            # RGB image - split normally (note: OpenCV uses BGR order)
            b, g, r = cv2.split(image)
            alpha_channel = None
        
        # Apply luminance curve first if present
        if "l" in curves and curves["l"]:
            # Calculate luminance using ITU-R BT.709 standard
            # L = 0.2126*R + 0.7152*G + 0.0722*B
            luminance = 0.2126 * r.astype(np.float32) + 0.7152 * g.astype(np.float32) + 0.0722 * b.astype(np.float32)
            
            # Generate luminance LUT
            lut_l = self.generate_lut_from_points(curves["l"], interpolation_mode)
            
            # Apply luminance curve
            # Avoid division by zero
            luminance_adjusted = cv2.LUT(luminance.astype(np.uint8), lut_l).astype(np.float32)
            
            # Calculate scaling factor for each pixel
            mask = luminance > 0
            scale_factor = np.ones_like(luminance, dtype=np.float32)
            scale_factor[mask] = luminance_adjusted[mask] / luminance[mask]
            
            # Apply scaling to each channel while preserving color ratios
            r = np.clip(r.astype(np.float32) * scale_factor, 0, 255).astype(np.uint8)
            g = np.clip(g.astype(np.float32) * scale_factor, 0, 255).astype(np.uint8)
            b = np.clip(b.astype(np.float32) * scale_factor, 0, 255).astype(np.uint8)
        
        # Apply curves to individual RGB channels
        if "r" in curves and curves["r"]:
            lut_r = self.generate_lut_from_points(curves["r"], interpolation_mode)
            r = cv2.LUT(r, lut_r)
        if "g" in curves and curves["g"]:
            lut_g = self.generate_lut_from_points(curves["g"], interpolation_mode)
            g = cv2.LUT(g, lut_g)
        if "b" in curves and curves["b"]:
            lut_b = self.generate_lut_from_points(curves["b"], interpolation_mode)
            b = cv2.LUT(b, lut_b)
        
        # Merge channels back, including alpha if present
        if alpha_channel is not None:
            return cv2.merge([b, g, r, alpha_channel])
        else:
            return cv2.merge([b, g, r])
    
    def rotate_image(self, image, angle, scale=1.0):
        """Rota la imagen completa y devuelve el canvas ajustado."""
        (h, w) = image.shape[:2]
        center = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D(center, angle, scale)
        cos = abs(M[0, 0])
        sin = abs(M[0, 1])
        
        # Asegurarnos de que las dimensiones sean enteros válidos
        nW = int((h * sin + w * cos) * scale + 0.5)  # +0.5 para redondeo adecuado
        nH = int((h * cos + w * sin) * scale + 0.5)
        
        # Ajustar la matriz de rotación para el nuevo tamaño
        M[0, 2] += (nW / 2) - center[0]
        M[1, 2] += (nH / 2) - center[1]
        
        # Verificar que nW y nH sean positivos
        if nW <= 0 or nH <= 0:
            raise ValueError("Las dimensiones calculadas para la imagen rotada no son válidas.")
    
        return cv2.warpAffine(image, M, (nW, nH), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(37,37,38,255))
    
    def get_largest_inscribed_rect_dims(self, angle):
        """Calcula las dimensiones del rectángulo máximo inscrito tras rotación."""
        w, h = self.original.shape[1], self.original.shape[0]
        if w <= 0 or h <= 0:
            return 0, 0
        angle = abs(angle) % np.pi
        if angle > np.pi / 2:
            angle = np.pi - angle
        sin_a = np.sin(angle)
        cos_a = np.cos(angle)
        width_is_longer = w >= h
        long_side = w if width_is_longer else h
        short_side = h if width_is_longer else w
        if short_side <= 2 * sin_a * cos_a * long_side:
            x = 0.5 * short_side
            wr = x / sin_a if width_is_longer else x / cos_a
            hr = x / cos_a if width_is_longer else x / sin_a
        else:
            cos_2a = cos_a * cos_a - sin_a * sin_a
            wr = (w * cos_a - h * sin_a) / cos_2a
            hr = (h * cos_a - w * sin_a) / cos_2a
        return int(wr), int(hr)

    def crop_rotate_flip(self, image, crop_rect, angle=0, flip_horizontal=False, flip_vertical=False):
        """Aplica rotación y recorte en una sola operación."""
        x, y, w, h = crop_rect
        
        # First crop the image
        cropped = image[y:y+h, x:x+w].copy()
        
        # Apply rotation if needed
        if angle != 0:
            center = (w // 2, h // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            cropped = cv2.warpAffine(cropped, rotation_matrix, (w, h), flags=cv2.INTER_LINEAR)
        
        # Apply flips if needed
        if flip_horizontal:
            cropped = cv2.flip(cropped, 1)
        if flip_vertical:
            cropped = cv2.flip(cropped, 0)
            
        return cropped

    def _curves_data_changed(self, current_curves, last_curves):
        """
        Check if curves data has actually changed by comparing the content.
        Handles None values and nested dictionary comparison properly.
        """
        # If both are None, no change
        if current_curves is None and last_curves is None:
            return False
        
        # If one is None and the other isn't, there's a change
        if current_curves is None or last_curves is None:
            return True
        
        # If neither is a dict, compare directly
        if not isinstance(current_curves, dict) or not isinstance(last_curves, dict):
            return current_curves != last_curves
        
        # Compare the 'curves' data if it exists
        current_curves_dict = current_curves.get('curves', {})
        last_curves_dict = last_curves.get('curves', {})
        
        # Compare interpolation mode
        current_interp = current_curves.get('interpolation_mode', 'Linear')
        last_interp = last_curves.get('interpolation_mode', 'Linear')
        
        if current_interp != last_interp:
            return True
        
        # Compare each channel's curves
        for channel in ['r', 'g', 'b', 'l']:
            current_channel_curves = current_curves_dict.get(channel, [])
            last_channel_curves = last_curves_dict.get(channel, [])
            
            # Convert to tuples for proper comparison
            current_points = [tuple(point) for point in current_channel_curves] if current_channel_curves else []
            last_points = [tuple(point) for point in last_channel_curves] if last_channel_curves else []
            
            if current_points != last_points:
                return True
        
        return False
