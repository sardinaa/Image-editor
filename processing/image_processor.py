import cv2
import numpy as np

class ImageProcessor:
    def __init__(self, image):
        """
        image: NumPy array in RGB format.
        """
        self.original = image.copy()
        self.current = image.copy()
        
        # Complete edit history - all edits are stored and reapplied from original
        self.committed_global_edits = {
            'exposure': 0,
            'illumination': 0,
            'contrast': 1.0,
            'shadow': 0,
            'highlights': 0,
            'whites': 0,
            'blacks': 0,
            'saturation': 1.0,
            'texture': 0,
            'grain': 0,
            'temperature': 0,
            'curves_data': None
        }
        
        # Committed mask edits - stored per mask
        self.committed_mask_edits = {}  # mask_id -> edit_data
        
        # Current editing parameters (temporary, not yet committed)
        self.exposure = 0
        self.illumination = 0
        self.contrast = 1.0
        self.shadow = 0
        self.highlights = 0
        self.whites = 0
        self.blacks = 0
        self.saturation = 1.0
        self.texture = 0
        self.grain = 0
        self.temperature = 0
        self.curves_data = None
        
        # Mask editing properties
        self.mask_editing_enabled = False
        self.current_mask = None
        self.current_mask_id = None
        
        # Optimization tracking
        self.cached_states = {}
        self.last_full_state_hash = None
        self.enable_optimization = True

    def reset(self):
        """Reset all edits and return to original image."""
        self.current = self.original.copy()
        
        # Reset all committed edits
        self.committed_global_edits = {
            'exposure': 0,
            'illumination': 0,
            'contrast': 1.0,
            'shadow': 0,
            'highlights': 0,
            'whites': 0,
            'blacks': 0,
            'saturation': 1.0,
            'texture': 0,
            'grain': 0,
            'temperature': 0,
            'curves_data': None
        }
        self.committed_mask_edits.clear()
        
        # Reset current parameters
        self.exposure = 0
        self.illumination = 0
        self.contrast = 1.0
        self.shadow = 0
        self.highlights = 0
        self.whites = 0
        self.blacks = 0
        self.saturation = 1.0
        self.texture = 0
        self.grain = 0
        self.temperature = 0
        self.curves_data = None
        
        # Reset mask editing
        self.mask_editing_enabled = False
        self.current_mask = None
        self.current_mask_id = None
        
        # Clear cache
        self.clear_optimization_cache()
        
    def commit_edits_to_base(self, curves_data=None):
        """
        Commits current global edits to the committed global edits.
        This should be called when switching from global editing to mask editing.
        """
        # Store current global parameters as committed
        self.committed_global_edits.update({
            'exposure': self.exposure,
            'illumination': self.illumination,
            'contrast': self.contrast,
            'shadow': self.shadow,
            'highlights': self.highlights,
            'whites': self.whites,
            'blacks': self.blacks,
            'saturation': self.saturation,
            'texture': self.texture,
            'grain': self.grain,
            'temperature': self.temperature,
            'curves_data': curves_data if curves_data else self.curves_data
        })
        
        # Clear optimization cache when edits change
        self.clear_optimization_cache()
    
    def commit_mask_edits(self):
        """
        Commits current mask edits to the committed mask edits.
        """
        if self.mask_editing_enabled and self.current_mask_id is not None:
            self.committed_mask_edits[self.current_mask_id] = {
                'mask': self.current_mask.copy() if self.current_mask is not None else None,
                'exposure': self.exposure,
                'illumination': self.illumination,
                'contrast': self.contrast,
                'shadow': self.shadow,
                'highlights': self.highlights,
                'whites': self.whites,
                'blacks': self.blacks,
                'saturation': self.saturation,
                'texture': self.texture,
                'grain': self.grain,
                'temperature': self.temperature,
                'curves_data': self.curves_data
            }
        
        # Clear optimization cache when edits change
        self.clear_optimization_cache()
    
    def delete_mask_edits(self, mask_id):
        """
        Delete committed edits for a specific mask and adjust indices for subsequent masks.
        
        Args:
            mask_id: The mask ID to delete (typically an integer index)
        """
        # Remove committed edits for the deleted mask
        if mask_id in self.committed_mask_edits:
            del self.committed_mask_edits[mask_id]
        
        # Adjust mask IDs for masks after the deleted one (only for integer IDs)
        if isinstance(mask_id, int):
            new_committed_mask_edits = {}
            for mid, edit_data in self.committed_mask_edits.items():
                if isinstance(mid, int) and mid > mask_id:
                    # Shift index down by 1
                    new_committed_mask_edits[mid - 1] = edit_data
                elif isinstance(mid, int) and mid < mask_id:
                    # Keep same index
                    new_committed_mask_edits[mid] = edit_data
                else:
                    # Handle non-integer mask IDs
                    new_committed_mask_edits[mid] = edit_data
            
            self.committed_mask_edits = new_committed_mask_edits
        
        # Update current mask if it was the deleted one or needs index adjustment
        if isinstance(self.current_mask_id, int) and isinstance(mask_id, int):
            if self.current_mask_id == mask_id:
                self.current_mask_id = None
                self.mask_editing_enabled = False
                self.current_mask = None
            elif self.current_mask_id > mask_id:
                self.current_mask_id -= 1
        
        # Clear optimization cache
        self.clear_optimization_cache()
    
    def load_global_parameters(self):
        """
        Loads committed global parameters into current parameters for global editing mode.
        """
        self.exposure = self.committed_global_edits['exposure']
        self.illumination = self.committed_global_edits['illumination']
        self.contrast = self.committed_global_edits['contrast']
        self.shadow = self.committed_global_edits['shadow']
        self.highlights = self.committed_global_edits['highlights']
        self.whites = self.committed_global_edits['whites']
        self.blacks = self.committed_global_edits['blacks']
        self.saturation = self.committed_global_edits['saturation']
        self.texture = self.committed_global_edits['texture']
        self.grain = self.committed_global_edits['grain']
        self.temperature = self.committed_global_edits['temperature']
        self.curves_data = self.committed_global_edits['curves_data']
        
        # Clear optimization cache as parameters changed
        self.clear_optimization_cache()
    
    def load_mask_parameters(self, mask_id):
        """
        Loads committed mask parameters for the specified mask.
        Returns True if parameters were loaded, False if no committed edits exist.
        """
        if mask_id in self.committed_mask_edits:
            edit_data = self.committed_mask_edits[mask_id]
            
            self.exposure = edit_data['exposure']
            self.illumination = edit_data['illumination']
            self.contrast = edit_data['contrast']
            self.shadow = edit_data['shadow']
            self.highlights = edit_data['highlights']
            self.whites = edit_data['whites']
            self.blacks = edit_data['blacks']
            self.saturation = edit_data['saturation']
            self.texture = edit_data['texture']
            self.grain = edit_data['grain']
            self.temperature = edit_data['temperature']
            self.curves_data = edit_data['curves_data']
            
            # Clear optimization cache as parameters changed
            self.clear_optimization_cache()
            return True
        else:
            # Reset to defaults for new mask
            self.reset_current_parameters()
            return False
    
    def reset_current_parameters(self):
        """
        Resets current parameters to default values.
        """
        self.exposure = 0
        self.illumination = 0
        self.contrast = 1.0
        self.shadow = 0
        self.highlights = 0
        self.whites = 0
        self.blacks = 0
        self.saturation = 1.0
        self.texture = 0
        self.grain = 0
        self.temperature = 0
        self.curves_data = None
        
        # Clear optimization cache as parameters changed
        self.clear_optimization_cache()
    
    def set_mask_editing(self, enabled, mask=None, mask_id=None):
        """
        Enable or disable mask-based editing
        
        Args:
            enabled (bool): Whether to enable mask editing
            mask (numpy.ndarray): Binary mask array (boolean or 0-255) or None to disable
            mask_id: Unique identifier for the mask (for tracking edits)
        """
        self.mask_editing_enabled = enabled
        self.current_mask_id = mask_id
        
        if enabled and mask is not None:
            # Handle different mask formats
            if mask.dtype == bool:
                # Boolean mask - convert to float32 0-1 range
                self.current_mask = mask.astype(np.float32)
            else:
                # Ensure mask is binary and same dimensions as image
                if len(mask.shape) == 3:
                    mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
                
                # Handle different numeric ranges
                if mask.max() > 1:
                    # 0-255 range mask - normalize to 0-1
                    self.current_mask = (mask > 127).astype(np.float32)
                else:
                    # Already 0-1 range
                    self.current_mask = mask.astype(np.float32)
        else:
            self.current_mask = None
            self.current_mask_id = None
            self.mask_editing_enabled = False
    
    def switch_to_global_editing(self):
        """
        Switches to global editing mode. Loads global parameters and ensures
        proper state management.
        """
        # If we were in mask editing, commit any changes first
        if self.mask_editing_enabled:
            self.commit_mask_edits()
        
        self.mask_editing_enabled = False
        self.current_mask = None
        self.current_mask_id = None
        
        # Load global parameters into current parameters
        self.load_global_parameters()
        
        # Clear optimization cache for fresh start
        self.clear_optimization_cache()
    
    def switch_to_mask_editing(self, mask=None, mask_id=None):
        """
        Switches to mask editing mode. Commits any pending global changes
        and sets up for mask editing.
        
        Args:
            mask: Binary mask for editing
            mask_id: Unique identifier for the mask
        """
        # If we were in global editing mode, commit current changes
        if not self.mask_editing_enabled:
            self.commit_edits_to_base()
        
        # Set up mask editing
        self.set_mask_editing(True, mask, mask_id)
        
        # Try to load existing mask parameters, or reset to defaults
        if not self.load_mask_parameters(mask_id):
            self.reset_current_parameters()
    
    def finalize_mask_edits(self):
        """
        Finalizes mask edits by committing them to the edit history.
        This should be called when switching between masks or back to global editing.
        """
        if self.mask_editing_enabled and self.current_mask_id is not None:
            self.commit_mask_edits()
            
            # Reset current parameters since they're now committed
            self.reset_current_parameters()
            
            # Clear optimization cache
            self.clear_optimization_cache()


    def apply_all_edits(self):
        """
        Applies all edits from scratch starting with the original image.
        This ensures complete persistence of all edits across mode switches.
        
        The processing order is:
        1. Start with original image
        2. Apply committed global edits OR current global parameters (not both)
        3. Apply all committed mask edits
        4. Apply current mask parameters if in mask editing mode
        """
        # Always start from the original image
        img = self.original.copy()
        
        # Step 1: Apply global edits
        if self.mask_editing_enabled:
            # In mask editing mode: apply committed global edits as the base
            img = self._apply_parameter_set(img, self.committed_global_edits)
        else:
            # In global editing mode: apply current global parameters
            # (these may be the same as committed if just switched, or different if editing)
            current_global_params = {
                'exposure': self.exposure,
                'illumination': self.illumination,
                'contrast': self.contrast,
                'shadow': self.shadow,
                'highlights': self.highlights,
                'whites': self.whites,
                'blacks': self.blacks,
                'saturation': self.saturation,
                'texture': self.texture,
                'grain': self.grain,
                'temperature': self.temperature,
                'curves_data': self.curves_data
            }
            img = self._apply_parameter_set(img, current_global_params)
        
        # Step 2: Apply committed mask edits, excluding the one currently being edited
        for mask_id, edit_data in self.committed_mask_edits.items():
            # Skip the current mask if we're editing it - its current parameters will be applied instead
            if (edit_data['mask'] is not None and 
                not (self.mask_editing_enabled and mask_id == self.current_mask_id)):
                img = self._apply_mask_edit(img, edit_data)
        
        # Step 3: Apply current mask parameters if in mask editing mode
        if self.mask_editing_enabled and self.current_mask is not None:
            # Apply current mask edits (these replace any committed edits for this mask)
            current_edit_data = {
                'mask': self.current_mask,
                'exposure': self.exposure,
                'illumination': self.illumination,
                'contrast': self.contrast,
                'shadow': self.shadow,
                'highlights': self.highlights,
                'whites': self.whites,
                'blacks': self.blacks,
                'saturation': self.saturation,
                'texture': self.texture,
                'grain': self.grain,
                'temperature': self.temperature,
                'curves_data': self.curves_data
            }
            img = self._apply_mask_edit(img, current_edit_data)
        
        self.current = img
        return img
    
    def _apply_parameter_set(self, image, params):
        """
        Apply a set of parameters to an image.
        
        Args:
            image: Input image
            params: Dictionary of parameters to apply
            
        Returns:
            Processed image
        """
        img = image.copy()
        
        # Apply edits in processing order
        img = self.apply_exposure(img, params.get('exposure', 0))
        img = self.apply_illumination(img, params.get('illumination', 0))
        img = self.apply_contrast(img, params.get('contrast', 1.0))
        img = self.apply_shadow(img, params.get('shadow', 0))
        img = self.apply_highlights(img, params.get('highlights', 0))
        img = self.apply_whites(img, params.get('whites', 0))
        img = self.apply_blacks(img, params.get('blacks', 0))
        img = self.apply_saturation(img, params.get('saturation', 1.0))
        img = self.apply_texture(img, params.get('texture', 0))
        img = self.apply_grain(img, params.get('grain', 0))
        img = self.apply_temperature(img, params.get('temperature', 0))
        
        # Apply curves if available
        curves_data = params.get('curves_data')
        if curves_data and 'curves' in curves_data:
            curves = curves_data['curves']
            interpolation_mode = curves_data.get('interpolation_mode', 'Linear')
            img = self.apply_rgb_curves(img, curves, interpolation_mode)
        
        return img
    
    def _apply_mask_edit(self, image, edit_data):
        """
        Apply mask-specific edits to an image.
        
        Args:
            image: Input image
            edit_data: Dictionary containing mask and edit parameters
            
        Returns:
            Processed image
        """
        mask = edit_data['mask']
        if mask is None:
            return image
        
        # Create a copy of the image for masked editing
        img = image.copy()
        
        # Apply edits to a copy of the entire image
        params = {
            'exposure': edit_data.get('exposure', 0),
            'illumination': edit_data.get('illumination', 0),
            'contrast': edit_data.get('contrast', 1.0),
            'shadow': edit_data.get('shadow', 0),
            'highlights': edit_data.get('highlights', 0),
            'whites': edit_data.get('whites', 0),
            'blacks': edit_data.get('blacks', 0),
            'saturation': edit_data.get('saturation', 1.0),
            'texture': edit_data.get('texture', 0),
            'grain': edit_data.get('grain', 0),
            'temperature': edit_data.get('temperature', 0),
            'curves_data': edit_data.get('curves_data')
        }
        
        masked_img = self._apply_parameter_set(img, params)
        
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
        
        return img
    
    def get_display_image(self):
        """
        Returns the current display image without modifying internal state.
        This is the image that should be shown in the UI.
        """
        return self.apply_all_edits()
    
    def clear_optimization_cache(self):
        """Clear the optimization cache."""
        self.cached_states.clear()
        self.last_full_state_hash = None
    
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

    def apply_rgb_curves(self, image, curves, interpolation_mode="Spline"):
        """
        Applies custom RGB curves to the image.
        curves: a dict with keys 'r', 'g', 'b', 'l'. Each value is a list of control points [(x, y), ...]
        where x and y are in [0,255]. If a channel is missing, that channel remains unchanged.
        interpolation_mode: "Linear" or "Spline"
        
        This method now works directly with the provided image and doesn't handle masking internally.
        Masking is handled at the apply_all_edits level.
        """
        if curves is None:
            return image
        
        # Apply curves to the entire provided image
        return self._apply_curves_with_luminance(image, curves, interpolation_mode)
    
    def generate_lut_from_points(self, points, interpolation_mode="Spline"):
        """Generate a lookup table from control points using specified interpolation"""
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

    def _apply_curves_with_luminance(self, image, curves, interpolation_mode="Spline"):
        """Enhanced curves application with luminance support"""        
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

    def _params_equal(self, params1, params2):
        """
        Compare two parameter dictionaries for equality.
        
        Args:
            params1: First parameter dictionary
            params2: Second parameter dictionary
            
        Returns:
            bool: True if parameters are equal, False otherwise
        """
        # List of parameter keys to compare (excluding 'mask')
        param_keys = [
            'exposure', 'illumination', 'contrast', 'shadow', 'highlights',
            'whites', 'blacks', 'saturation', 'texture', 'grain', 'temperature'
        ]
        
        # Compare each parameter
        for key in param_keys:
            if abs(params1.get(key, 0) - params2.get(key, 0)) > 1e-6:
                return False
        
        # Compare curves data
        curves1 = params1.get('curves_data')
        curves2 = params2.get('curves_data')
        
        if curves1 is None and curves2 is None:
            return True
        elif curves1 is None or curves2 is None:
            return False
        else:
            # Compare curve data structures
            try:
                import numpy as np
                if isinstance(curves1, dict) and isinstance(curves2, dict):
                    for channel in ['rgb', 'red', 'green', 'blue']:
                        curve1 = curves1.get(channel)
                        curve2 = curves2.get(channel)
                        if curve1 is None and curve2 is None:
                            continue
                        elif curve1 is None or curve2 is None:
                            return False
                        elif not np.allclose(curve1, curve2, atol=1e-6):
                            return False
                    return True
                else:
                    return curves1 == curves2
            except:
                # Fallback to simple equality check
                return curves1 == curves2
