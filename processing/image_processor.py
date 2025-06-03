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
        # Default parameters
        self.exposure = 0
        self.illumination = 0
        self.contrast = 1.0
        self.shadow = 0
        self.whites = 0
        self.blacks = 0
        self.saturation = 1.0
        self.texture = 0
        self.grain = 0
        self.temperature = 0
        self.rgb_curves = None  # Placeholder for custom curves
        
        # Mask editing properties
        self.mask_editing_enabled = False
        self.current_mask = None

    def reset(self):
        self.current = self.original.copy()
    
    def set_mask_editing(self, enabled, mask=None):
        """
        Enable or disable mask-based editing
        
        Args:
            enabled (bool): Whether to enable mask editing
            mask (numpy.ndarray): Binary mask array (0-255) or None to disable
        """
        self.mask_editing_enabled = enabled
        if enabled and mask is not None:
            # Ensure mask is binary and same dimensions as image
            if len(mask.shape) == 3:
                mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
            
            # Normalize mask to 0-1 range
            self.current_mask = (mask > 127).astype(np.float32)
        else:
            self.current_mask = None
            self.mask_editing_enabled = False

    def apply_all_edits(self):
        """
        Applies all the editing functions in sequence to the original image.
        If mask editing is enabled, applies edits only to masked areas.
        """
        img = self.original.copy()
        
        if self.mask_editing_enabled and self.current_mask is not None:
            # Apply edits only to masked areas
            mask = self.current_mask
            
            # Create a copy of the masked area for editing
            masked_img = img.copy()
            masked_img = self.apply_exposure(masked_img, self.exposure)
            masked_img = self.apply_illumination(masked_img, self.illumination)
            masked_img = self.apply_contrast(masked_img, self.contrast)
            masked_img = self.apply_shadow(masked_img, self.shadow)
            masked_img = self.apply_whites(masked_img, self.whites)
            masked_img = self.apply_blacks(masked_img, self.blacks)
            masked_img = self.apply_saturation(masked_img, self.saturation)
            masked_img = self.apply_texture(masked_img, self.texture)
            masked_img = self.apply_grain(masked_img, self.grain)
            masked_img = self.apply_temperature(masked_img, self.temperature)
            
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
            img = self.apply_whites(img, self.whites)
            img = self.apply_blacks(img, self.blacks)
            img = self.apply_saturation(img, self.saturation)
            img = self.apply_texture(img, self.texture)
            img = self.apply_grain(img, self.grain)
            img = self.apply_temperature(img, self.temperature)
        
        self.current = img
        return img

    def apply_exposure(self, image, value):
        # Adjust exposure by adding a constant brightness (beta value)
        return cv2.convertScaleAbs(image, alpha=1.0, beta=value)

    def apply_illumination(self, image, value):
        # Simple gamma correction as a proxy for illumination adjustment
        gamma = 1.0 + value / 100.0
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in range(256)]).astype("uint8")
        return cv2.LUT(image, table)

    def apply_contrast(self, image, value):
        # Adjust contrast by scaling pixel values
        return cv2.convertScaleAbs(image, alpha=value, beta=0)

    def apply_shadow(self, image, value):
        """
        Adjusts shadow areas (darker regions). A positive value brightens dark areas.
        We compute a luminance mask and then blend in an adjustment.
        """
        # Convert image to float for precision.
        img = image.astype(np.float32)
        # Compute approximate luminance as the mean of RGB channels.
        luminance = np.mean(img, axis=2)
        # Create a mask: consider pixels with luminance below a threshold (e.g., 128)
        mask = (luminance < 128).astype(np.float32)
        # Normalize the slider value to an adjustment factor.
        # For instance, if value=100 then add up to 60% of 255 (~153)
        adjustment = (value / 100.0) * 153  # tweak multiplier as needed
        # For dark pixels, add the adjustment (mask has shape [H, W], expand to 3 channels)
        img += adjustment * mask[..., None]
        img = np.clip(img, 0, 255)
        return img.astype(np.uint8)

    def apply_whites(self, image, value):
        """
        Adjusts highlight areas (bright regions). A positive value will reduce brightness in highlights.
        We create a mask for pixels above a threshold (e.g., luminance > 200).
        """
        img = image.astype(np.float32)
        luminance = np.mean(img, axis=2)
        mask = (luminance > 200).astype(np.float32)
        # Normalize value: if value=100, then subtract up to 60% of 255 (~153) from highlights.
        adjustment = (value / 100.0) * 153  # tweak multiplier as needed
        # Subtract adjustment only in bright areas.
        img -= adjustment * mask[..., None]
        img = np.clip(img, 0, 255)
        return img.astype(np.uint8)

    def apply_blacks(self, image, value):
        """
        Adjusts very dark areas. A positive value lightens dark areas.
        We create a mask for pixels below a low luminance threshold (e.g., < 50).
        """
        img = image.astype(np.float32)
        luminance = np.mean(img, axis=2)
        mask = (luminance < 50).astype(np.float32)
        adjustment = (value / 100.0) * 153  # tweak multiplier as needed
        img += adjustment * mask[..., None]
        img = np.clip(img, 0, 255)
        return img.astype(np.uint8)

    def apply_saturation(self, image, value):
        # Convert to HSV, adjust the saturation, then convert back
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[..., 1] *= value
        hsv[..., 1] = np.clip(hsv[..., 1], 0, 255)
        hsv = hsv.astype(np.uint8)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    def apply_texture(self, image, value):
        # Apply a simple sharpening filter as a placeholder for texture control.
        if value == 0:
            return image
        kernel = np.array([[-1, -1, -1],
                           [-1, 9 + value, -1],
                           [-1, -1, -1]])
        return cv2.filter2D(image, -1, kernel)

    def apply_grain(self, image, value):
        # Add random noise to simulate grain.
        if value == 0:
            return image
        noise = np.random.randint(0, value, image.shape, dtype='uint8')
        return cv2.add(image, noise)

    def apply_temperature(self, image, value):
        # Adjust temperature by shifting the color channels.
        b, g, r = cv2.split(image)
        if value > 0:
            r = cv2.add(r, np.full(r.shape, value, dtype=np.uint8))
        else:
            b = cv2.add(b, np.full(b.shape, -value, dtype=np.uint8))
        return cv2.merge([b, g, r])

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
            curves_applied = self._apply_curves_to_image(image, curves, interpolation_mode)
            
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
            return self._apply_curves_to_image(image, curves, interpolation_mode)
    
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
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(100, 100, 100, 255))
    
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

    def apply_exposure_to_mask(self, value):
        """Apply exposure adjustment to the current mask."""
        if self.current_mask is not None:
            # Scale the exposure value to the range of the mask
            scaled_value = value * 255.0 / 100.0
            # Adjust the mask: clip values to [0,255] range
            self.current_mask = np.clip(self.current_mask * scaled_value, 0, 255).astype(np.uint8)

    def apply_illumination_to_mask(self, value):
        """Apply illumination (gamma correction) to the current mask."""
        if self.current_mask is not None:
            gamma = 1.0 + value / 100.0
            invGamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** invGamma) * 255 for i in range(256)]).astype("uint8")
            self.current_mask = cv2.LUT(self.current_mask, table)

    def apply_contrast_to_mask(self, value):
        """Apply contrast adjustment to the current mask."""
        if self.current_mask is not None:
            self.current_mask = cv2.convertScaleAbs(self.current_mask, alpha=value, beta=0)

    def apply_shadow_to_mask(self, value):
        """
        Adjusts shadow areas of the mask (darker regions). A positive value brightens dark areas.
        We compute a luminance mask and then blend in an adjustment.
        """
        if self.current_mask is not None:
            # Convert mask to float for precision.
            mask_f = self.current_mask.astype(np.float32)
            # Compute approximate luminance as the mean of RGB channels.
            luminance = np.mean(mask_f, axis=2)
            # Create a mask: consider pixels with luminance below a threshold (e.g., 128)
            shadow_mask = (luminance < 128).astype(np.float32)
            # Normalize the slider value to an adjustment factor.
            adjustment = (value / 100.0) * 153  # tweak multiplier as needed
            # For dark pixels, add the adjustment (shadow_mask has shape [H, W], expand to 3 channels)
            mask_f += adjustment * shadow_mask[..., None]
            mask_f = np.clip(mask_f, 0, 255)
            self.current_mask = mask_f.astype(np.uint8)

    def apply_whites_to_mask(self, value):
        """
        Adjusts highlight areas of the mask (bright regions). A positive value will reduce brightness in highlights.
        We create a mask for pixels above a threshold (e.g., luminance > 200).
        """
        if self.current_mask is not None:
            mask_f = self.current_mask.astype(np.float32)
            luminance = np.mean(mask_f, axis=2)
            highlight_mask = (luminance > 200).astype(np.float32)
            # Normalize value: if value=100, then subtract up to 60% of 255 (~153) from highlights.
            adjustment = (value / 100.0) * 153  # tweak multiplier as needed
            # Subtract adjustment only in bright areas.
            mask_f -= adjustment * highlight_mask[..., None]
            mask_f = np.clip(mask_f, 0, 255)
            self.current_mask = mask_f.astype(np.uint8)

    def apply_blacks_to_mask(self, value):
        """
        Adjusts very dark areas of the mask. A positive value lightens dark areas.
        We create a mask for pixels below a low luminance threshold (e.g., < 50).
        """
        if self.current_mask is not None:
            mask_f = self.current_mask.astype(np.float32)
            luminance = np.mean(mask_f, axis=2)
            dark_mask = (luminance < 50).astype(np.float32)
            adjustment = (value / 100.0) * 153  # tweak multiplier as needed
            mask_f += adjustment * dark_mask[..., None]
            mask_f = np.clip(mask_f, 0, 255)
            self.current_mask = mask_f.astype(np.uint8)

    def apply_saturation_to_mask(self, value):
        # Convert mask to HSV, adjust the saturation, then convert back
        if self.current_mask is not None:
            hsv = cv2.cvtColor(self.current_mask, cv2.COLOR_RGB2HSV).astype(np.float32)
            hsv[..., 1] *= value
            hsv[..., 1] = np.clip(hsv[..., 1], 0, 255)
            hsv = hsv.astype(np.uint8)
            self.current_mask = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    def apply_texture_to_mask(self, value):
        # Apply a simple sharpening filter as a placeholder for texture control.
        if value == 0 or self.current_mask is None:
            return
        kernel = np.array([[-1, -1, -1],
                           [-1, 9 + value, -1],
                           [-1, -1, -1]])
        self.current_mask = cv2.filter2D(self.current_mask, -1, kernel)

    def apply_grain_to_mask(self, value):
        # Add random noise to simulate grain.
        if value == 0 or self.current_mask is None:
            return
        noise = np.random.randint(0, value, self.current_mask.shape, dtype='uint8')
        self.current_mask = cv2.add(self.current_mask, noise)

    def apply_temperature_to_mask(self, value):
        # Adjust temperature by shifting the color channels.
        if self.current_mask is not None:
            b, g, r = cv2.split(self.current_mask)
            if value > 0:
                r = cv2.add(r, np.full(r.shape, value, dtype=np.uint8))
            else:
                b = cv2.add(b, np.full(b.shape, -value, dtype=np.uint8))
            self.current_mask = cv2.merge([b, g, r])

    def apply_rgb_curves_to_mask(self, curves, interpolation_mode="Linear"):
        """
        Applies custom RGB curves to the current mask.
        curves: a dict with keys 'r', 'g', 'b'. Each value is a list of control points [(x, y), ...]
        where x and y are in [0,255]. If a channel is missing, that channel remains unchanged.
        interpolation_mode: "Linear" or "Spline"
        """
        if self.current_mask is None or curves is None:
            return
        # Split channels (note: OpenCV uses BGR order)
        b, g, r = cv2.split(self.current_mask)
        if "r" in curves and curves["r"]:
            lut_r = self.generate_lut_from_points(curves["r"], interpolation_mode)
            r = cv2.LUT(r, lut_r)
        if "g" in curves and curves["g"]:
            lut_g = self.generate_lut_from_points(curves["g"], interpolation_mode)
            g = cv2.LUT(g, lut_g)
        if "b" in curves and curves["b"]:
            lut_b = self.generate_lut_from_points(curves["b"], interpolation_mode)
            b = cv2.LUT(b, lut_b)
        self.current_mask = cv2.merge([b, g, r])
