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

    def reset(self):
        self.current = self.original.copy()

    def apply_all_edits(self):
        """
        Applies all the editing functions in sequence to the original image.
        """
        img = self.original.copy()
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
        # For crop/rotate and RGB curves, you might trigger separate workflows.
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

    def apply_rgb_curves(self, image, curves):
        """
        Applies custom RGB curves to the image.
        curves: a dict with keys 'r', 'g', 'b'. Each value is a list of control points [(x, y), ...]
        where x and y are in [0,255]. If a channel is missing, that channel remains unchanged.
        """
        if curves is None:
            return image
        # Split channels (note: OpenCV uses BGR order)
        b, g, r = cv2.split(image)
        if "r" in curves and curves["r"]:
            lut_r = self.generate_lut_from_points(curves["r"])
            r = cv2.LUT(r, lut_r)
        if "g" in curves and curves["g"]:
            lut_g = self.generate_lut_from_points(curves["g"])
            g = cv2.LUT(g, lut_g)
        if "b" in curves and curves["b"]:
            lut_b = self.generate_lut_from_points(curves["b"])
            b = cv2.LUT(b, lut_b)
        return cv2.merge([b, g, r])
    
    def generate_lut_from_points(self, points):
        # Simple linear interpolation for a 256-entry LUT.
        import numpy as np
        if len(points) < 2:
            return np.arange(256, dtype=np.uint8)
        points = sorted(points, key=lambda p: p[0])
        xs, ys = zip(*points)
        xs = np.array(xs)
        ys = np.array(ys)
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

    def crop_rotate_flip(self, image, crop_rect, angle, flip_horizontal=False, flip_vertical=False):
        """Aplica rotación y recorte en una sola operación."""
        rotated = self.rotate_image(image, angle)
        x, y, w, h = crop_rect
        cropped = rotated[y:y+h, x:x+w]
        if flip_horizontal:
            cropped = cv2.flip(cropped, 1)
        if flip_vertical:
            cropped = cv2.flip(cropped, 0)
        return cropped
