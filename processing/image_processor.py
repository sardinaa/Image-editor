import cv2
import numpy as np

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


    def crop_rotate_flip(self, image, crop_rect, angle, flip_horizontal=False, flip_vertical=False):
        """
        Crop, rotate, and optionally flip the image.
        crop_rect: (x, y, width, height) in original image coordinates.
        angle: rotation in degrees.
        flip_horizontal/vertical: booleans.
        
        Uses BORDER_REPLICATE to avoid black areas.
        """
        x, y, w, h = crop_rect
        # Crop the image using the provided rectangle.
        cropped = image[y:y+h, x:x+w]
        (h_img, w_img) = cropped.shape[:2]
        center = (w_img // 2, h_img // 2)
        # Obtain rotation matrix; note: positive angle rotates counter-clockwise.
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        # Use BORDER_REPLICATE to fill borders.
        rotated = cv2.warpAffine(cropped, M, (w_img, h_img), borderMode=cv2.BORDER_REPLICATE)
        if flip_horizontal:
            rotated = cv2.flip(rotated, 1)
        if flip_vertical:
            rotated = cv2.flip(rotated, 0)
        return rotated

    def generate_lut_from_points(points):
        """
        Given a list of control points (x, y) where both x and y are in the range 0-255,
        generates a 256-length lookup table using linear interpolation.
        """
        if len(points) < 2:
            # Not enough points, return identity LUT
            return np.arange(256, dtype=np.uint8)
        points = sorted(points, key=lambda p: p[0])
        xs, ys = zip(*points)
        xs = np.array(xs)
        ys = np.array(ys)
        # Create LUT for x = 0 to 255
        lut = np.interp(np.arange(256), xs, ys)
        return lut.astype(np.uint8)
