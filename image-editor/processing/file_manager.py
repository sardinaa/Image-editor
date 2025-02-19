# processing/file_manager.py
import cv2
import numpy as np
import os
import rawpy

def load_image(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext in ['.raw', '.arw']:
        with rawpy.imread(file_path) as raw:
            image = raw.postprocess()  # returns RGB image
    else:
        image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError("Could not load image: " + file_path)
        if len(image.shape) == 2:  # grayscale -> convert to RGB
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif len(image.shape) == 3:
            if image.shape[2] == 3:  # BGR image
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            elif image.shape[2] == 4:  # BGRA image
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
    return image

def save_image(file_path, image):
    ext = os.path.splitext(file_path)[1].lower()
    # Prepare image for saving: if it has 4 channels (RGBA), convert to BGRA; if 3 channels, convert RGB->BGR.
    if len(image.shape) == 3:
        if image.shape[2] == 3:
            image_to_save = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        elif image.shape[2] == 4:
            image_to_save = cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA)
        else:
            image_to_save = image
    else:
        image_to_save = image
    success = cv2.imwrite(file_path, image_to_save)
    if not success:
        raise IOError("Could not save image: " + file_path)
