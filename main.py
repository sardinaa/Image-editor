import time
import dearpygui.dearpygui as dpg
from processing.image_processor import ImageProcessor
from processing.file_manager import load_image, save_image
from ui.main_window import MainWindow
from ui.crop_rotate import CropRotateUI
import numpy as np
import cv2

processor = None
main_window = None
crop_rotate_ui = None
last_update_time = 0
UPDATE_THRESHOLD = 0.1
updated_image = None
current_image_path = None

def update_image_callback():
    global last_update_time, updated_image, processor, crop_rotate_ui
    current_time = time.time()
    if current_time - last_update_time < UPDATE_THRESHOLD:
        return
    last_update_time = current_time

    if processor is None or crop_rotate_ui is None:
        return

    params = main_window.get_tool_parameters()
    processor.exposure = params.get('exposure', 0)
    processor.illumination = params.get('illumination', 0.0)
    processor.contrast = params.get('contrast', 1.0)
    processor.shadow = params.get('shadow', 0)
    processor.whites = params.get('whites', 0)
    processor.blacks = params.get('blacks', 0)
    processor.saturation = params.get('saturation', 1.0)
    processor.texture = params.get('texture', 0)
    processor.grain = params.get('grain', 0)
    processor.temperature = params.get('temperature', 0)
    
    updated_image = processor.apply_all_edits()
    curves = params.get('curves', None)
    if curves:
        updated_image = processor.apply_rgb_curves(updated_image, curves)
    
    crop_rotate_ui.original_image = updated_image
    crop_rotate_ui.update_image(None, None, None)

def load_image_callback():
    dpg.show_item("file_dialog_load")

def file_load_callback(sender, app_data, user_data):
    global processor, crop_rotate_ui, current_image_path
    file_path = app_data["file_path_name"]
    if not file_path:
        print("No se seleccionó ningún archivo.")
        return
    try:
        new_image = load_image(file_path)
    except Exception as e:
        print("Error al cargar la imagen:", e)
        return

    current_image_path = file_path
    processor = ImageProcessor(new_image.copy())
    crop_rotate_ui = CropRotateUI(new_image, processor)
    with dpg.texture_registry():
        gray_background = np.full((crop_rotate_ui.texture_h, crop_rotate_ui.texture_w, 4), 
                                 [100, 100, 100, 255], dtype=np.uint8)
        offset_x = (crop_rotate_ui.texture_w - crop_rotate_ui.orig_w) // 2
        offset_y = (crop_rotate_ui.texture_h - crop_rotate_ui.orig_h) // 2
        if new_image.shape[2] == 3:
            new_image_rgba = cv2.cvtColor(new_image, cv2.COLOR_RGB2RGBA)
        else:
            new_image_rgba = new_image
        gray_background[offset_y:offset_y + crop_rotate_ui.orig_h, 
                        offset_x:offset_x + crop_rotate_ui.orig_w] = new_image_rgba
        dpg.add_raw_texture(crop_rotate_ui.texture_w, crop_rotate_ui.texture_h,
                           gray_background.flatten() / 255.0,
                           format=dpg.mvFormat_Float_rgba,
                           tag=crop_rotate_ui.texture_tag)
    
    main_window.set_crop_rotate_ui(crop_rotate_ui)

def save_image_callback():
    dpg.show_item("file_dialog_save")

def file_save_callback(sender, app_data, user_data):
    global processor, crop_rotate_ui
    file_path = app_data["file_path_name"]
    if not processor or not crop_rotate_ui:
        print("No hay imagen para guardar.")
        return
    angle = dpg.get_value("rotation_slider")
    offset_x = (crop_rotate_ui.texture_w - crop_rotate_ui.orig_w) // 2
    offset_y = (crop_rotate_ui.texture_h - crop_rotate_ui.orig_h) // 2
    rx, ry, rw, rh = map(int, (crop_rotate_ui.user_rect["x"] - offset_x, 
                               crop_rotate_ui.user_rect["y"] - offset_y, 
                               crop_rotate_ui.user_rect["w"], crop_rotate_ui.user_rect["h"]))
    cropped = processor.crop_rotate_flip(processor.current, (rx, ry, rw, rh), angle)
    try:
        save_image(file_path, cropped)
    except Exception as e:
        print("Error al guardar la imagen:", e)

def main():
    global main_window
    dpg.create_context()
    dpg.create_viewport(title='Photo Editor', width=1200, height=800)
    dpg.setup_dearpygui()

    main_window = MainWindow(None, update_image_callback, load_image_callback, save_image_callback)
    main_window.setup()

    with dpg.file_dialog(directory_selector=False, show=False, callback=file_load_callback, tag="file_dialog_load"):
        dpg.add_file_extension(".png")
        dpg.add_file_extension(".jpg")
        dpg.add_file_extension(".jpeg")
        dpg.add_file_extension(".tif")
        dpg.add_file_extension(".bmp")
        dpg.add_file_extension(".ARW")
        dpg.add_file_extension(".RAW")

    with dpg.file_dialog(directory_selector=False, show=False, callback=file_save_callback, tag="file_dialog_save"):
        dpg.add_file_extension(".png")
        dpg.add_file_extension(".jpg")
        dpg.add_file_extension(".jpeg")
        dpg.add_file_extension(".tif")
        dpg.add_file_extension(".bmp")

    dpg.show_viewport()
    dpg.start_dearpygui()
    dpg.destroy_context()

if __name__ == "__main__":
    main()