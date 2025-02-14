# main.py
import time
import dearpygui.dearpygui as dpg
from processing.image_processor import ImageProcessor
from processing.file_manager import load_image, save_image
from ui.main_window import MainWindow

processor = None
main_window = None
last_update_time = 0
UPDATE_THRESHOLD = 0.1  # seconds

def update_image_callback():
    global last_update_time
    current_time = time.time()
    if current_time - last_update_time < UPDATE_THRESHOLD:
        return
    last_update_time = current_time

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
    main_window.update_preview(updated_image, reset_offset=False)

def load_image_callback():
    dpg.show_item("file_dialog_load")

def file_load_callback(sender, app_data, user_data):
    global processor
    file_path = app_data["file_path_name"]
    try:
        new_image = load_image(file_path)
    except Exception as e:
        print("Error loading image:", e)
        return
    processor.original = new_image.copy()
    processor.current = new_image.copy()
    main_window.update_preview(new_image, reset_offset=True)

def save_image_callback():
    dpg.show_item("file_dialog_save")

def file_save_callback(sender, app_data, user_data):
    global processor
    file_path = app_data["file_path_name"]
    try:
        save_image(file_path, processor.current)
    except Exception as e:
        print("Error saving image:", e)

def main():
    global processor, main_window

    default_image_path = "sample.png"
    try:
        original_image = load_image(default_image_path)
    except Exception as e:
        raise ValueError("Error loading default image: " + str(e))

    processor = ImageProcessor(original_image)

    dpg.create_context()
    dpg.create_viewport(title='Photo Editor', width=1200, height=800)
    dpg.setup_dearpygui()

    main_window = MainWindow(original_image,
                             update_callback=update_image_callback,
                             load_callback=load_image_callback,
                             save_callback=save_image_callback)
    main_window.setup()

    with dpg.file_dialog(directory_selector=False, show=False, callback=file_load_callback, tag="file_dialog_load"):
        dpg.add_file_extension(".*")
    with dpg.file_dialog(directory_selector=False, show=False, callback=file_save_callback, tag="file_dialog_save"):
        dpg.add_file_extension(".*")

    dpg.show_viewport()
    update_image_callback()
    dpg.start_dearpygui()
    dpg.destroy_context()

if __name__ == "__main__":
    main()
