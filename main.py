import time
import dearpygui.dearpygui as dpg
from processing.image_processor import ImageProcessor
from processing.file_manager import load_image, save_image
from ui.main_window import MainWindow

processor = None
main_window = None
last_update_time = 0
UPDATE_THRESHOLD = 0.1  # seconds
updated_image = None
current_image_path = None  # Store the currently loaded image path

def update_image_callback():
    global last_update_time, updated_image
    current_time = time.time()
    if current_time - last_update_time < UPDATE_THRESHOLD:
        return
    last_update_time = current_time

    if processor is None:
        return  # Prevent updates if no image is loaded

    # Get all parameters from the tools panel
    params = main_window.get_tool_parameters()
    
    # Update basic editing parameters
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
    
    # Apply the basic edits
    updated_image = processor.apply_all_edits()
    
    # If any curves are defined in the tools panel, apply them.
    curves = params.get('curves', None)
    if curves:
        updated_image = processor.apply_rgb_curves(updated_image, curves)
    
    # Apply crop/rotate/flip transformation.
    updated_image = processor.crop_rotate_flip(
        updated_image,
        params.get('crop_rect'),
        params.get('rotate_angle'),
        flip_horizontal=params.get('flip_horizontal'),
        flip_vertical=params.get('flip_vertical')
    )
    
    # Update the preview with the final processed image.
    main_window.update_preview(updated_image, reset_offset=False)

def load_image_callback():
    """ Opens the file dialog for the user to select an image. """
    dpg.show_item("file_dialog_load")

def file_load_callback(sender, app_data, user_data):
    """ Handles the actual image loading after the user selects a file. """
    global processor, current_image_path

    file_path = app_data["file_path_name"]  # The selected file path

    if not file_path:
        print("No file selected.")
        return

    try:
        new_image = load_image(file_path)
    except Exception as e:
        print("Error loading image:", e)
        return

    current_image_path = file_path  # Store the loaded image path
    processor = ImageProcessor(new_image.copy())  # Initialize the processor

    h, w = new_image.shape[:2]

    # Update crop sliders to match image dimensions.
    dpg.set_value("crop_x", 0)
    dpg.set_value("crop_y", 0)
    dpg.set_value("crop_w", w)
    dpg.set_value("crop_h", h)
    dpg.configure_item("crop_w", max_value=w)
    dpg.configure_item("crop_h", max_value=h)

    # Update the preview with the new image
    main_window.update_preview(new_image, reset_offset=True)

def save_image_callback():
    """ Opens the file dialog for saving the image. """
    dpg.show_item("file_dialog_save")

def file_save_callback(sender, app_data, user_data):
    """ Saves the currently processed image to a user-specified path. """
    global processor
    file_path = app_data["file_path_name"]

    if not processor or processor.current is None:
        print("No image to save.")
        return

    try:
        save_image(file_path, processor.current)
    except Exception as e:
        print("Error saving image:", e)

def main():
    global processor, main_window

    dpg.create_context()
    dpg.create_viewport(title='Photo Editor', width=1200, height=800)
    dpg.setup_dearpygui()

    # Initialize the UI without an image (user must load one)
    main_window = MainWindow(
        None,  # No default image
        update_callback=update_image_callback,
        load_callback=load_image_callback,
        save_callback=save_image_callback
    )
    main_window.setup()

    # File dialogs for loading/saving images
    with dpg.file_dialog(directory_selector=False, show=False, callback=file_load_callback, tag="file_dialog_load"):
        dpg.add_file_extension(".png")
        dpg.add_file_extension(".jpg")
        dpg.add_file_extension(".jpeg")
        dpg.add_file_extension(".tif")
        dpg.add_file_extension(".bmp")

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
