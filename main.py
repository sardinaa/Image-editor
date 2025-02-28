import time
import dearpygui.dearpygui as dpg
from processing.image_processor import ImageProcessor
from processing.file_manager import load_image, save_image
from ui.main_window import MainWindow

processor = None
main_window = None
last_update_time = 0
UPDATE_THRESHOLD = 0.1  # segundos
updated_image = None
current_image_path = None  # Ruta de la imagen cargada actualmente

def update_image_callback():
    global last_update_time, updated_image
    current_time = time.time()
    if current_time - last_update_time < UPDATE_THRESHOLD:
        return
    last_update_time = current_time

    if processor is None:
        return

    params = main_window.get_tool_parameters()
    
    # Actualizar parámetros de edición básicos
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
    
    angle = params.get('rotate_angle', 0)
    flip_h = params.get('flip_horizontal', False)
    flip_v = params.get('flip_vertical', False)

    if dpg.get_value("crop_mode"):
        # Se rota la imagen completa para la vista previa
        updated_image = processor.rotate_image(updated_image, angle,
                                            flip_horizontal=flip_h, flip_vertical=flip_v)
        # Obtener dimensiones máximas del rectángulo inscrito
        max_crop_w, max_crop_h = processor.get_largest_inscribed_rect_dims(angle)
        # Actualizar los sliders de tamaño
        dpg.configure_item("crop_w", max_value=max_crop_w)
        dpg.configure_item("crop_h", max_value=max_crop_h)
        dpg.configure_item("crop_x", max_value=updated_image.shape[0]//2)
        dpg.configure_item("crop_y", max_value=updated_image.shape[1]//2)
    else:
        # Cuando se aplica el recorte (crop_mode inactivo), se usa el grid para recortar la imagen final
        updated_image = processor.crop_rotate_flip(
            updated_image,
            params.get('crop_rect'),
            angle,
            flip_horizontal=flip_h,
            flip_vertical=flip_v
        )
        # Se puede ocultar el grid
        dpg.set_value("crop_mode", False)
        dpg.configure_item("crop_panel", show=False)
    
    main_window.update_preview(updated_image, reset_offset=False)


def load_image_callback():
    """Abre el diálogo para que el usuario seleccione una imagen."""
    dpg.show_item("file_dialog_load")

def file_load_callback(sender, app_data, user_data):
    """Carga la imagen seleccionada por el usuario."""
    global processor, current_image_path

    file_path = app_data["file_path_name"]  # Ruta del archivo seleccionado

    if not file_path:
        print("No se seleccionó ningún archivo.")
        return

    try:
        new_image = load_image(file_path)
    except Exception as e:
        print("Error al cargar la imagen:", e)
        return

    current_image_path = file_path  # Se guarda la ruta de la imagen cargada
    processor = ImageProcessor(new_image.copy())  # Se inicializa el procesador

    h, w = new_image.shape[:2]

    # Actualiza los sliders de recorte para que se ajusten a las dimensiones de la imagen
    dpg.set_value("crop_x", 0)
    dpg.set_value("crop_y", 0)
    dpg.set_value("crop_w", w)
    dpg.set_value("crop_h", h)
    dpg.configure_item("crop_w", max_value=w)
    dpg.configure_item("crop_h", max_value=h)

    # Actualiza la vista previa con la nueva imagen
    main_window.update_preview(new_image, reset_offset=True)

def save_image_callback():
    """Abre el diálogo para guardar la imagen."""
    dpg.show_item("file_dialog_save")

def file_save_callback(sender, app_data, user_data):
    """Guarda la imagen procesada en la ruta especificada por el usuario."""
    global processor
    file_path = app_data["file_path_name"]

    if not processor or processor.current is None:
        print("No hay imagen para guardar.")
        return

    try:
        save_image(file_path, processor.current)
    except Exception as e:
        print("Error al guardar la imagen:", e)

def main():
    global processor, main_window

    dpg.create_context()
    dpg.create_viewport(title='Photo Editor', width=1200, height=800)
    dpg.setup_dearpygui()

    # Inicializa la interfaz sin imagen (el usuario debe cargar una)
    main_window = MainWindow(
        None,  # Sin imagen por defecto
        update_callback=update_image_callback,
        load_callback=load_image_callback,
        save_callback=save_image_callback
    )
    main_window.setup()

    # Diálogos de archivo para cargar y guardar imágenes
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
        dpg.add_file_extension(".ARW")
        dpg.add_file_extension(".RAW")

    dpg.show_viewport()
    dpg.start_dearpygui()
    dpg.destroy_context()

if __name__ == "__main__":
    main()
