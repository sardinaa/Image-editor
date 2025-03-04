import cv2
import numpy as np
import dearpygui.dearpygui as dpg
from prova import CropAndRotate

def create_menu():
    with dpg.menu_bar():
        with dpg.menu(label="Archivo"):
            dpg.add_menu_item(label="Abrir", callback=lambda: print("Abrir"))
            dpg.add_menu_item(label="Guardar", callback=lambda: print("Guardar"))
            dpg.add_menu_item(label="Salir", callback=lambda: dpg.stop_dearpygui())
        with dpg.menu(label="Editar"):
            dpg.add_menu_item(label="Deshacer", callback=lambda: print("Deshacer"))
            dpg.add_menu_item(label="Rehacer", callback=lambda: print("Rehacer"))

def create_central_panel(crop_and_rotate):
    with dpg.child_window(tag="Central Panel", width=-1, height=-1):
        dpg.add_text("Contenido principal")
        with dpg.plot(label="Image Plot", no_mouse_pos=False, height=-1, width=-1):
            x_axis = dpg.add_plot_axis(dpg.mvXAxis, label="X", no_gridlines=True, tag="x_axis")
            y_axis = dpg.add_plot_axis(dpg.mvYAxis, label="Y", no_gridlines=True, tag="y_axis")
            # Calcular los bounds para la image_series
            dpg.add_image_series(crop_and_rotate.texture_tag,
                                 bounds_min=[0, 0],
                                 bounds_max=[crop_and_rotate.texture_w, crop_and_rotate.texture_h],
                                 parent=y_axis,
                                 tag="central_image")

def create_right_panel(crop_and_rotate):
    main_width = dpg.get_viewport_client_width()
    with dpg.child_window(tag="Right Panel", width=main_width // 4, autosize_y=True):
        dpg.add_text("Propiedades")
        dpg.add_slider_float(label="Ángulo de Rotación", tag=crop_and_rotate.rotation_slider,
                            min_value=0, max_value=360, default_value=0, callback=crop_and_rotate.update_image)
        dpg.add_color_picker(label="Color Tint", tag="color_picker", default_value=[1.0, 1.0, 1.0, 1.0],
                            no_alpha=True, callback=crop_and_rotate.update_image)
        dpg.add_button(label="Máxima Área", callback=crop_and_rotate.set_to_max_rect)
        dpg.add_button(label="Crop", callback=crop_and_rotate.crop_image)

def register_mouse_handlers(crop_and_rotate):
    with dpg.handler_registry():
        dpg.add_mouse_down_handler(callback=crop_and_rotate.on_mouse_down)
        dpg.add_mouse_drag_handler(callback=crop_and_rotate.on_mouse_drag)
        dpg.add_mouse_release_handler(callback=crop_and_rotate.on_mouse_release)

def main():
    dpg.create_context()
    dpg.create_viewport(title="Raviewer GUI", width=1920, height=1080)
    
    crop_and_rotate = CropAndRotate("no.png")
    crop_and_rotate.texture_tag = "crop_rotate_texture"
    
    # Calculate texture size based on the image diagonal (square texture)
    diagonal = int(np.ceil(np.sqrt(crop_and_rotate.orig_w**2 + crop_and_rotate.orig_h**2)))
    texture_w = diagonal
    texture_h = diagonal
    crop_and_rotate.texture_w = texture_w
    crop_and_rotate.texture_h = texture_h
    
    # Initialize with gray background
    gray_background = np.full((texture_h, texture_w, 4), [100, 100, 100, 255], dtype=np.uint8)
    offset_x = (texture_w - crop_and_rotate.orig_w) // 2
    offset_y = (texture_h - crop_and_rotate.orig_h) // 2
    gray_background[offset_y:offset_y + crop_and_rotate.orig_h, offset_x:offset_x + crop_and_rotate.orig_w] = crop_and_rotate.original_image
    
    with dpg.texture_registry():
        dpg.add_raw_texture(texture_w, texture_h,
                           gray_background.flatten() / 255.0,
                           format=dpg.mvFormat_Float_rgba,
                           tag=crop_and_rotate.texture_tag)
    
    with dpg.window(label="Raviewer GUI", tag="Main Window", no_collapse=True, no_title_bar=True):
        create_menu()
        with dpg.table(header_row=False, resizable=False, policy=dpg.mvTable_SizingStretchProp):
            dpg.add_table_column()
            dpg.add_table_column(width_fixed=True)
            with dpg.table_row():
                create_central_panel(crop_and_rotate)
                create_right_panel(crop_and_rotate)
    
    register_mouse_handlers(crop_and_rotate)
    
    dpg.set_primary_window("Main Window", True)
    dpg.setup_dearpygui()
    def on_resize():
        crop_and_rotate.update_image(None, None, None)

    dpg.set_viewport_resize_callback(on_resize)
    dpg.show_viewport()
    
    crop_and_rotate.update_image(None, None, None)
    
    dpg.start_dearpygui()
    dpg.destroy_context()

if __name__ == "__main__":
    main()