import cv2
import numpy as np
import dearpygui.dearpygui as dpg
from prova import CropAndRotate  # Asegúrate de que 'prova.py' esté en el path adecuado

def update_panel_background(sender, app_data, user_data):
        color = [int(c * 255) for c in app_data[:3]]
        # Prova de configurar directament la propietat background_color (si és compatible)
        try:
            dpg.configure_item("Central Panel", background_color=color)
        except Exception as e:
            print("No s'ha pogut configurar background_color directament:", e)
        # Crear un theme per forçar el color de fons al child_window
        with dpg.theme() as bg_theme:
            with dpg.theme_component(dpg.mvChildWindow):
                dpg.add_theme_color(dpg.mvThemeCol_ChildBg, color, category=dpg.mvThemeCat_Core)
        dpg.bind_item_theme("Central Panel", bg_theme)

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
    # El child_window se configura para ocupar el espacio disponible
    with dpg.child_window(tag="Central Panel", width=-1, height=-1):
        dpg.add_text("Contenido principal")
        # Agregar el widget de imagen; la textura ya se creó previamente
        dpg.add_image(crop_and_rotate.texture_tag, tag="central_image")

def create_right_panel(crop_and_rotate):
    main_width = dpg.get_viewport_client_width()
    with dpg.child_window(tag="Right Panel", width=main_width // 4, autosize_y=True):
        dpg.add_text("Propiedades")
        dpg.add_slider_float(label="Ángulo de Rotación", tag=crop_and_rotate.rotation_slider,
                             min_value=0, max_value=360, default_value=0, callback=crop_and_rotate.update_image)
        dpg.add_slider_float(label="Zoom", tag=crop_and_rotate.zoom_slider,
                             default_value=crop_and_rotate.default_scale, min_value=0.1, max_value=2.0, callback=crop_and_rotate.update_image)
        dpg.add_color_picker(label="Color Tint", tag="color_picker", default_value=[1.0, 1.0, 1.0, 1.0],
                             no_alpha=True, callback=crop_and_rotate.update_image)
        dpg.add_color_picker(label="Fons del Panel", tag="panel_bg_picker", default_value=[1.0, 1.0, 1.0, 1.0],
                             no_alpha=False, callback=update_panel_background)
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
    
    # Instanciar la clase con la imagen deseada
    crop_and_rotate = CropAndRotate("no.png")
    # Definir un tag fijo para la textura
    crop_and_rotate.texture_tag = "crop_rotate_texture"
    
    # Crear una textura placeholder con las dimensiones originales para evitar el error en add_image
    with dpg.texture_registry():
        dpg.add_raw_texture(crop_and_rotate.orig_w, crop_and_rotate.orig_h,
                                crop_and_rotate.original_image.flatten() / 255.0,
                                format=dpg.mvFormat_Float_rgba,
                                tag=crop_and_rotate.texture_tag)
    
    with dpg.window(label="Raviewer GUI", tag="Main Window", no_collapse=True, no_title_bar=True):
        create_menu()
        with dpg.table(header_row=False, resizable=False, policy=dpg.mvTable_SizingStretchProp):
            dpg.add_table_column()                   # Columna Central: se estira automáticamente.
            dpg.add_table_column(width_fixed=True)   # Columna Right: ancho fijo.
            with dpg.table_row():
                create_central_panel(crop_and_rotate)
                create_right_panel(crop_and_rotate)
    
    register_mouse_handlers(crop_and_rotate)
    
    dpg.set_primary_window("Main Window", True)
    dpg.setup_dearpygui()
    dpg.show_viewport()
    
    # Actualizar la imagen para que se procese con las dimensiones actuales del Central Panel
    crop_and_rotate.update_image(None, None, None)
    
    dpg.start_dearpygui()
    dpg.destroy_context()

if __name__ == "__main__":
    main()
