import dearpygui.dearpygui as dpg
import dearpygui_extend as dpge
from prova import CropAndRotate  # Importamos la clase refactorizada

# Layout base (usa "tab" para la indentación)
layout = '''
LAYOUT example center center
	COL left_menu 0.2
	COL body_content 0.6
	COL right_menu 0.2
'''

def print_me(sender):
    print(f"Menu Item: {sender}")

dpg.create_context()
dpg.create_viewport(title='Custom Title')

with dpg.window(tag="Primary Window", no_close=True, no_title_bar=False, no_move=True):
    dpge.add_layout(layout, border=Fa)
    with dpg.menu_bar():
        with dpg.menu(label="File"):
            dpg.add_menu_item(label="Load", callback=print_me)
            dpg.add_menu_item(label="Save", callback=print_me)
            dpg.add_menu_item(label="Save As", callback=print_me)

            with dpg.menu(label="Settings"):
                dpg.add_menu_item(label="Setting 1", callback=print_me, check=True)
                dpg.add_menu_item(label="Setting 2", callback=print_me)

        dpg.add_menu_item(label="Help", callback=print_me)

        with dpg.menu(label="Widget Items"):
            dpg.add_checkbox(label="Pick Me", callback=print_me)
            dpg.add_button(label="Press Me", callback=print_me)
            dpg.add_color_picker(label="Color Me", callback=print_me)

# Acceder a los paneles de layout
with dpg.group(parent='left_menu'):
    dpg.add_text('User login:')
    dpg.add_input_text(label='username')
    dpg.add_input_text(label='password')
    dpg.add_button(label='Login')
     
with dpg.group(parent='right_menu', height=150):
    # Aquí se añadirán los sliders y botones para transformar la imagen
    pass

with dpg.group(parent='body_content'):
    # Aquí se visualizará la imagen resultante
    pass

# Inicializar la herramienta CropAndRotate e integrarla en los grupos correspondientes
crop_and_rotate = CropAndRotate("sample.png")
crop_and_rotate.setup_ui(parent_image='body_content', parent_controls='right_menu')

dpg.setup_dearpygui()
dpg.show_viewport()
dpg.set_primary_window("Primary Window", True)
dpg.start_dearpygui()
dpg.destroy_context()
