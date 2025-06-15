"""
Layout management system for the main window.
Separates layout logic from core window functionality.
"""
import dearpygui.dearpygui as dpg
from typing import Dict, Tuple


class LayoutManager:
    """Manages window layout and resizing logic."""
    
    def __init__(self, main_window):
        self.main_window = main_window
        self.layout_config = {
            'tool_panel_ratio': 0.25,  # 25% of width for tools
            'menu_bar_height': 30,
            'plot_margin': 5,
            'min_plot_size': 200
        }
    
    def get_layout_dimensions(self) -> Dict[str, int]:
        """Calculate layout dimensions based on viewport size."""
        viewport_width = dpg.get_viewport_client_width()
        viewport_height = dpg.get_viewport_client_height()
        
        available_height = viewport_height - self.layout_config['menu_bar_height']
        tool_panel_width = int(viewport_width * self.layout_config['tool_panel_ratio'])
        central_panel_width = viewport_width - tool_panel_width
        
        return {
            'viewport_width': viewport_width,
            'viewport_height': viewport_height,
            'available_height': available_height,
            'tool_panel_width': tool_panel_width,
            'central_panel_width': central_panel_width
        }
    
    def setup_main_layout(self):
        """Set up the main window layout."""
        dims = self.get_layout_dimensions()
        
        with dpg.window(
            label="Photo Editor", 
            tag=self.main_window.window_tag,
            width=dims['viewport_width'], 
            height=dims['viewport_height'],
            no_scrollbar=True, 
            no_move=True, 
            no_resize=True, 
            no_collapse=True
        ):
            self._create_menu_bar()
            
            with dpg.group(horizontal=True):
                # Central Panel
                self._create_central_panel(dims)
                # Tool Panel
                self._create_tool_panel(dims)
    
    def _create_menu_bar(self):
        """Create the application menu bar."""
        with dpg.menu_bar():
            with dpg.menu(label="File"):
                dpg.add_menu_item(label="Load", callback=self.main_window.load_callback)
                dpg.add_menu_item(label="Save", callback=self.main_window.save_callback)
                dpg.add_menu_item(label="Exit", callback=lambda: dpg.stop_dearpygui())
            
            with dpg.menu(label="Edit"):
                dpg.add_menu_item(label="Undo", callback=lambda: print("Undo"))
                dpg.add_menu_item(label="Redo", callback=lambda: print("Redo"))
            
            with dpg.menu(label="View"):
                dpg.add_menu_item(label="Reset View", callback=self._reset_view)
                dpg.add_menu_item(label="Fit to Window", callback=self._fit_to_window)
    
    def _create_central_panel(self, dims: Dict[str, int]):
        """Create the central image display panel."""
        with dpg.child_window(
            tag=self.main_window.central_panel_tag,
            width=dims['central_panel_width'],
            height=dims['available_height'],
            border=False,
            no_scrollbar=True
        ):
            pass  # Content will be added dynamically
    
    def _create_tool_panel(self, dims: Dict[str, int]):
        """Create the tool panel."""
        with dpg.child_window(
            tag=self.main_window.right_panel_tag,
            width=dims['tool_panel_width'],
            height=dims['available_height'],
            no_scrollbar=True,
            horizontal_scrollbar=False
        ):
            if hasattr(self.main_window, 'tool_panel') and self.main_window.tool_panel:
                self.main_window.tool_panel.draw()
    
    def create_image_plot(self) -> Tuple[int, int]:
        """Create the central image plot and return its dimensions."""
        if not hasattr(self.main_window, 'crop_rotate_ui') or not self.main_window.crop_rotate_ui:
            return 0, 0
        
        dims = self.get_layout_dimensions()
        
        # Calculate plot dimensions
        margin = self.layout_config['plot_margin']
        plot_width = max(dims['central_panel_width'] - margin, self.layout_config['min_plot_size'])
        plot_height = max(dims['available_height'] - margin, self.layout_config['min_plot_size'])
        
        # Clear existing content
        if dpg.does_item_exist(self.main_window.central_panel_tag):
            dpg.delete_item(self.main_window.central_panel_tag, children_only=True)
        
        # Create plot container
        with dpg.child_window(
            parent=self.main_window.central_panel_tag,
            width=-1,
            height=-1,
            border=False,
            no_scrollbar=True
        ):
            with dpg.plot(
                label="Image Plot",
                no_mouse_pos=False,
                height=plot_height,
                width=plot_width,
                anti_aliased=True,
                pan_button=dpg.mvMouseButton_Left,
                fit_button=dpg.mvMouseButton_Middle,
                track_offset=True,
                tag="image_plot"
            ):
                dpg.add_plot_axis(dpg.mvXAxis, label="X", no_gridlines=True, tag="x_axis")
                dpg.add_plot_axis(dpg.mvYAxis, label="Y", no_gridlines=True, tag="y_axis")
                
                # Add image series
                y_axis = dpg.last_item()
                dpg.add_image_series(
                    self.main_window.crop_rotate_ui.texture_tag,
                    bounds_min=[0, 0],
                    bounds_max=[
                        self.main_window.crop_rotate_ui.texture_w,
                        self.main_window.crop_rotate_ui.texture_h
                    ],
                    parent=y_axis,
                    tag="central_image"
                )
        
        return plot_width, plot_height
    
    def handle_resize(self):
        """Handle window resize events."""
        dims = self.get_layout_dimensions()
        
        # Update panel sizes
        dpg.configure_item(
            self.main_window.central_panel_tag,
            width=dims['central_panel_width'],
            height=dims['available_height']
        )
        dpg.configure_item(
            self.main_window.right_panel_tag,
            width=dims['tool_panel_width'],
            height=dims['available_height']
        )
        
        # Update plot size
        if dpg.does_item_exist("image_plot"):
            margin = self.layout_config['plot_margin']
            plot_width = max(dims['central_panel_width'] - margin, self.layout_config['min_plot_size'])
            plot_height = max(dims['available_height'] - margin, self.layout_config['min_plot_size'])
            
            dpg.configure_item("image_plot", width=plot_width, height=plot_height)
            
            # Update drawlist size if it exists
            if hasattr(self.main_window, 'draw_list_tag') and dpg.does_item_exist(self.main_window.draw_list_tag):
                dpg.configure_item(self.main_window.draw_list_tag, width=plot_width, height=plot_height)
        
        # Update tool panel components
        if (hasattr(self.main_window, 'tool_panel') and 
            self.main_window.tool_panel and 
            hasattr(self.main_window.tool_panel, 'curves_panel') and
            self.main_window.tool_panel.curves_panel):
            self.main_window.tool_panel.curves_panel.resize_plot()
        
        # Update crop rotate UI
        if hasattr(self.main_window, 'crop_rotate_ui') and self.main_window.crop_rotate_ui:
            self.main_window.crop_rotate_ui.update_image(None, None, None)
            if hasattr(self.main_window, 'update_axis_limits'):
                self.main_window.update_axis_limits(initial=True)
    
    def _reset_view(self):
        """Reset the view to show the entire image."""
        if hasattr(self.main_window, 'update_axis_limits'):
            self.main_window.update_axis_limits(initial=True)
    
    def _fit_to_window(self):
        """Fit the image to the current window size."""
        self._reset_view()


class ThemeManager:
    """Manages application themes and styling."""
    
    def __init__(self):
        self.current_theme = "dark"
        self.themes = {}
    
    def create_dark_theme(self):
        """Create the dark theme."""
        with dpg.theme() as dark_theme:
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_color(dpg.mvThemeCol_WindowBg, [45, 45, 45, 255])
                dpg.add_theme_color(dpg.mvThemeCol_ChildBg, [35, 35, 35, 255])
                dpg.add_theme_color(dpg.mvThemeCol_Text, [255, 255, 255, 255])
                dpg.add_theme_color(dpg.mvThemeCol_Button, [60, 60, 60, 255])
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, [80, 80, 80, 255])
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, [100, 100, 100, 255])
        
        self.themes["dark"] = dark_theme
        return dark_theme
    
    def create_light_theme(self):
        """Create the light theme."""
        with dpg.theme() as light_theme:
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_color(dpg.mvThemeCol_WindowBg, [240, 240, 240, 255])
                dpg.add_theme_color(dpg.mvThemeCol_ChildBg, [250, 250, 250, 255])
                dpg.add_theme_color(dpg.mvThemeCol_Text, [0, 0, 0, 255])
                dpg.add_theme_color(dpg.mvThemeCol_Button, [220, 220, 220, 255])
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, [200, 200, 200, 255])
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, [180, 180, 180, 255])
        
        self.themes["light"] = light_theme
        return light_theme
    
    def apply_theme(self, theme_name: str):
        """Apply a theme to the application."""
        if theme_name in self.themes:
            dpg.bind_theme(self.themes[theme_name])
            self.current_theme = theme_name
    
    def setup_default_themes(self):
        """Set up default themes."""
        self.create_dark_theme()
        self.create_light_theme()
        self.apply_theme("dark")
