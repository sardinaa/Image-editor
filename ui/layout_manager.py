import dearpygui.dearpygui as dpg
from typing import Dict
from ui.components.tool_panel_modular import ModularToolPanel

class LayoutManager:
    """Manages window layout and resizing logic."""
    
    def __init__(self, main_window):
        self.main_window = main_window
        self.layout_config = {
            'tool_panel_ratio': 0.25,
            'menu_bar_height': 30,
            'status_bar_height': 35,
            'plot_margin': 5,
            'min_plot_size': 200,
            'border_padding': 10
        }
        self.theme = ThemeManager()
    
    def get_layout_dimensions(self) -> Dict[str, int]:
        """Calculate layout dimensions based on viewport size."""
        try:
            viewport_width = dpg.get_viewport_client_width()
            viewport_height = dpg.get_viewport_client_height()
        except (SystemError, RuntimeError):
            viewport_width = 1400
            viewport_height = 900
        
        available_height = viewport_height - self.layout_config['menu_bar_height'] - self.layout_config['status_bar_height'] - 10
        
        # Calculate available width for panels (account for border padding)
        border_padding = self.layout_config['border_padding']
        available_width = viewport_width - (border_padding * 2)  # Total padding for both panels
        
        # Calculate panel widths from available width to prevent horizontal scrolling
        tool_panel_width = int(available_width * self.layout_config['tool_panel_ratio'])
        central_panel_width = available_width - tool_panel_width
        
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
            label="Photo Editor - Production", 
            tag=self.main_window.window_tag,
            width=dims['viewport_width'], 
            height=dims['viewport_height'],
            no_scrollbar=True, 
            no_move=True, 
            no_resize=True, 
            no_collapse=True
        ):
            self.theme.setup_default_themes()
            self._create_menu_bar()
            
            with dpg.group(horizontal=True):
                # Central Panel
                self._create_central_panel(dims)
                # Tool Panel
                self._create_tool_panel(dims)
            
            # Add status bar at the bottom
            self._create_status_bar()
    
    def _create_status_bar(self):
        """Create the status bar at the bottom of the window."""
        with dpg.child_window(
            height=self.layout_config['status_bar_height'],
            width=-1,
            border=True,
            no_scrollbar=True,
            no_scroll_with_mouse=True
        ):
            with dpg.group(horizontal=True):
                dpg.add_spacer(width=10)
                dpg.add_text("Ready", tag="status_text")
                dpg.add_spacer(width=20)
                dpg.add_text("", tag="status_info")
                dpg.add_spacer(width=10)
    
    def _create_menu_bar(self):
        """Create the application menu bar."""
        with dpg.menu_bar():
            with dpg.menu(label="File"):
                dpg.add_menu_item(label="Open Image", callback=self.main_window._open_image)
                dpg.add_menu_item(label="Export Image", callback=self.main_window._export_image)
                dpg.add_separator()
                dpg.add_menu_item(label="Exit", callback=lambda: dpg.stop_dearpygui())
            
            with dpg.menu(label="Edit"):
                dpg.add_menu_item(label="Reset All", callback=self.main_window._reset_all_processing)
                dpg.add_menu_item(label="Clear Masks", callback=self.main_window._clear_all_masks)

            #with dpg.menu(label="View"):
                #dpg.add_checkbox(label="Dark Mode", tag= "dark_mode", default_value= True, callback=self.theme.apply_theme())
    
    def _create_central_panel(self, dims: Dict[str, int]):
        """Create the central image display panel."""
        with dpg.child_window(
            tag=self.main_window.central_panel_tag,
            width=dims['central_panel_width'],
            height=dims['available_height'],
            border=False
        ):
            self._create_image_plot(dims)
    
    def _create_image_plot(self, dims: Dict[str, int]):
        """Create the main image display plot."""
        # Calculate plot dimensions with reduced margin to save space
        margin = 0  # Reduced from 20 to save space
        plot_width = dims['central_panel_width'] - (2 * margin)
        plot_height = dims['available_height'] - (2 * margin)
        
        with dpg.child_window(
            parent=self.main_window.central_panel_tag,
            width=-1,
            height=-1,
            border=False,
            no_scrollbar=True
        ):
            with dpg.plot(
                label="Image Display",
                tag=self.main_window.image_plot_tag,
                height=plot_height,
                width=plot_width,
                anti_aliased=True,
                pan_button= dpg.mvMouseButton_Right,
                box_select_button=dpg.mvMouseButton_Middle,
                fit_button=dpg.mvMouseButton_Middle,
                no_mouse_pos=False,
                track_offset=True,
                equal_aspects=True  # Maintain aspect ratio to prevent squeezing
            ):
                dpg.add_plot_axis(dpg.mvXAxis, label="X", no_gridlines=True, tag=self.main_window.x_axis_tag)
                dpg.add_plot_axis(dpg.mvYAxis, label="Y", no_gridlines=True, tag=self.main_window.y_axis_tag)
    
    def _create_tool_panel(self, dims: Dict[str, int]):
        """Create the tool panel."""
        with dpg.child_window(
            tag=self.main_window.right_panel_tag,
            width=dims['tool_panel_width'],
            height=dims['available_height'],
            border=True
        ):
            
            # Create modular tool panel using service layer
            self.main_window.tool_panel = ModularToolPanel(
                update_callback=self.main_window._on_parameter_change,
                app_service=self.main_window.app_service,
                main_window=self.main_window
            )
            
            self.main_window.tool_panel.setup()
            
            # Draw the tool panel UI
            if hasattr(self.main_window.tool_panel, 'draw'):
                self.main_window.tool_panel.draw()
            else:
                # Fallback: create simple tool panel UI
                dpg.add_text("Tool Panel")
                dpg.add_separator()
                dpg.add_text("Image Editor Tools")
                dpg.add_button(label="Load Image", callback=self.main_window._open_image)
                dpg.add_button(label="Save Image", callback=self.main_window._save_image)
    
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
            self.main_window.crop_rotate_ui.panel_id = self.main_window.central_panel_tag
            if hasattr(self.main_window.crop_rotate_ui, 'update_axis_limits'):
                self.main_window.crop_rotate_ui.update_axis_limits(force=True)
                
            if hasattr(self.main_window.crop_rotate_ui, 'update_image'):
                self.main_window.crop_rotate_ui.update_image(None, None, None)
        
        # Update main window axis limits using DisplayService
        if hasattr(self.main_window, '_update_axis_limits'):
            self.main_window._update_axis_limits(initial=True)

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
    
    def apply_theme(self, theme_name: str = None):
        """Apply a theme to the application."""
        dark_mode = dpg.get_value("dark_mode")
        if dark_mode:
            dpg.bind_theme(self.themes["dark"])
        else:
            dpg.bind_theme(self.themes["light"])
        self.current_theme = theme_name
    
    def setup_default_themes(self):
        """Set up default themes."""
        self.create_dark_theme()
        self.create_light_theme()
