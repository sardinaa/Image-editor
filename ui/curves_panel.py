import dearpygui.dearpygui as dpg
import numpy as np

class CurvesPanel:
    def __init__(self, callback):
        self.callback = callback  # Callback to update the image after curves changes
        self.current_channel = "RGB"  # Default to RGB (all channels) 
        self.channels = ["RGB", "R", "G", "B"]
        self.curves = {
            "r": [(0, 0), (128, 128), (255, 255)],  # Default linear curve for R
            "g": [(0, 0), (128, 128), (255, 255)],  # Default linear curve for G
            "b": [(0, 0), (128, 128), (255, 255)]   # Default linear curve for B
        }
        self.plot_tag = "curves_plot"
        self.points_tag = "curves_points_series"  # Changed to be more descriptive and unique
        self.x_axis_tag = "curves_x_axis"
        self.y_axis_tag = "curves_y_axis"
        self.line_tag = "curves_line_series"  # Changed to be more descriptive and unique
        self.dragging_point = None
        self.point_size = 5
        
        # Create theme tags
        self.theme_rgb = "curves_theme_rgb"
        self.theme_r = "curves_theme_r"
        self.theme_g = "curves_theme_g"
        self.theme_b = "curves_theme_b"
        
        # Create themes for each channel
        self._create_channel_themes()
        
    def show(self):
        """Show the curves editor directly in the tool panel"""
        self.create_panel()
    
    def hide(self):
        """Hide functionality is no longer needed as the panel is always visible"""
        pass
    
    def on_window_close(self):
        """No longer needed as the panel is always visible"""
        pass
    
    def create_panel(self):
        """Create the curves editor panel directly in the parent"""
        # Create a group for the curves editor
        with dpg.group():
            # Fix radio buttons: Use a single radio button group
            with dpg.group(horizontal=True):
                dpg.add_text("Channel: ")
                # Create a single radio button widget with all options
                dpg.add_radio_button(
                    items=self.channels,
                    tag="curves_channel_selector",
                    callback=self.change_channel,
                    default_value=self.current_channel,
                    horizontal=True
                )
            
            # Add plot for curves
            with dpg.plot(tag=self.plot_tag, height=200, width=-1):
                # Add plot legend
                dpg.add_plot_legend()
                
                # Add plot axes with unique tags
                dpg.add_plot_axis(dpg.mvXAxis, label="Input", tag=self.x_axis_tag)
                dpg.add_plot_axis(dpg.mvYAxis, label="Output", tag=self.y_axis_tag)
                
                # Set axes limits
                dpg.set_axis_limits(self.x_axis_tag, 0, 255)
                dpg.set_axis_limits(self.y_axis_tag, 0, 255)
                
                # Add reference line using a simple series instead of a draw layer
                dpg.add_line_series([0, 255], [0, 255], label="Reference", parent=self.y_axis_tag)
            
            # Add handler for mouse interactions (with unique tag)
            with dpg.handler_registry(tag="curves_mouse_handler"):
                dpg.add_mouse_click_handler(callback=self.on_click)
                dpg.add_mouse_drag_handler(callback=self.on_drag)
                dpg.add_mouse_release_handler(callback=self.on_release)
                # Add hover handler to highlight points
                dpg.add_mouse_move_handler(callback=self.on_mouse_move)
            
            # Add only the reset button, removing apply and close
            with dpg.group(horizontal=True):
                dpg.add_button(label="Reset Curve", callback=self.reset_curve, width=-1)
            
            # Update the plot immediately
            self.update_plot()
    
    def _create_channel_themes(self):
        """Create themes for each channel to ensure consistent colors"""
        # RGB theme (white)
        with dpg.theme(tag=self.theme_rgb):
            with dpg.theme_component(dpg.mvLineSeries):
                dpg.add_theme_color(dpg.mvPlotCol_Line, [255, 255, 255], category=dpg.mvThemeCat_Plots)
            with dpg.theme_component(dpg.mvScatterSeries):
                dpg.add_theme_color(dpg.mvPlotCol_MarkerFill, [255, 255, 255], category=dpg.mvThemeCat_Plots)
                dpg.add_theme_color(dpg.mvPlotCol_MarkerOutline, [255, 255, 255], category=dpg.mvThemeCat_Plots)
        
        # Red theme
        with dpg.theme(tag=self.theme_r):
            with dpg.theme_component(dpg.mvLineSeries):
                dpg.add_theme_color(dpg.mvPlotCol_Line, [255, 0, 0], category=dpg.mvThemeCat_Plots)
            with dpg.theme_component(dpg.mvScatterSeries):
                dpg.add_theme_color(dpg.mvPlotCol_MarkerFill, [255, 0, 0], category=dpg.mvThemeCat_Plots)
                dpg.add_theme_color(dpg.mvPlotCol_MarkerOutline, [255, 0, 0], category=dpg.mvThemeCat_Plots)
        
        # Green theme
        with dpg.theme(tag=self.theme_g):
            with dpg.theme_component(dpg.mvLineSeries):
                dpg.add_theme_color(dpg.mvPlotCol_Line, [0, 255, 0], category=dpg.mvThemeCat_Plots)
            with dpg.theme_component(dpg.mvScatterSeries):
                dpg.add_theme_color(dpg.mvPlotCol_MarkerFill, [0, 255, 0], category=dpg.mvThemeCat_Plots)
                dpg.add_theme_color(dpg.mvPlotCol_MarkerOutline, [0, 255, 0], category=dpg.mvThemeCat_Plots)
        
        # Blue theme
        with dpg.theme(tag=self.theme_b):
            with dpg.theme_component(dpg.mvLineSeries):
                dpg.add_theme_color(dpg.mvPlotCol_Line, [0, 0, 255], category=dpg.mvThemeCat_Plots)
            with dpg.theme_component(dpg.mvScatterSeries):
                dpg.add_theme_color(dpg.mvPlotCol_MarkerFill, [0, 0, 255], category=dpg.mvThemeCat_Plots)
                dpg.add_theme_color(dpg.mvPlotCol_MarkerOutline, [0, 0, 255], category=dpg.mvThemeCat_Plots)

    def update_plot(self):
        """Update the plot with current curve data"""
        # Check if the plot exists before updating
        if not dpg.does_item_exist(self.plot_tag) or not dpg.does_item_exist(self.y_axis_tag):
            return
            
        try:
            # Get the current channel to draw (lowercase for dictionary key)
            channel_key = self.current_channel.lower()
            if channel_key == "rgb":
                # If RGB (all channels), use R for display
                channel_key = "r"
            
            # If there are fewer than 2 points, reset to default
            if len(self.curves[channel_key]) < 2:
                self.curves[channel_key] = [(0, 0), (255, 255)]
            
            # Sort points by x coordinate
            self.curves[channel_key] = sorted(self.curves[channel_key], key=lambda p: p[0])
            
            # Update or create curve line
            x_values = [float(point[0]) for point in self.curves[channel_key]]
            y_values = [float(point[1]) for point in self.curves[channel_key]]
            
            # Debug output to verify data
            print(f"X values: {x_values}")
            print(f"Y values: {y_values}")
            
            # Get theme based on channel
            theme_tag = self.theme_rgb  # Default for RGB
            if self.current_channel == "R":
                theme_tag = self.theme_r
            elif self.current_channel == "G":
                theme_tag = self.theme_g
            elif self.current_channel == "B":
                theme_tag = self.theme_b
            
            # Delete previous series if they exist
            if dpg.does_item_exist(self.line_tag):
                dpg.delete_item(self.line_tag)
            
            if dpg.does_item_exist(self.points_tag):
                dpg.delete_item(self.points_tag)
            
            # Ensure the axis exists before adding series
            if not dpg.does_item_exist(self.y_axis_tag):
                print(f"Y-axis with tag {self.y_axis_tag} does not exist!")
                return
                
            # Ensure there are no duplicate x values (could cause issues)
            seen_x = set()
            unique_points = []
            for i, x in enumerate(x_values):
                if x not in seen_x:
                    seen_x.add(x)
                    unique_points.append((x, y_values[i]))
            
            if len(unique_points) < len(x_values):
                print(f"Warning: Removed {len(x_values) - len(unique_points)} duplicate x values")
                x_values = [p[0] for p in unique_points]
                y_values = [p[1] for p in unique_points]
            
            # Ensure there are at least 2 points for a valid line
            if len(x_values) < 2:
                print("Warning: Not enough points for a valid curve, adding default end points")
                x_values = [0.0, 255.0]
                y_values = [0.0, 255.0]
            
            # Add the new line series
            dpg.add_line_series(
                x=x_values, 
                y=y_values, 
                label=f"{self.current_channel} Curve",
                tag=self.line_tag,
                parent=self.y_axis_tag
            )
            # Bind theme to line series
            dpg.bind_item_theme(self.line_tag, theme_tag)
            
            # Add new scatter points
            dpg.add_scatter_series(
                x=x_values,
                y=y_values,
                tag=self.points_tag,
                label="Control Points",
                parent=self.y_axis_tag
            )
            # Bind theme to scatter points
            dpg.bind_item_theme(self.points_tag, theme_tag)
            
        except Exception as e:
            print(f"Error updating curve plot: {e}")
            import traceback
            traceback.print_exc()
    
    def change_channel(self, sender, app_data, user_data):
        """Change the current channel"""
        self.current_channel = app_data  # Now app_data contains the selected value
        self.update_plot()
    
    def on_click(self, sender, app_data):
        """Handle mouse click to add or select a point"""
        if not self.is_mouse_over_plot():
            return
        
        # Get mouse position in plot coordinates
        x, y = self.get_mouse_plot_coordinates()
        
        # Clamp values
        x = max(0, min(255, x))
        y = max(0, min(255, y))
        
        # Get the current channel key
        channel_key = self.current_channel.lower()
        
        # Fix: Use "r" for point checking when "rgb" is selected
        check_key = "r" if channel_key == "rgb" else channel_key
        
        # If RGB selected, modify all channels
        channels_to_modify = ["r", "g", "b"] if channel_key == "rgb" else [channel_key]
        
        # Increase hit-test radius - make it easier to select points
        hit_radius = 25  # Increased from 10
        
        # Check if clicked near an existing point - use check_key instead of channel_key
        for idx, point in enumerate(self.curves[check_key]):
            if abs(point[0] - x) < hit_radius and abs(point[1] - y) < hit_radius:
                # Found a point near click
                self.dragging_point = (idx, channels_to_modify)
                return
        
        # If no point found, add a new one to each affected channel
        for ch in channels_to_modify:
            self.curves[ch].append((x, y))
        
        # Update the plot
        self.update_plot()
        
        # Notify of changes - Fix: Pass all required parameters to callback
        self.callback(sender, app_data, None)
    
    def on_drag(self, sender, app_data):
        """Handle dragging a point"""
        if self.dragging_point is None:
            return
        
        # Get mouse position in plot coordinates using the helper method
        x, y = self.get_mouse_plot_coordinates()
        
        # Clamp values
        x = max(0, min(255, x))
        y = max(0, min(255, y))
        
        # Update point in all affected channels
        idx, channels = self.dragging_point
        for ch in channels:
            if 0 <= idx < len(self.curves[ch]):
                # Special cases for endpoints
                if idx == 0:  # First point
                    x = 0  # Keep at 0
                elif idx == len(self.curves[ch]) - 1:  # Last point
                    x = 255  # Keep at 255
                self.curves[ch][idx] = (x, y)
        
        # Update the plot
        self.update_plot()
        
        # Notify of changes - Fix: Pass all required parameters to callback
        self.callback(sender, app_data, None)
    
    def on_release(self, sender, app_data):
        """Handle mouse release"""
        self.dragging_point = None
    
    def on_mouse_move(self, sender, app_data):
        """Handle mouse movement for hover effects"""
        if not self.is_mouse_over_plot():
            return
            
        # Could implement hover highlighting here
        # For performance reasons, we'll just update the cursor
        x, y = self.get_mouse_plot_coordinates()
        
        # Get the current channel key
        channel_key = self.current_channel.lower()
        check_key = "r" if channel_key == "rgb" else channel_key
        
        # Check if mouse is near any point
        hover_radius = 15
        is_near_point = False
        
        for point in self.curves[check_key]:
            if abs(point[0] - x) < hover_radius and abs(point[1] - y) < hover_radius:
                is_near_point = True
                break
                
        # Could change cursor here if over a point
        # But that's outside the scope of this fix
    
    def get_mouse_plot_coordinates(self):
        """Convert screen coordinates to plot coordinates"""
        # Get mouse position and plot position/size
        mouse_pos = dpg.get_mouse_pos()
        plot_pos = dpg.get_item_rect_min(self.plot_tag)
        plot_size = dpg.get_item_rect_size(self.plot_tag)
        
        # Get the actual plotting area (accounting for axis labels, etc.)
        # This is approximate and may need adjustment
        plot_padding_x = 40  # Estimated padding for y-axis labels
        plot_padding_y = 30  # Estimated padding for x-axis labels
        
        # Calculate effective plot area
        effective_width = plot_size[0] - plot_padding_x
        effective_height = plot_size[1] - plot_padding_y
        effective_x = plot_pos[0] + plot_padding_x
        effective_y = plot_pos[1]
        
        # Convert mouse position to normalized coordinates [0,1]
        # Use effective plot area for more accurate coordinates
        if effective_width <= 0 or effective_height <= 0:
            return 0, 0
            
        x_norm = (mouse_pos[0] - effective_x) / effective_width
        y_norm = 1.0 - (mouse_pos[1] - effective_y) / effective_height
        
        # Convert to curve coordinates [0,255]
        x = int(x_norm * 255)
        y = int(y_norm * 255)
        
        return x, y
    
    def is_mouse_over_plot(self):
        """Check if mouse is over the plot area"""
        # Get mouse position
        mouse_pos = dpg.get_mouse_pos()
        
        # Get plot position and size
        if not dpg.does_item_exist(self.plot_tag):
            return False
            
        plot_rect_min = dpg.get_item_rect_min(self.plot_tag)
        plot_rect_max = dpg.get_item_rect_max(self.plot_tag)
        
        # Check if mouse is within plot bounds
        return (plot_rect_min[0] <= mouse_pos[0] <= plot_rect_max[0] and 
                plot_rect_min[1] <= mouse_pos[1] <= plot_rect_max[1])
    
    def reset_curve(self, sender=None, app_data=None, user_data=None):
        """Reset the curve to default (linear)"""
        channel_key = self.current_channel.lower()
        if channel_key == "rgb":
            # Reset all channels
            self.curves = {
                "r": [(0, 0), (128, 128), (255, 255)],
                "g": [(0, 0), (128, 128), (255, 255)],
                "b": [(0, 0), (128, 128), (255, 255)]
            }
        else:
            # Reset only the selected channel
            self.curves[channel_key] = [(0, 0), (128, 128), (255, 255)]
        
        self.update_plot()
        # Fix: Pass all required parameters to callback
        self.callback(sender, app_data, user_data)
    
    def apply_curve(self, sender=None, app_data=None, user_data=None):
        """Apply the curve to the image and close the window"""
        # Fix: Pass all required parameters to callback
        self.callback(sender, app_data, user_data)
        self.hide()
    
    def get_curves(self):
        """Return the curves data for image processing"""
        return self.curves
