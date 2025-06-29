import dearpygui.dearpygui as dpg
import numpy as np

class CurvesPanel:
    def __init__(self, callback):
        self.callback = callback  # Callback to update the image after curves changes
        self.current_channel = "RGB"  # Default to RGB (all channels) 
        self.channels = ["RGB", "R", "G", "B", "L"]  # Added Luminance
        self.curves = {
            "r": [(0, 0), (128, 128), (255, 255)],  # Default linear curve for R
            "g": [(0, 0), (128, 128), (255, 255)],  # Default linear curve for G
            "b": [(0, 0), (128, 128), (255, 255)],  # Default linear curve for B
            "l": [(0, 0), (128, 128), (255, 255)]   # Default linear curve for Luminance
        }
        self.plot_tag = "curves_plot"
        self.points_tag = "curves_points_series"  # Changed to be more descriptive and unique
        self.x_axis_tag = "curves_x_axis"
        self.y_axis_tag = "curves_y_axis"
        self.line_tag = "curves_line_series"  # Changed to be more descriptive and unique
        self.dragging_point = None
        self.selected_point = None  # Track selected point for deletion
        self.point_size = 5
        self.pending_single_click = None  # Store pending single click data
        self.single_click_delay = 0.3  # Delay single clicks to detect double clicks
        self.double_click_threshold = 0.5  # 500ms - ignore single clicks within this time after double-click
        
        # Interpolation mode: "Linear" or "Spline"
        self.interpolation_modes = ["Linear", "Spline"]
        self.current_interpolation = "Spline"
        self.interpolation_combo_tag = "curves_interpolation_combo"
        
        # Plot sizing - reduced for better space utilization  
        self.plot_size = 120  # Reduced from 150 to save space
        self.min_plot_size = 80   # Reduced minimum
        self.max_plot_size = 200  # Reduced maximum
        
        # Create theme tags
        self.theme_rgb = "curves_theme_rgb"
        self.theme_r = "curves_theme_r"
        self.theme_g = "curves_theme_g"
        self.theme_b = "curves_theme_b"
        self.theme_l = "curves_theme_l"  # Theme for luminance
        self.theme_blue_selected = "curves_theme_blue_selected"
        
        # Create themes for each channel
        self._create_channel_themes()
        
    def _calculate_plot_size(self):
        """Calculate optimal plot size based on available space in tool panel"""
        try:
            # Get the viewport dimensions
            viewport_width = dpg.get_viewport_client_width()
            
            # Calculate available width for the tool panel (25% of viewport)
            tool_panel_width = int(viewport_width * 0.25)
            
            # Account for padding and margins in the tool panel
            margin = 40  # Margins for plot borders and scrollbars
            available_width = tool_panel_width - margin
            
            # Calculate square plot size - use available width but respect min/max constraints
            plot_size = min(max(available_width, self.min_plot_size), self.max_plot_size)
            
            # Ensure it's a reasonable size for the curves editor
            self.plot_size = max(plot_size, self.min_plot_size)
            
        except Exception as e:
            # Fallback to default size if any error occurs
            self.plot_size = 150
    
    def resize_plot(self):
        """Resize the plot when window dimensions change"""
        if dpg.does_item_exist(self.plot_tag):
            self._calculate_plot_size()
            dpg.configure_item(self.plot_tag, width=self.plot_size, height=self.plot_size)
    
    def show(self):
        """Show the curves editor directly in the tool panel"""
        self.create_panel()
    
    def create_panel(self):
        """Create the curves editor panel directly in the parent"""
        # Calculate the optimal plot size for current window dimensions
        self._calculate_plot_size()
        
        # Create a group for the curves editor
        with dpg.group():
            # Fix radio buttons: Use a single radio button group - more compact
            with dpg.group(horizontal=True):
                dpg.add_text("Ch:")  # Shorter label
                # Create a single radio button widget with all options
                dpg.add_radio_button(
                    items=self.channels,
                    tag="curves_channel_selector",
                    callback=self.change_channel,
                    default_value=self.current_channel,
                    horizontal=True
                )
            
            # Add the plot for curves - square aspect ratio (1:1) that resizes
            with dpg.plot(tag=self.plot_tag, height=self.plot_size, width=self.plot_size,
                         callback=self.on_plot_callback):
                
                # Add plot axes with unique tags
                dpg.add_plot_axis(dpg.mvXAxis, label="Input", tag=self.x_axis_tag)
                dpg.add_plot_axis(dpg.mvYAxis, label="Output", tag=self.y_axis_tag)
                
                # Set axes limits
                dpg.set_axis_limits(self.x_axis_tag, 0, 255)
                dpg.set_axis_limits(self.y_axis_tag, 0, 255)
                
                # Add reference line using a simple series instead of a draw layer
                dpg.add_line_series([0, 255], [0, 255], parent=self.y_axis_tag)
            
            # Add plot-specific mouse handlers using item handlers
            with dpg.item_handler_registry() as plot_handlers:
                dpg.add_item_clicked_handler(callback=self.on_plot_clicked_direct)
                dpg.add_item_hover_handler(callback=self.on_plot_hover_direct)
            
            # Bind the handlers to the plot
            dpg.bind_item_handler_registry(self.plot_tag, plot_handlers)
            
            # Add global mouse handlers for better drag support
            with dpg.handler_registry():
                dpg.add_mouse_drag_handler(callback=self.on_global_drag, button=dpg.mvMouseButton_Left)
                dpg.add_mouse_release_handler(callback=self.on_global_release, button=dpg.mvMouseButton_Left)
            
            # Add controls - more compact
            with dpg.group(horizontal=True):
                dpg.add_button(label="Reset", callback=self.reset_curve, width=55, height=18)
                dpg.add_button(label="Delete", callback=self.delete_selected_point, width=55, height=18)
            
            # Add interpolation mode dropdown - smaller
            with dpg.group(horizontal=True):
                dpg.add_text("Mode:")
                dpg.add_combo(
                    items=self.interpolation_modes,
                    tag=self.interpolation_combo_tag,
                    callback=self.change_interpolation_mode,
                    default_value=self.current_interpolation,
                    width=75
                )
            
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

        # Luminance theme (gray/white)
        with dpg.theme(tag=self.theme_l):
            with dpg.theme_component(dpg.mvLineSeries):
                dpg.add_theme_color(dpg.mvPlotCol_Line, [192, 192, 192], category=dpg.mvThemeCat_Plots)
            with dpg.theme_component(dpg.mvScatterSeries):
                dpg.add_theme_color(dpg.mvPlotCol_MarkerFill, [192, 192, 192], category=dpg.mvThemeCat_Plots)
                dpg.add_theme_color(dpg.mvPlotCol_MarkerOutline, [192, 192, 192], category=dpg.mvThemeCat_Plots)

        # Blue theme for selected points
        with dpg.theme(tag=self.theme_blue_selected):
            with dpg.theme_component(dpg.mvScatterSeries):
                dpg.add_theme_color(dpg.mvPlotCol_MarkerFill, [30, 144, 255], category=dpg.mvThemeCat_Plots)
                dpg.add_theme_color(dpg.mvPlotCol_MarkerOutline, [30, 144, 255], category=dpg.mvThemeCat_Plots)

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

            if channel_key not in self.curves:
                self.curves[channel_key] = [(0, 0), (128, 128), (255, 255)]
            
            # If there are fewer than 2 points, reset to default
            if len(self.curves[channel_key]) < 2:
                self.curves[channel_key] = [(0, 0), (255, 255)]
            
            # Sort points by x coordinate
            self.curves[channel_key] = sorted(self.curves[channel_key], key=lambda p: p[0])
            
            # Update or create curve line
            x_values = [float(point[0]) for point in self.curves[channel_key]]
            y_values = [float(point[1]) for point in self.curves[channel_key]]
            
            # Get theme based on channel
            theme_tag = self.theme_rgb  # Default for RGB
            if self.current_channel == "R":
                theme_tag = self.theme_r
            elif self.current_channel == "G":
                theme_tag = self.theme_g
            elif self.current_channel == "B":
                theme_tag = self.theme_b
            elif self.current_channel == "L":
                theme_tag = self.theme_l
            
            # Delete previous series if they exist
            if dpg.does_item_exist(self.line_tag):
                dpg.delete_item(self.line_tag)
            
            if dpg.does_item_exist(self.points_tag):
                dpg.delete_item(self.points_tag)
            
            # Delete selected point marker if it exists
            selected_tag = f"{self.points_tag}_selected"
            if dpg.does_item_exist(selected_tag):
                dpg.delete_item(selected_tag)
            
            # Ensure the axis exists before adding series
            if not dpg.does_item_exist(self.y_axis_tag):
                return
                
            # Ensure there are no duplicate x values (could cause issues)
            seen_x = set()
            unique_points = []
            for i, x in enumerate(x_values):
                if x not in seen_x:
                    seen_x.add(x)
                    unique_points.append((x, y_values[i]))
            
            if len(unique_points) < len(x_values):
                x_values = [p[0] for p in unique_points]
                y_values = [p[1] for p in unique_points]
            
            # Ensure there are at least 2 points for a valid line
            if len(x_values) < 2:
                x_values = [0.0, 255.0]
                y_values = [0.0, 255.0]
            
            # Generate curve data based on interpolation mode
            if self.current_interpolation == "Spline" and len(x_values) >= 3:
                # Generate smooth spline curve
                curve_x, curve_y = self._generate_spline_curve(x_values, y_values)
            else:
                # Use linear interpolation (default)
                curve_x, curve_y = x_values, y_values
            
            # Add the new line series
            dpg.add_line_series(
                x=curve_x, 
                y=curve_y, 
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
                parent=self.y_axis_tag
            )
            # Bind theme to scatter points
            dpg.bind_item_theme(self.points_tag, theme_tag)
            
            # Add a larger highlight for selected point if any
            if self.selected_point is not None:
                idx, _ = self.selected_point
                # Use the channel we're currently displaying for the selected point
                if 0 <= idx < len(self.curves[channel_key]):
                    selected_x, selected_y = self.curves[channel_key][idx]
                    dpg.add_scatter_series(
                        x=[selected_x],
                        y=[selected_y],
                        tag=f"{self.points_tag}_selected",
                        parent=self.y_axis_tag
                    )
                    # Use blue theme for the selected point to make it stand out
                    dpg.bind_item_theme(f"{self.points_tag}_selected", self.theme_blue_selected)
            
        except Exception as e:
            import traceback
            traceback.print_exc()
    
    def _generate_spline_curve(self, x_values, y_values):
        """Generate a smooth spline curve through the control points"""
        try:
            from scipy.interpolate import CubicSpline
            
            # Create a cubic spline interpolation
            cs = CubicSpline(x_values, y_values, bc_type='natural')
            
            # Generate smooth curve points
            x_smooth = np.linspace(x_values[0], x_values[-1], 100)
            y_smooth = cs(x_smooth)
            
            # Clamp y values to valid range [0, 255]
            y_smooth = np.clip(y_smooth, 0, 255)
            
            return x_smooth.tolist(), y_smooth.tolist()
            
        except ImportError:
            # Fallback to linear interpolation if scipy is not available
            print("Warning: scipy not available, falling back to linear interpolation")
            return x_values, y_values
        except Exception as e:
            print(f"Error generating spline curve: {e}")
            return x_values, y_values
    
    def change_channel(self, sender, app_data, user_data):
        """Change the current channel"""
        self.current_channel = app_data  # Now app_data contains the selected value
        self.update_plot()
    
    def change_interpolation_mode(self, sender, app_data, user_data=None):
        """Change the interpolation mode between Linear and Spline"""
        self.current_interpolation = app_data
        self.update_plot()
        # Notify of changes to update the image
        self.callback(sender, app_data, user_data)
    
    def on_global_drag(self, sender, app_data):
        """Global drag handler for better drag support"""
        # Only handle drag if we have a point being dragged and mouse button is still pressed
        if (self.dragging_point is not None and 
            self.is_mouse_over_plot() and 
            dpg.is_mouse_button_down(dpg.mvMouseButton_Left)):
            x, y = self.get_mouse_plot_coordinates()
            if x >= 0 and y >= 0:  # Valid coordinates
                self.update_dragging_point(x, y)
    
    def on_global_release(self, sender, app_data):
        """Handle mouse release to stop dragging"""
        if self.dragging_point is not None:
            self.dragging_point = None
    
    def update_dragging_point(self, x, y):
        """Update the position of the currently dragging point"""
        if self.dragging_point is None:
            return
        
        # Safety check: if mouse button is not pressed, stop dragging
        if not dpg.is_mouse_button_down(dpg.mvMouseButton_Left):
            self.dragging_point = None
            return
        
        # Clamp values to valid range
        x = max(0, min(255, x))
        y = max(0, min(255, y))
        
        # Update point in all affected channels
        idx, channels = self.dragging_point
        for ch in channels:
            if 0 <= idx < len(self.curves[ch]):
                # Special cases for endpoints - lock X coordinate
                if idx == 0:  # First point
                    x = 0  # Keep at 0
                elif idx == len(self.curves[ch]) - 1:  # Last point
                    x = 255  # Keep at 255
                
                # Update the point
                self.curves[ch][idx] = (x, y)
                
                # Sort points by X coordinate to maintain proper curve order
                # but keep track of our current point for non-endpoints
                if idx != 0 and idx != len(self.curves[ch]) - 1:
                    point_value = (x, y)
                    self.curves[ch] = sorted(self.curves[ch], key=lambda p: p[0])
                    # Update the dragging index to match the new position
                    try:
                        new_idx = self.curves[ch].index(point_value)
                        self.dragging_point = (new_idx, channels)
                        self.selected_point = (new_idx, channels)
                    except ValueError:
                        pass  # Point not found, keep current index
        
        # Update the plot
        self.update_plot()
        
        # Notify of changes
        self.callback(None, None, None)
    
    def get_mouse_plot_coordinates(self):
        """Convert screen coordinates to plot coordinates"""
        try:
            # Get mouse position using the same method that works for boundary detection
            mouse_pos = dpg.get_mouse_pos(local=False)  # Use global coordinates for consistency
            
            # Get the plot's actual bounds
            if not dpg.does_item_exist(self.plot_tag):
                return 0, 0
                
            plot_rect_min = dpg.get_item_rect_min(self.plot_tag)
            plot_rect_size = dpg.get_item_rect_size(self.plot_tag)
            
            # Calculate relative position within the entire plot area
            relative_x = mouse_pos[0] - plot_rect_min[0]
            relative_y = mouse_pos[1] - plot_rect_min[1]
            
            # Check if the axes exist and get their actual limits
            if not dpg.does_item_exist(self.x_axis_tag) or not dpg.does_item_exist(self.y_axis_tag):
                return 0, 0
            
            # Get the actual axis limits (should be 0-255 for both)
            x_limits = dpg.get_axis_limits(self.x_axis_tag)
            y_limits = dpg.get_axis_limits(self.y_axis_tag)
            
            # Calculate the actual plot area (excluding margins/labels)
            # For DearPyGui plots, we need to account for the plot margins
            # These are approximate values that work well with DearPyGui
            left_margin = 60    # Space for Y axis labels
            right_margin = 30   # Right padding
            top_margin = 30     # Top padding 
            bottom_margin = 50  # Space for X axis labels
            
            # Calculate the actual plotting area
            plot_area_width = plot_rect_size[0] - left_margin - right_margin
            plot_area_height = plot_rect_size[1] - top_margin - bottom_margin
            
            # Adjust relative position to account for margins
            plot_relative_x = relative_x - left_margin
            plot_relative_y = relative_y - top_margin
            
            
            # Check if we're within the actual plot area
            if (plot_relative_x < 0 or plot_relative_x > plot_area_width or 
                plot_relative_y < 0 or plot_relative_y > plot_area_height):
                return 0, 0
            
            # Convert to normalized coordinates [0,1] within the plot area
            if plot_area_width <= 0 or plot_area_height <= 0:
                return 0, 0
            
            x_norm = plot_relative_x / plot_area_width
            y_norm = 1.0 - (plot_relative_y / plot_area_height)  # Flip Y coordinate
            
            
            # Clamp to [0,1] range
            x_norm = max(0.0, min(1.0, x_norm))
            y_norm = max(0.0, min(1.0, y_norm))
            
            # Convert to plot data coordinates using axis limits
            x_range = x_limits[1] - x_limits[0]  # Should be 255 - 0 = 255
            y_range = y_limits[1] - y_limits[0]  # Should be 255 - 0 = 255
            
            x = int(x_limits[0] + (x_norm * x_range))
            y = int(y_limits[0] + (y_norm * y_range))
            
            
            return x, y
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return 0, 0
    
    def is_mouse_over_plot(self):
        """Check if mouse is over the plot area"""
        try:
            # Use the same coordinate system as get_mouse_plot_coordinates for consistency
            mouse_pos = dpg.get_mouse_pos(local=False)  # Global coordinates
            
            # Get plot position and size
            if not dpg.does_item_exist(self.plot_tag):
                return False
            
            # Get plot bounds in screen coordinates
            plot_rect_min = dpg.get_item_rect_min(self.plot_tag)
            plot_rect_max = dpg.get_item_rect_max(self.plot_tag)
            
            # Check if mouse is within plot bounds with a small tolerance
            tolerance = 5
            is_over = (plot_rect_min[0] - tolerance <= mouse_pos[0] <= plot_rect_max[0] + tolerance and 
                      plot_rect_min[1] - tolerance <= mouse_pos[1] <= plot_rect_max[1] + tolerance)
            
            
            return is_over
            
        except Exception as e:
            return False
    
    def reset_curve(self, sender=None, app_data=None, user_data=None):
        """Reset the curve to default (linear)"""
        channel_key = self.current_channel.lower()
        if channel_key == "rgb":
            # Reset all RGB channels (don't reset luminance when resetting RGB)
            self.curves["r"] = [(0, 0), (128, 128), (255, 255)]
            self.curves["g"] = [(0, 0), (128, 128), (255, 255)]
            self.curves["b"] = [(0, 0), (128, 128), (255, 255)]
        else:
            # Reset only the selected channel (including luminance)
            self.curves[channel_key] = [(0, 0), (128, 128), (255, 255)]
        
        self.update_plot()
        # Fix: Pass all required parameters to callback
        self.callback(sender, app_data, user_data)
    
    def get_curves(self):
        """Return the curves data for image processing"""
        return {
            "curves": self.curves,
            "interpolation_mode": self.current_interpolation
        }
    
    def set_curves(self, curves_data):
        """Set the curves data and update the UI
        
        Args:
            curves_data: Dictionary containing:
                - 'curves': Dict with 'r', 'g', 'b' keys and list of (x, y) tuples
                - 'interpolation_mode': String, either "Linear" or "Spline"
        """
        if not isinstance(curves_data, dict):
            return
            
        # Set curves data if provided
        if 'curves' in curves_data and isinstance(curves_data['curves'], dict):
            # Validate and set each channel
            for channel in ['r', 'g', 'b', 'l']:
                if channel in curves_data['curves']:
                    channel_curves = curves_data['curves'][channel]
                    if isinstance(channel_curves, list):
                        # Validate that all points are tuples/lists with 2 elements
                        valid_points = []
                        for point in channel_curves:
                            if isinstance(point, (list, tuple)) and len(point) >= 2:
                                x, y = float(point[0]), float(point[1])
                                # Clamp values to valid range
                                x = max(0, min(255, x))
                                y = max(0, min(255, y))
                                valid_points.append((x, y))
                        
                        if valid_points:
                            # Ensure we have at least the endpoints
                            if len(valid_points) < 2:
                                valid_points = [(0, 0), (255, 255)]
                            
                            # Sort by x coordinate and remove duplicates
                            valid_points = sorted(set(valid_points), key=lambda p: p[0])
                            
                            # Ensure endpoints are present
                            if valid_points[0][0] != 0:
                                valid_points.insert(0, (0, 0))
                            if valid_points[-1][0] != 255:
                                valid_points.append((255, 255))
                            
                            self.curves[channel] = valid_points
        
        # Set interpolation mode if provided
        if 'interpolation_mode' in curves_data:
            mode = curves_data['interpolation_mode']
            if mode in self.interpolation_modes:
                self.current_interpolation = mode
                # Update the UI combo box if it exists
                if dpg.does_item_exist(self.interpolation_combo_tag):
                    dpg.set_value(self.interpolation_combo_tag, mode)
        
        # Update the plot to reflect the new curves
        self.update_plot()
    
    def on_plot_callback(self, sender, app_data):
        """Direct plot callback - receives plot coordinates directly"""
        
        # Check if app_data contains plot coordinates
        if app_data and isinstance(app_data, (list, tuple)) and len(app_data) >= 2:
            x, y = app_data[0], app_data[1]
            
            # Ensure coordinates are within valid range
            x = max(0, min(255, int(x)))
            y = max(0, min(255, int(y)))
            
            self.handle_plot_interaction(x, y)
        else:
            # Fall back to mouse coordinate conversion if needed
            x, y = self.get_mouse_plot_coordinates()
            if x >= 0 and y >= 0:  # Only handle if coordinates are valid
                self.handle_plot_interaction(x, y)
        
    def on_plot_clicked_direct(self, sender, app_data):
        """Direct plot click handler - simplified without double-click logic"""
        
        # Get coordinates and handle the interaction
        x, y = self.get_mouse_plot_coordinates()
        
        # Handle the click
        self.handle_plot_interaction(int(x), int(y))
            
    def on_plot_hover_direct(self, sender, app_data):
        """Direct plot hover handler"""
        # We can use this for hover effects if needed
        pass
    
    def delete_selected_point(self, sender=None, app_data=None, user_data=None):
        """Delete the currently selected point (if any)"""
        if self.selected_point is None:
            return
            
        idx, channels = self.selected_point
        
        # Get current channel for checking endpoint restrictions
        channel_key = self.current_channel.lower()
        check_key = "r" if channel_key == "rgb" else channel_key
        
        # Ensure the check_key exists
        if check_key not in self.curves:
            self.curves[check_key] = [(0, 0), (128, 128), (255, 255)]
            self.selected_point = None
            return
        
        # Don't allow deletion of first and last points (endpoints)
        if idx in (0, len(self.curves[check_key]) - 1):
            self.selected_point = None
            return
        
        # Delete the point from all affected channels
        for ch in channels:
            if ch in self.curves and 0 <= idx < len(self.curves[ch]):
                del self.curves[ch][idx]
        
        # Clear selection
        self.selected_point = None
        
        # Update the plot
        self.update_plot()
        
        # Notify of changes
        self.callback(None, None, None)
    
    def handle_plot_interaction(self, x, y):
        """Handle plot interaction with given coordinates"""
        
        # Get the current channel key
        channel_key = self.current_channel.lower()
        check_key = "r" if channel_key == "rgb" else channel_key
        channels_to_modify = ["r", "g", "b"] if channel_key == "rgb" else [channel_key]
        
        # Ensure the check_key exists in curves dictionary
        if check_key not in self.curves:
            self.curves[check_key] = [(0, 0), (128, 128), (255, 255)]
        
        # Ensure all channels to modify exist
        for ch in channels_to_modify:
            if ch not in self.curves:
                self.curves[ch] = [(0, 0), (128, 128), (255, 255)]
        
        # Check if clicking near an existing point - increased hit radius for easier selection
        hit_radius = 40  # Increased from 30 for better user experience
        for idx, point in enumerate(self.curves[check_key]):
            distance = abs(point[0] - x) + abs(point[1] - y)
            if distance < hit_radius:
                self.dragging_point = (idx, channels_to_modify)
                self.selected_point = (idx, channels_to_modify)  # Also track as selected
                
                # Update plot to show selection
                self.update_plot()
                return
        
        # If no point found, add a new one
        for ch in channels_to_modify:
            self.curves[ch].append((x, y))
        
        # Clear selection when adding new point
        self.selected_point = None
        
        # Update the plot
        self.update_plot()
        
        # Notify of changes
        self.callback(None, None, None)
