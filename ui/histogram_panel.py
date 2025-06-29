# ui/histogram_panel.py
import dearpygui.dearpygui as dpg
import numpy as np
import cv2

class HistogramPanel:
    def __init__(self, width=None):
        self.plot_tag = "histogram_plot"
        self.x_axis_tag = "histogram_x_axis"
        self.y_axis_tag = "histogram_y_axis"
        
        # Series tags for each channel
        self.red_series_tag = "histogram_red_series"
        self.green_series_tag = "histogram_green_series"
        self.blue_series_tag = "histogram_blue_series"
        self.luminance_series_tag = "histogram_luminance_series"
        
        # Plot dimensions - dynamic width to use available space
        self.plot_width = width if width is not None else 220  # Default fallback
        self.plot_height = 120  # Keep height reasonable
        
        # Current display mode
        self.show_rgb = True
        self.show_luminance = False
        
        # Update throttling for histogram - much more responsive for real-time updates
        self.last_update_time = 0
        self.update_threshold = 0.05  # Update at most every 50ms for smooth real-time feel
        self.current_hist_data = None
        
        # Create themes for histogram colors
        self._create_histogram_themes()
        
    def _create_histogram_themes(self):
        """Create themes for different histogram channels"""
        # Red channel theme
        if not dpg.does_item_exist("histogram_theme_red"):
            with dpg.theme(tag="histogram_theme_red"):
                with dpg.theme_component(dpg.mvLineSeries):
                    dpg.add_theme_color(dpg.mvPlotCol_Line, [255, 100, 100, 150], category=dpg.mvThemeCat_Plots)
        
        # Green channel theme
        if not dpg.does_item_exist("histogram_theme_green"):
            with dpg.theme(tag="histogram_theme_green"):
                with dpg.theme_component(dpg.mvLineSeries):
                    dpg.add_theme_color(dpg.mvPlotCol_Line, [100, 255, 100, 150], category=dpg.mvThemeCat_Plots)
        
        # Blue channel theme
        if not dpg.does_item_exist("histogram_theme_blue"):
            with dpg.theme(tag="histogram_theme_blue"):
                with dpg.theme_component(dpg.mvLineSeries):
                    dpg.add_theme_color(dpg.mvPlotCol_Line, [100, 100, 255, 150], category=dpg.mvThemeCat_Plots)
        
        # Luminance theme
        if not dpg.does_item_exist("histogram_theme_luminance"):
            with dpg.theme(tag="histogram_theme_luminance"):
                with dpg.theme_component(dpg.mvLineSeries):
                    dpg.add_theme_color(dpg.mvPlotCol_Line, [220, 220, 220, 180], category=dpg.mvThemeCat_Plots)
    
    def calculate_histogram(self, image):
        """Calculate histogram data for RGB channels and luminance"""
        if image is None:
            return None, None, None, None
        
        # Ensure image is RGB (not RGBA)
        if len(image.shape) == 3 and image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        elif len(image.shape) == 3 and image.shape[2] == 3:
            # Image is already RGB
            pass
        else:
            return None, None, None, None
            
        # For real-time performance, use 128 bins instead of 256 (reduces computation)
        num_bins = 128
        hist_range = [0, 256]
        
        # Calculate histogram for each channel
        hist_r = cv2.calcHist([image], [0], None, [num_bins], hist_range)
        hist_g = cv2.calcHist([image], [1], None, [num_bins], hist_range)
        hist_b = cv2.calcHist([image], [2], None, [num_bins], hist_range)
        
        # Calculate luminance histogram
        # Convert to grayscale for luminance
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        hist_lum = cv2.calcHist([gray], [0], None, [num_bins], hist_range)
        
        # Flatten the histograms
        hist_r = hist_r.flatten()
        hist_g = hist_g.flatten()
        hist_b = hist_b.flatten()
        hist_lum = hist_lum.flatten()
        
        # Apply minimal smoothing for real-time performance (smaller window for speed)
        def smooth_histogram(hist, window=1):
            """Apply a simple moving average to smooth the histogram"""
            if window <= 0:
                return hist
            smoothed = np.copy(hist)
            for i in range(window, len(hist) - window):
                smoothed[i] = np.mean(hist[i-window:i+window+1])
            return smoothed
        
        hist_r = smooth_histogram(hist_r)
        hist_g = smooth_histogram(hist_g)
        hist_b = smooth_histogram(hist_b)
        hist_lum = smooth_histogram(hist_lum)
        
        # Use square root scaling instead of log for smoother appearance
        hist_r = np.sqrt(hist_r)
        hist_g = np.sqrt(hist_g)
        hist_b = np.sqrt(hist_b)
        hist_lum = np.sqrt(hist_lum)
        
        return hist_r, hist_g, hist_b, hist_lum
    
    def set_width(self, width):
        """Update the histogram plot width."""
        self.plot_width = width
        # Update the plot width if it exists - leave small right margin
        if dpg.does_item_exist(self.plot_tag):
            plot_width = width - 10 if width > 0 else - 10  # Leave 15px right margin
            dpg.configure_item(self.plot_tag, width=plot_width)
    
    def create_panel(self):
        """Create the histogram panel"""
        with dpg.group():
            # Header with toggle options
            with dpg.group(horizontal=True):
                dpg.add_text("Histogram", color=[176, 204, 255])
                dpg.add_spacer(width=10)
                dpg.add_checkbox(label="RGB", tag="histogram_show_rgb", 
                               default_value=True, callback=self.toggle_display_mode)
                dpg.add_checkbox(label="Lum", tag="histogram_show_luminance", 
                               default_value=False, callback=self.toggle_display_mode)
            
            # Histogram plot - use available width with small right margin
            plot_width = self.plot_width - 5 if self.plot_width > 0 else -5  # Leave 15px right margin
            with dpg.plot(tag=self.plot_tag, height=self.plot_height, width=plot_width,
                         no_mouse_pos=True, no_box_select=True):
                # Add axes
                dpg.add_plot_axis(dpg.mvXAxis, label="", tag=self.x_axis_tag, no_gridlines=True, no_tick_labels=True)
                dpg.add_plot_axis(dpg.mvYAxis, label="", tag=self.y_axis_tag, no_gridlines=True, no_tick_labels=True)
                
                # Set axes limits
                dpg.set_axis_limits(self.x_axis_tag, 0, 255)
                dpg.set_axis_limits(self.y_axis_tag, 0, 10)  # Will adjust based on data
    
    def toggle_display_mode(self, sender, app_data, user_data):
        """Toggle between different histogram display modes"""
        self.show_rgb = dpg.get_value("histogram_show_rgb")
        self.show_luminance = dpg.get_value("histogram_show_luminance")
        
        # Update the histogram display
        self.update_display()
    
    def update_histogram(self, image):
        """Update histogram with new image data"""
        import time
        
        # Throttle updates to prevent oscillations
        current_time = time.time()
        if current_time - self.last_update_time < self.update_threshold:
            return
        self.last_update_time = current_time
        
        if not dpg.does_item_exist(self.plot_tag):
            return
            
        # Calculate histogram data
        hist_r, hist_g, hist_b, hist_lum = self.calculate_histogram(image)
        
        if hist_r is None:
            return
        
        # Store the data for potential redraw
        self.current_hist_data = (hist_r, hist_g, hist_b, hist_lum)
        
        # Update the display
        self.update_display()
    
    def update_display(self):
        """Update the histogram display based on current settings"""
        if not hasattr(self, 'current_hist_data') or self.current_hist_data is None:
            return
            
        hist_r, hist_g, hist_b, hist_lum = self.current_hist_data
        
        # Clear existing series
        self._clear_series()
        
        # X values (0-127 for 128 bins, scaled to 0-255 range for display)
        x_values = [i * 2 for i in range(len(hist_r))]  # Scale to 0-255 range
        
        # Find max value for Y axis scaling with some stability
        max_val = 0
        if self.show_rgb:
            max_val = max(max_val, np.max(hist_r), np.max(hist_g), np.max(hist_b))
        if self.show_luminance:
            max_val = max(max_val, np.max(hist_lum))
        
        # Set Y axis limit with some padding and minimum threshold
        if max_val > 0:
            # Add padding and ensure minimum scale for stability
            y_max = max(max_val * 1.2, 10)
            dpg.set_axis_limits(self.y_axis_tag, 0, y_max)
        
        # Add series based on current settings
        if self.show_rgb:
            # Add RGB channel series
            dpg.add_line_series(x=x_values, y=hist_r.tolist(), 
                               tag=self.red_series_tag, parent=self.y_axis_tag)
            dpg.bind_item_theme(self.red_series_tag, "histogram_theme_red")
            
            dpg.add_line_series(x=x_values, y=hist_g.tolist(), 
                               tag=self.green_series_tag, parent=self.y_axis_tag)
            dpg.bind_item_theme(self.green_series_tag, "histogram_theme_green")
            
            dpg.add_line_series(x=x_values, y=hist_b.tolist(), 
                               tag=self.blue_series_tag, parent=self.y_axis_tag)
            dpg.bind_item_theme(self.blue_series_tag, "histogram_theme_blue")
        
        if self.show_luminance:
            # Add luminance series
            dpg.add_line_series(x=x_values, y=hist_lum.tolist(), 
                               tag=self.luminance_series_tag, parent=self.y_axis_tag)
            dpg.bind_item_theme(self.luminance_series_tag, "histogram_theme_luminance")
    
    def _clear_series(self):
        """Clear all existing histogram series"""
        series_tags = [
            self.red_series_tag,
            self.green_series_tag, 
            self.blue_series_tag,
            self.luminance_series_tag
        ]
        
        for tag in series_tags:
            if dpg.does_item_exist(tag):
                dpg.delete_item(tag)
    
    def show(self):
        """Show the histogram panel"""
        self.create_panel()