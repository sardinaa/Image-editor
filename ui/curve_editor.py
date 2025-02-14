import dearpygui.dearpygui as dpg
import numpy as np

class CurveEditor:
    def __init__(self, channel="r"):
        # Default control points: identity line
        self.channels = {"r": [(0,0), (255,255)],
                         "g": [(0,0), (255,255)],
                         "b": [(0,0), (255,255)]}
        self.active_channel = channel
        self.plot_tag = "curve_plot"
    
    def draw(self):
        # Draw a plot of the current curve for the active channel
        with dpg.window(label="Curve Editor", width=400, height=300):
            dpg.add_text(f"Editing {self.active_channel.upper()} channel")
            # Here you could add interactive controls to adjust the control points.
            # For simplicity, we draw the curve based on current points.
            points = self.channels[self.active_channel]
            # Create an array for the curve (using linear interpolation)
            xs = np.linspace(0, 255, 256)
            ys = np.interp(xs, [p[0] for p in points], [p[1] for p in points])
            # Use dpg.draw_line to plot the curve inside a child window:
            with dpg.drawlist(width=380, height=200, tag=self.plot_tag):
                # Draw axes:
                dpg.draw_line([0, 200], [380, 200], color=[200,200,200,255])
                dpg.draw_line([0, 0], [0, 200], color=[200,200,200,255])
                # Map points to the drawlist coordinates (assume linear mapping)
                curve_points = []
                for i in range(256):
                    x_coord = i * (380/255.0)
                    y_coord = 200 - (ys[i] * (200/255.0))
                    curve_points.append([x_coord, y_coord])
                # Draw the curve as a series of connected lines
                dpg.draw_polyline(curve_points, color=[255,0,0,255], thickness=2)
            # Add controls to switch channels and save/load presets
            dpg.add_combo(items=["R", "G", "B"], default_value=self.active_channel.upper(),
                          callback=self.channel_callback)
            dpg.add_button(label="Save Preset", callback=self.save_preset)
            dpg.add_button(label="Load Preset", callback=self.load_preset)
    
    def channel_callback(self, sender, app_data, user_data):
        self.active_channel = app_data.lower()
        # Redraw the editor (in a real app you’d update the plot)
    
    def save_preset(self, sender, app_data, user_data):
        # Save self.channels to file or a global preset store
        print("Preset saved:", self.channels)
    
    def load_preset(self, sender, app_data, user_data):
        # Load a preset into self.channels
        print("Preset loaded")
