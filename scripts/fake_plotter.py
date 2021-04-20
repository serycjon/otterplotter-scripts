print('Using fake OtterPlotter object...')


class FakePlotter:
    def __init__(self, port="/dev/ttyUSB0", baud=9600, debug=False):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def finalize(self):
        pass

    def reset(self):
        pass

    def load_config(self, path):
        pass

    def set_plotter_limits(self, tl_xy, tr_xy, bl_xy, br_xy):
        pass

    def set_input_limits(self, tl_xy, tr_xy, bl_xy, br_xy):
        pass

    def get_pos(self):
        pass

    def get_steps(self):
        pass

    def move_motors(self, left_steps, right_steps):
        pass

    def send(self, msg):
        pass

    def wait_for_plotter_ready(self):
        pass

    def wait_for_msg(self, msg):
        pass

    def wait_for_lines(self, n_lines):
        pass

    def start(self):
        pass

    def stop(self):
        pass

    def enable(self):
        pass

    def disable(self):
        pass

    def set_speed(self, speed):
        pass

    def penup(self, urgent=False):
        pass

    def pendown(self, urgent=False):
        pass

    def set_penup(self, value):
        pass

    def set_pendown(self, value):
        pass

    def is_duplicate(self, x, y, n_decimal, max_diff=0):
        pass

    def split_segment(self, targetx, targety, max_segment_len=200.0):
        pass

    def goto(self, target_x, target_y, draw=None, skip_dupes=True, n_decimal=0,
             max_segment_len=10, max_dupe_diff=0):
        pass

    def draw_polyline(self, coords, n_decimal=0, progressbar=True, pbar=None):
        pass

    def draw_polylines(self, polylines, n_decimal=0):
        pass

    def draw_layers(self, layers, n_decimal=None):
        pass
