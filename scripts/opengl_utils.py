"""
Based on

This example is a port to ModernGL of code by Nicolas P. Rougier from his "Python & OpenGL
for Scientific Visualization" free online book. Available under the (new) BSD License.

Book is available here:
https://github.com/rougier/python-opengl

Background information on this code:
https://github.com/rougier/python-opengl/blob/master/09-lines.rst

Original code on which this example is based:
https://github.com/rougier/python-opengl/blob/master/code/chapter-09/geom-path.py
"""

import numpy as np
from pyrr import Matrix44
from pathlib import Path

import moderngl
import moderngl_window
from moderngl_window import resources
from moderngl_window.resources import programs
from moderngl_window.meta import ProgramDescription
from moderngl_window.timers.clock import Timer
from primitives import drawing_bbox

import logging
logger = logging.getLogger(__name__)
logging.getLogger("moderngl_window").setLevel(logging.WARNING)


# prepare geometry
def star(inner=0.45, outer=1.0, n=5):
    R = np.array([inner, outer] * n)
    T = np.linspace(-0.5 * np.pi, 1.5 * np.pi, 2 * n, endpoint=False)
    P = np.zeros((2 * n, 2))
    P[:, 0] = R * np.cos(T)
    P[:, 1] = R * np.sin(T)
    return np.vstack([P, P[0]])


def rect(x, y, w, h):
    return np.array([(x, y), (x + w, y), (x + w, y + h), (x, y + h), (x, y)])


def random_lines(n=1):
    return np.random.rand(n, 2)


def build_buffers(lines, line_colors):
    """Prepare the buffers for multi-polyline rendering. Closed polyline must have their
    last point identical to their first point."""

    lines = [np.array(line, dtype="f4") for line in lines]

    indices = []
    reset_index = [-1]
    start_index = 0
    colors = []
    for line_i, line in enumerate(lines):
        if np.all(line[0] == line[-1]):  # closed path
            idx = np.arange(len(line) + 3) - 1
            idx[0], idx[-2], idx[-1] = len(line) - 1, 0, 1
        else:
            idx = np.arange(len(line) + 2) - 1
            idx[0], idx[-1] = 0, len(line) - 1

        indices.append(idx + start_index)
        start_index += len(line)
        indices.append(reset_index)

        color = np.array(line_colors[line_i]).reshape(1, 4)
        colors.append(np.repeat(color, len(line), axis=0))
    if len(lines) == 0:
        return None, None, None

    vertices = np.vstack(lines).astype("f4")
    colors = np.vstack(colors).astype("f4")

    vbo = vertices
    # vbo = np.hstack((vertices, colors)).astype("f4")
    ibo = np.concatenate(indices).astype("i4")
    cbo = colors
    return vbo, ibo, cbo


def build_point_buffers(points, point_colors):
    colors = []
    for group_i, point_group in enumerate(points):
        color = np.array(point_colors[group_i]).reshape(1, 4)
        colors.append(np.repeat(color, len(point_group), axis=0))
    if len(points) == 0:
        points = point_colors = np.zeros(0, dtype="f4")
    else:
        points = np.vstack(points).astype("f4")
        point_colors = np.vstack(colors).astype("f4")

    return points, point_colors


class RichLines():

    def __init__(self, lines=None, line_colors=None, lw=1,
                 points=None, point_colors=None, point_r=1):
        self.point_r = point_r
        # Configure to use pyglet window
        window_str = 'moderngl_window.context.pyglet.Window'
        window_cls = moderngl_window.get_window_cls(window_str)
        window = window_cls(
            title="My Window",
            gl_version=(3, 3),
            # aspect_ratio=1.0,
            # resizable=False,
            # size=(1600, 800),
        )
        self.wnd = window
        moderngl_window.activate_context(ctx=window.ctx)
        # self.wnd.gl_version = (3, 3)
        resources.register_dir(Path(__file__).parent.absolute())
        self.ctx = self.wnd.ctx

        # register event methods
        self.wnd.resize_func = self.resize
        # self.wnd.iconify_func = self.iconify
        # self.wnd.key_event_func = self.key_event
        # self.wnd.mouse_position_event_func = self.mouse_position_event
        # self.wnd.mouse_drag_event_func = self.mouse_drag_event
        # self.wnd.mouse_scroll_event_func = self.mouse_scroll_event
        # self.wnd.mouse_press_event_func = self.mouse_press_event
        # self.wnd.mouse_release_event_func = self.mouse_release_event
        # self.wnd.unicode_char_entered_func = self.unicode_char_entered

        self.line_prog = programs.load(ProgramDescription(path="rich_lines.glsl"))
        self.point_prog = programs.load(ProgramDescription(path="points.glsl"))

        bbox = drawing_bbox(lines + points)
        bbox = drawing_bbox(lines + points, padding=0.05 * bbox[2])
        self.bbox = bbox
        self.drawing_W = bbox[2]
        self.drawing_H = bbox[3]

        if len(lines) > 0:
            vertex, index, colors = build_buffers(lines, line_colors)

            vbo = self.ctx.buffer(vertex)
            ibo = self.ctx.buffer(index)
            cbo = self.ctx.buffer(colors)
            self.line_vao = self.ctx.vertex_array(self.line_prog,
                                                  [
                                                      (vbo, "2f", "in_position"),
                                                      (cbo, "4f", "in_color"),
                                                  ],
                                                  index_buffer=ibo)
        else:
            self.line_vao = None

        if len(points) > 0:
            point_vertex, point_color = build_point_buffers(points, point_colors)
            vbo = self.ctx.buffer(point_vertex)
            cbo = self.ctx.buffer(point_color)
            self.point_vao = self.ctx.vertex_array(self.point_prog,
                                                   [
                                                       (vbo, "2f", "in_position"),
                                                       (cbo, "4f", "in_color"),
                                                   ])
        else:
            self.point_vao = None

        # Set the desired properties for the lines.
        # Note:
        # - round cap/ends are used if miter_limit < 0
        # - antialias value is in model space and should probably be scaled to be ~1.5px in
        #   screen space

        self.line_prog["linewidth"].value = lw
        self.line_prog["antialias"].value = 1.5
        self.line_prog["miter_limit"].value = -1
        # self.line_prog["color"].value = 0, 0, 0, 1

        self.update_projection()

    def update_projection(self):
        """ https://stackoverflow.com/a/35817409/1705970 """
        bbox = self.bbox
        bbox_ar = bbox[2] / bbox[3]
        viewport = self.wnd.viewport
        viewport_ar = viewport[2] / viewport[3]

        if viewport_ar >= bbox_ar:
            # wide viewport, use full height
            proj_width = (viewport_ar / bbox_ar) * bbox[2]
            proj_height = bbox[3]
        else:
            # tall viewport, use full width
            proj_width = bbox[2]
            proj_height = (bbox_ar / viewport_ar) * bbox[3]

        bbox_center = bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2
        proj_left = bbox_center[0] - proj_width / 2
        proj_right = proj_left + proj_width

        proj_top = bbox_center[1] - proj_height / 2
        proj_bottom = proj_top + proj_height

        proj_matrix = Matrix44.orthogonal_projection(left=proj_left, right=proj_right,
                                                     top=proj_bottom, bottom=proj_top,  # opengl has y-axis aiming up
                                                     near=0.5, far=-0.5, dtype="f4")
        self.line_prog["projection"].write(proj_matrix)
        self.point_prog["projection"].write(proj_matrix)

    def render(self, time, frame_time):
        self.ctx.clear(1, 1, 1, 1)
        if self.line_vao is not None:
            self.ctx.enable_only(moderngl.BLEND)
            self.line_vao.render(moderngl.LINE_STRIP_ADJACENCY)

        if self.point_vao is not None:
            self.ctx.enable_only(moderngl.PROGRAM_POINT_SIZE | moderngl.BLEND)
            self.point_vao.render(moderngl.POINTS)

    def run(self):
        timer = Timer()
        timer.start()

        while not self.wnd.is_closing:
            self.wnd.clear()
            time, frame_time = timer.next_frame()
            self.render(time, frame_time)
            self.wnd.swap_buffers()

        self.wnd.destroy()

    def resize(self, width: int, height: int):
        # print("Window was resized. buffer size is {} x {}".format(width, height))
        self.line_prog["antialias"].value = 1.5 * (self.bbox[2] / width)

        # recompute point radius from drawing units to pixels ???
        self.point_prog["point_r"].value = self.point_r * (width / self.bbox[2])
        self.update_projection()

    def key_event(self, key, action, modifiers):
        keys = self.wnd.keys

        # Key presses
        if action == keys.ACTION_PRESS:
            if key == keys.SPACE:
                print("SPACE key was pressed")

            # Using modifiers (shift and ctrl)

            if key == keys.Z and modifiers.shift:
                print("Shift + Z was pressed")

            if key == keys.Z and modifiers.ctrl:
                print("ctrl + Z was pressed")

        # Key releases
        elif action == self.wnd.keys.ACTION_RELEASE:
            if key == keys.SPACE:
                print("SPACE key was released")

        # Move the window around with AWSD
        if action == keys.ACTION_PRESS:
            if key == keys.A:
                self.wnd.position = self.wnd.position[0] - 10, self.wnd.position[1]
            if key == keys.D:
                self.wnd.position = self.wnd.position[0] + 10, self.wnd.position[1]
            if key == keys.W:
                self.wnd.position = self.wnd.position[0], self.wnd.position[1] - 10
            if key == keys.S:
                self.wnd.position = self.wnd.position[0], self.wnd.position[1] + 10

            # toggle cursor
            if key == keys.C:
                self.wnd.cursor = not self.wnd.cursor

            # Toggle mouse exclusivity
            if key == keys.M:
                self.wnd.mouse_exclusivity = not self.wnd.mouse_exclusivity

    def mouse_position_event(self, x, y, dx, dy):
        print("Mouse position pos={} {} delta={} {}".format(x, y, dx, dy))

    def mouse_drag_event(self, x, y, dx, dy):
        print("Mouse drag pos={} {} delta={} {}".format(x, y, dx, dy))

    def mouse_scroll_event(self, x_offset, y_offset):
        print("mouse_scroll_event", x_offset, y_offset)

    def mouse_press_event(self, x, y, button):
        print("Mouse button {} pressed at {}, {}".format(button, x, y))
        print("Mouse states:", self.wnd.mouse_states)

    def mouse_release_event(self, x: int, y: int, button: int):
        print("Mouse button {} released at {}, {}".format(button, x, y))
        print("Mouse states:", self.wnd.mouse_states)

    def unicode_char_entered(self, char):
        print("unicode_char_entered:", char)

    def close(self):
        print("Window was closed")


class Drawer():
    def __init__(self, point_r=1, lw=1):
        self.lines = []
        self.colors = []
        self.points = []
        self.point_colors = []
        self.point_r = point_r
        self.lw = lw

    def add_lines(self, drawing, color=None):
        '''
        args:
            drawing: a list of K (N, 2) numpy arrays
            color: color code or a rgba 4-tuple (values 0-1)
        '''
        color = _decode_color(color)
        self.lines.extend(drawing)
        self.colors.extend([color] * len(drawing))

    def add_points(self, points, color=None):
        """
        args:
            points: a list of K (N, 2) numpy arrays
            color: color code or a rgba 4-tuple (values 0-1)
        """
        color = _decode_color(color)
        self.points.extend(points)
        self.point_colors.extend([color] * len(points))

    def draw(self):
        app = RichLines(lines=self.lines, line_colors=self.colors,
                        points=self.points, point_colors=self.point_colors,
                        point_r=self.point_r, lw=self.lw)
        app.run()


def _decode_color(color):
    color_dict = {'k': (0, 0, 0, 1),
                  'r': (1, 0, 0, 1),
                  'g': (0, 1, 0, 1),
                  'b': (0, 0, 1, 1),
                  }
    if color is None:
        color = (0, 0, 0, 1)
    if color in color_dict:
        color = color_dict[color]
    return color


if __name__ == '__main__':
    format = "[%(asctime)s] %(levelname)s:%(name)s:%(message)s"
    logging.basicConfig(level=logging.DEBUG, format=format)
    lines = [
        star(n=5) * 300 + (400, 400),
        star(n=8) * 150 + (900, 200),
        rect(900, 600, 150, 50),
        [(1200, 100), (1400, 200), (1300, 100)],
        random_lines(n=100000) * 700 + (600, 400),
        rect(900, 600, 50, 150)
    ]

    line_colors = [(1, 0, 0, 1), (1, 0, 0, 1),
                   (0, 1, 0, 1), (0, 1, 0, 1),
                   (0, 0, 0, 0.005), (0, 1, 0, 1)]

    line_colors = [np.random.rand(3).tolist() + [1] for line in lines]

    # points = []
    # point_colors = []
    points = [np.array([[400, 400]]), np.array([[900, 200]])]
    point_colors = [(0, 0, 0, 1), (1, 0, 0, 1)]

    RichLines(lines=lines, line_colors=line_colors,
              points=points, point_colors=point_colors,
              lw=2, point_r=15).run()
