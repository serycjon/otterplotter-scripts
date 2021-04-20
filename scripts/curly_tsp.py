# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import

import os
import sys
import argparse

import cv2
import numpy as np

from concorde_tsp import concorde_tsp
from caching import CachedComputation
from img_tools import ascencio2010stippling
from img_tools import clip_to_img, resize_to_max
from repro import tmp_np_seed
from various_utils import with_debugger
from scipy.ndimage import distance_transform_edt
from scipy.signal import medfilt

from opensimplex import OpenSimplex
try:
    from OtterPlotter import Plotter
except ImportError:
    from fake_plotter import FakePlotter as Plotter
from geometry import remap
from primitives import resize_and_center, vis_drawing
from primitives import optimize, rotate
from primitives import rotation_matrix
from repro import ReproSaver
import matplotlib.pyplot as plt
from opengl_utils import Drawer

import tqdm
import logging

logger = logging.getLogger(__name__)
format = "[%(asctime)s] %(levelname)s:%(name)s:%(message)s"
logging.basicConfig(level=logging.DEBUG, format=format)
logging.getLogger("asyncio").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    exclusive = parser.add_mutually_exclusive_group(required=True)
    exclusive.add_argument("--img", help="image path")
    exclusive.add_argument('--tonal_map', help='compute tonal mapping LUT', action='store_true')
    exclusive.add_argument('--straight_test', help='test scribbling on straight line', action='store_true')
    parser.add_argument('--noopt', help='disable optimization', action='store_true')
    parser.add_argument('--nosave', help='do not save results', action='store_true')
    parser.add_argument('--force', help='force recomputation of cached results', action='store_true')
    parser.add_argument('--nolut', help='disable tonal mapping', action='store_true')
    return parser.parse_args()


@with_debugger
def run(args):
    saver = ReproSaver()
    saver.seed()

    if args.img is not None:
        drawing_name = os.path.splitext(os.path.basename(args.img))[0]
        img = cv2.imread(args.img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        drawing_name = 'tonal_map'
        orig_tonal_map = np.linspace(0, 1, num=256)
        gray = np.tile(255 * orig_tonal_map.reshape(1, orig_tonal_map.size), (40, 1)).astype(np.uint8)
    gray = resize_to_max(gray, 4000)

    logger.info("drawing_name: {}".format(drawing_name))
    edges = cv2.Canny(gray, 60, 80)
    edge_distance = distance_transform_edt(255 - edges)
    # cv2.imshow("cv: edges", edges)
    # cv2.imshow("cv: edge_distance", edge_distance / np.amax(edge_distance))
    # while True:
    #     c = cv2.waitKey(0)
    #     if c == ord('q'):
    #         import sys
    #         sys.exit(1)

    density = 1 - (gray.astype(np.float64) / 255)

    def stippling_density(density):
        return density ** 3

    logger.info("stippling")
    with tmp_np_seed(42):
        stippling_fn = CachedComputation(
            os.path.join("cache", f"curly_tsp_stipple_{drawing_name}.pkl"),
            force=args.force)(
                ascencio2010stippling)
        points = stippling_fn(stippling_density(density), K=200, min_r=2, max_r=75)

    gui = Drawer(point_r=10)
    # gui.add_lines([points.copy()], 'g')
    gui.add_points([points.copy()], 'k')
    # gui.draw()

    logger.info("TSP-ing")
    tsp_fn = CachedComputation(
        os.path.join("cache", f"curly_tsp_{drawing_name}.pkl"),
        force=args.force)(
            concorde_tsp)
    tsp_pts = tsp_fn(points)

    if args.straight_test:
        logger.debug('Using straight line instead of TSP path')
        tsp_pts = np.array([[20, gray.shape[0] / 2],
                            [gray.shape[1] - 20, gray.shape[0] / 2]])

    gui = Drawer()
    gui.add_lines([tsp_pts], 'k')
    # gui.draw()

    logger.info("squiggling")
    squiggle_fn = CachedComputation(
        os.path.join("cache", f"curly_tsp_squiggle_{drawing_name}.pkl"),
        force=args.force)(
            chiu2015tone_scribble)

    squiggle_fn = chiu2015tone_scribble

    if os.path.exists('curly_tsp_LUT.npy') and not args.tonal_map and not args.straight_test and not args.nolut:
        LUT = np.load('curly_tsp_LUT.npy')
        logger.info('using tonal mapping LUT')
        gray = apply_LUT(gray, LUT)

    speed_img = gray
    speed_img = speed_img.astype(np.float32) / 255
    if args.tonal_map or args.straight_test:
        edge_distance = None
    to_draw = [squiggle_fn(tsp_pts, r=185,
                           speed_img=speed_img,
                           edge_dist_img=edge_distance)]

    gui = Drawer(lw=5)
    gui.add_lines(to_draw, (0, 0, 0, 1))
    gui.draw()

    if args.tonal_map:
        rendered = render_lines(gray, to_draw[0])
        sigma = 19
        blurred_rendered = cv2.GaussianBlur(rendered, ksize=(0, 0), sigmaX=sigma)
        rendered_tonal_map = np.mean(blurred_rendered.astype(np.float32) / 255, axis=0)
        orig_tonal_map = gray[0, :].astype(np.float32) / 255
        rendered_tones = np.tile(rendered_tonal_map[np.newaxis, :], (gray.shape[0], 1))
        cv2.imshow("cv: rendered", rendered)
        cv2.imshow("cv: blurred rendered", blurred_rendered)
        cv2.imshow("cv: rendered_tones", rendered_tones)
        cv2.imshow("cv: gray", gray)
        while True:
            c = cv2.waitKey(0)
            if c == ord('q'):
                break
        cv2.destroyAllWindows()
        lut = compute_tonal_mapping(rendered_tonal_map, orig_tonal_map)
        np.save('curly_tsp_LUT', lut)

    if not args.nosave:
        saver.add_svg(to_draw)

    x_margin_mm = 5
    y_margin_mm = 5
    H = 210  # A4
    W = 297  # A4

    to_draw = rotate(to_draw)
    to_draw = resize_and_center(
        to_draw, H, W, x_margin_mm, x_margin_mm, y_margin_mm, y_margin_mm
    )

    border = [np.array([[0, 0], [W, 0], [W, H], [0, H], [0, 0]])]
    # gui = Drawer(point_r=2)
    # gui.add_lines(to_draw, 'b')
    # gui.add_lines(border, 'r')
    # gui.add_points(border, 'g')
    # gui.draw()
    # vis_drawing(to_draw, "r-", linewidth=0.5)
    if not args.noopt:
        to_draw = optimize(to_draw, line_simplification_threshold=0.1)

    vis_drawing(to_draw, "k-", linewidth=0.5)
    plt.plot([0, W, W, 0, 0], [0, 0, H, H, 0], "k:")
    plt.axis("equal")
    plt.gca().invert_yaxis()
    plt.show()

    with Plotter() as p:
        p.load_config("config.json")
        p.set_input_limits((0, 0), (W, 0), (0, H), (W, H))
        p.draw_polylines(to_draw)

    return 0


def squiggle(points, r=10):
    r_noise = OpenSimplex(42)
    angle_noise = OpenSimplex(211)
    result = []
    r_noise_mult = 0.01
    angle_noise_mult = 0.02
    angles = []
    pts = np.concatenate((points[-1, np.newaxis, :], points), axis=0).astype(np.float32)
    prev_dist = 0
    for i in tqdm.tqdm(range(1, len(pts))):
        prev_pt = pts[i - 1, :]
        pt = pts[i, :]
        dist = np.linalg.norm(prev_pt - pt)
        for step in np.linspace(0, 1, 100):
            cur_dist = step * dist + (1 - step) * prev_dist
            x = prev_pt[0] + step * (pt[0] - prev_pt[0])
            y = prev_pt[1] + step * (pt[1] - prev_pt[1])

            new_r = remap(
                r_noise.noise2d(x=x * r_noise_mult, y=y * r_noise_mult),
                -1,
                1,
                cur_dist / 2,
                cur_dist,
            )
            new_angle = remap(
                angle_noise.noise2d(x=x * angle_noise_mult, y=y * angle_noise_mult),
                -1,
                1,
                0,
                2 * np.pi,
            )
            # new_angle = angle
            angles.append(new_angle)
            x = new_r * np.cos(new_angle) + prev_pt[0] + step * (pt[0] - prev_pt[0])
            y = new_r * np.sin(new_angle) + prev_pt[1] + step * (pt[1] - prev_pt[1])
            result.append((x, y))
        prev_dist = dist

    return np.array(result)


def chiu2015tone_scribble(points, r=1,
                          speed_img=None, edge_dist_img=None):
    points = points.astype(np.float32)
    r_abs_min = 10
    r_min = 20
    r_max = r
    r = r  # disk radius

    # state
    tracer = PolylineTracer(points)
    pos = tracer.step(0)
    angle = 0

    angular_velocity = np.pi / 10  # radians per time unit
    # min_pos_velocity = 0.1  # pixels per time unit
    # max_pos_velocity = 5  # pixels per time unit
    min_pos_velocity = 0.1  # pixels per time unit
    max_pos_velocity = 13  # pixels per time unit
    step_frequency = 1  # steps per time unit

    def velocity_at_pos(x):
        if speed_img is None:
            coeff = 1
        else:
            int_pos = clip_to_img(np.int32(np.round(x)), speed_img)
            coeff = speed_img[int_pos[1], int_pos[0]]
            # coeff = piecewise_linear(coeff,
            #                          (0, 0.0),
            #                          (0.2, 0.1),
            #                          (0.21, 1),
            #                          (1, 1))
            # coeff = 0.2

        velocity = min_pos_velocity + coeff * (max_pos_velocity - min_pos_velocity)
        return velocity

    def radius_at_pos(x):
        int_pos = clip_to_img(np.int32(np.round(x)), speed_img)

        coeff = speed_img[int_pos[1], int_pos[0]]
        # coeff = piecewise_linear(coeff,
        #                          (0, 0.0),
        #                          (0.6, 0.1),
        #                          (0.61, 1),
        #                          (1, 1))
        radius = r_min + coeff * (r_max - r_min)
        if edge_dist_img is None:
            return radius
        else:
            edge_dist = edge_dist_img[int_pos[1], int_pos[0]]
            radius = np.clip(radius, r_min, edge_dist)
            radius = np.clip(radius, r_abs_min, r_max)
            return radius

    theta_noise = OpenSimplex(42)
    flat_noise = OpenSimplex(211)

    step = 0
    output_pts = []
    while pos is not None:
        step += 1
        # if step > 1000:
        #     break
        current_velocity = velocity_at_pos(pos)
        current_radius = radius_at_pos(pos)

        theta = remap(theta_noise.noise2d(x=0, y=step * 1e-2 / step_frequency),
                      -1, 1, 0, 2 * np.pi)
        flatness = remap(flat_noise.noise2d(x=0, y=step * 1e-2 / step_frequency),
                         -1, 1, 0.2, 1)
        R = rotation_matrix(theta)
        scribble_pos = np.array((flatness * current_radius * np.cos(angle),
                                 current_radius * np.sin(angle)))
        scribble_pos = np.matmul(R, scribble_pos)
        scribble_pos = pos + scribble_pos

        output_pts.append(scribble_pos)

        pos = tracer.step(current_velocity / step_frequency)
        # pos = tracer.step(4)
        angle += angular_velocity / step_frequency

    output_pts = np.array(output_pts)
    return output_pts


class PolylineTracer():
    def __init__(self, points):
        self.points = points.astype(np.float32)
        self.pos = points[0].copy()
        self.target_i = 1
        self.pbar = tqdm.tqdm(total=len(points) - 1)

    def step(self, dist):
        dist_left = dist
        while dist_left > 0:
            try:
                target = self.points[self.target_i]
            except IndexError:
                return None
            self.pos, dist_left = go_along_segment(self.pos, target, dist)
            if dist_left > 0:
                self.target_i += 1
                self.pbar.update(1)

        return self.pos


def go_along_segment(start, end, distance):
    seg_vector = end - start
    seg_length = np.linalg.norm(seg_vector)
    if distance > seg_length:
        return end, distance - seg_length
    else:
        direction = seg_vector / seg_length
        pos = start + direction * distance
        return pos, 0


def normalize(x):
    return x / (np.linalg.norm(x) + 1e-10)


def line_seg_pos(pt, start, end):
    segment_vector = end - start
    pos_vector = pt - start
    projection = np.dot(pos_vector, segment_vector)
    segment_fraction = projection / (np.linalg.norm(segment_vector) + 1e-10)
    return segment_fraction


def draw_tour(canvas, points, tour, color):
    for i in range(1, len(tour)):
        start_id = tour[i - 1]
        end_id = tour[i]

        cv2.line(
            canvas,
            (int(round(points[start_id, 0])), int(round(points[start_id, 1]))),
            (int(round(points[end_id, 0])), int(round(points[end_id, 1]))),
            color,
            1,
        )


def render_lines(orig_img, line):
    canvas = np.ones_like(orig_img) * 255
    for i in range(len(line) - 1):
        start = np.round(line[i]).astype(np.int32)
        end = np.round(line[i + 1]).astype(np.int32)
        cv2.line(canvas, (start[0], start[1]), (end[0], end[1]), 0, thickness=2)
    return canvas


def compute_tonal_mapping(rendered, orig):
    assert len(rendered.shape) == 1
    assert rendered.shape == orig.shape
    LUT = []
    orig_coords = np.round(np.linspace(0, len(orig) - 1, 256)).astype(np.int32)
    for orig_i in orig_coords:
        orig_val = orig[orig_i]
        best_diff = np.inf
        best_dist = np.inf
        best_rendered_i = None
        for rendered_i, rendered_val in enumerate(rendered):
            dist = np.abs(orig_i - rendered_i)
            diff = np.abs(orig_val - rendered_val)
            if diff == best_diff:
                if dist < best_dist:
                    best_dist = dist
                    best_diff = diff
                    best_rendered_i = rendered_i
            elif diff < best_diff:
                best_dist = dist
                best_diff = diff
                best_rendered_i = rendered_i
        LUT.append(int(np.round(255 * (best_rendered_i / rendered.size))))

    LUT = np.array(LUT)

    filtered_LUT = np.round(medfilt(LUT, kernel_size=3)).astype(np.int32)

    plt.plot(LUT, 'b-')
    plt.plot(filtered_LUT, 'r-')
    plt.show()
    return filtered_LUT


def apply_LUT(img, LUT):
    return LUT[img.flatten()].reshape(img.shape).astype(np.uint8)


def piecewise_linear(x, *control_pts):
    start = control_pts[0]
    for ctrl_x, ctrl_y in control_pts[1:]:
        if ctrl_x >= x:
            end = (ctrl_x, ctrl_y)
            break
        start = (ctrl_x, ctrl_y)

    frac = (x - start[0]) / (end[0] - start[0])
    val = start[1] + frac * (end[1] - start[1])
    return val


def main():
    args = parse_arguments()
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
