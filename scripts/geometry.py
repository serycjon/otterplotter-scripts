# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import

import numpy as np
import sys
import argparse

from primitives import vis_drawing, resize_and_center, count_segments
from primitives import optimize, drawing_stats


def parse_arguments():
    parser = argparse.ArgumentParser(description='',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    return parser.parse_args()

def inside(x, l, u):
    return x >= l and x <= u

def remap(x,
          src_min, src_max,
          dst_min, dst_max):
    x_01 = (x - src_min) / float(src_max - src_min)
    x_dst = x_01 * (dst_max - dst_min) + dst_min

    return x_dst

def segment_horizontal_intersection(seg, y):
    min_point = seg[0] if seg[0][1] < seg[1][1] else seg[1]
    max_point = seg[1] if seg[0][1] < seg[1][1] else seg[0]
    y_min = min_point[1]
    y_max = max_point[1]
    if inside(y, y_min, y_max):
        if y_min == y_max:
            return None
        x_min = min_point[0]
        x_max = max_point[0]
        y_shift = y - y_min
        fraction = y_shift / (y_max - y_min)
        x = x_min + fraction * (x_max - x_min)
        return (x, y)
    return None

def polyline_to_segments(polyline):
    segments = []
    for i in range(len(polyline) - 1):
        segments.append((polyline[i], polyline[i+1]))

    return segments

def horizontal_polygon_fill(polygon, y):
    segments = polyline_to_segments(polygon)

    intersections = []
    for seg in segments:
        intersection = segment_horizontal_intersection(seg, y)
        if intersection is not None:
            intersections.append(intersection[0])

    intersections.sort()
    fill = []
    for i in range(0, len(intersections), 2):
        x1 = intersections[i]
        x2 = intersections[i+1]
        if x1 != x2:
            fill.append(np.array([[x1, y], [x2, y]]))
    return fill

def horizontal_polygon_hatch(polygon, dist):
    y_min = np.amin(polygon[:, 1])
    y_max = np.amax(polygon[:, 1])
    ys = np.arange(y_min, y_max, dist)
    hatching = []
    for y in ys[1:]:
        hatching += horizontal_polygon_fill(polygon, y+0.0000001)
    return hatching

def rotate(drawing, angle, center=None):
    res = []
    for line in drawing:
        rotated = polyline_rotate(line, angle, center)
        res.append(rotated)
    return res

def polyline_rotate(line, angle, center=None):
    if center is None:
        center = np.array(((0, 0)))
    s = np.sin(np.radians(angle))
    c = np.cos(np.radians(angle))
    R = np.array([[c, -s],
                  [s, c]])
    shifted = line - center
    rotated = np.dot(R, shifted.T).T
    shifted = rotated + center
    return shifted

def polygon_hatch(polygon, dist, angle):
    backrotated_polygon = polyline_rotate(polygon, -angle)
    hatching = horizontal_polygon_hatch(backrotated_polygon, dist)
    hatching = [polyline_rotate(x, angle) for x in hatching]
    return hatching

def polygon_crosshatch(polygon, dist, angle):
    hatching = polygon_hatch(polygon, dist, angle)
    hatching += polygon_hatch(polygon, dist, angle - 90)
    return hatching

def segment_point(segment, fraction):
    return segment[0] + fraction * (segment[1] - segment[0])

def triangle_lengths(triangle):
    lengths = []
    for i in range(3):
        a = triangle[i]
        b = triangle[i+1]
        length = np.linalg.norm(a - b)
        lengths.append(length)

    return lengths

def subdivide_triangle(triangle, frac_a=0.5, frac_b=0.5, frac_c=0.5):
    c_a = segment_point((triangle[0], triangle[1]), frac_a)
    c_b = segment_point((triangle[1], triangle[2]), frac_b)
    c_c = segment_point((triangle[2], triangle[0]), frac_c)

    t1 = np.array(((triangle[0], c_a, c_c, triangle[0])))
    t2 = np.array(((c_a, triangle[1], c_b, c_a)))
    t3 = np.array(((c_c, c_b, triangle[2], c_c)))
    t4 = np.array(((c_a, c_b, c_c, c_a)))
    return [t1, t2, t3, t4]

def inset_triangle(tri, dist):
    tri_center = np.mean(tri[:-1, :], axis=0, keepdims=True)
    to_center = tri_center - tri
    to_center = to_center / np.linalg.norm(to_center, axis=1)[:, np.newaxis]
    return tri + dist*to_center

def run(args):
    try:
        from OtterPlotter import Plotter
    except ImportError:
        from fake_plotter import FakePlotter as Plotter
    import matplotlib.pyplot as plt
    # polygon = np.array([[0, 0], [2, 1], [1, 1.5], [0.5, 1], [0, 2], [3, 1], [0, 0]])
    # # hatching = polygon_hatch(polygon, 0.04, 45)
    # hatching = polygon_crosshatch(polygon, 0.04, 45)
    # vis_drawing([polygon] + hatching, 'b-', lw=0.5)

    x_margin_mm = 10
    y_margin_mm = 10
    H = 210 # A4
    W = 297 # A4

    min_hatch_dist = 0.05
    max_hatch_dist = 4

    min_subdivision_level = 2
    max_subdivision_level = 6

    triangle = np.array([[0, 0], [20, 0], [10, (np.sqrt(3)/2) * 20], [0, 0]]) * 10
    triangle = resize_and_center([triangle], H, W,
                                 x_margin_mm, x_margin_mm,
                                 y_margin_mm, y_margin_mm)[0]

    from collections import deque
    from opensimplex import OpenSimplex
    subdiv_noise = OpenSimplex()
    subdiv_noise_mult = 0.01

    hatch_noise = OpenSimplex(42)
    hatch_noise_mult = 0.01

    hatch_dist_noise = OpenSimplex(1117)
    hatch_dist_noise_mult = 0.05

    hatch_angle_noise = OpenSimplex(2111)
    hatch_angle_noise_mult = 0.01

    q = deque()
    q.append((triangle, 0))

    triangles = []
    while len(q) > 0:
        tri, level = q.pop()
        tri_center = np.mean(tri[:-1, :], axis=0)
        rnd = subdiv_noise.noise2d(x=tri_center[0]*subdiv_noise_mult*level,
                                   y=tri_center[1]*subdiv_noise_mult*level)
        rnd = remap(rnd, -1, 1, 0, 1)
        if not level <= min_subdivision_level \
           and (level >= max_subdivision_level or rnd < 0.4):
            triangles.append(tri)
        else:
            rnd = subdiv_noise.noise2d(x=tri_center[0]*subdiv_noise_mult*level,
                                       y=tri_center[1]*subdiv_noise_mult*level)
            rnd = remap(rnd, -1, 1, 0.3, 0.7)

            rnd2 = subdiv_noise.noise2d(x=tri_center[1]*subdiv_noise_mult*level,
                                        y=tri_center[0]*subdiv_noise_mult*level)
            rnd2 = remap(rnd2, -1, 1, 0.3, 0.7)

            rnd3 = subdiv_noise.noise2d(x=tri_center[0]*subdiv_noise_mult*level,
                                        y=tri_center[0]*subdiv_noise_mult*level)
            rnd3 = remap(rnd3, -1, 1, 0.3, 0.7)
            # rnd = 0.5
            subdivision = subdivide_triangle(tri, rnd, rnd2, rnd3)
            for new_tri in subdivision:
                q.append((new_tri, level+1))

    triangles = [inset_triangle(tri, 0.7) for tri in triangles]

    hatchings = []
    for tri in triangles:
        tri_c = np.mean(tri[:-1, :], axis=0)
        hatch_dist = remap(
            hatch_dist_noise.noise2d(x=tri_c[0]*hatch_dist_noise_mult,
                                     y=tri_c[1]*hatch_dist_noise_mult),
            -1, 1, min_hatch_dist, max_hatch_dist)
        # hatch_dist = min_hatch_dist
        hatch_angle = remap(
            hatch_angle_noise.noise2d(x=tri_c[0]*hatch_angle_noise_mult,
                                      y=tri_c[1]*hatch_angle_noise_mult),
            -1, 1, 0, 180)
        rnd = remap(hatch_noise.noise2d(x=tri_c[0]*hatch_noise_mult,
                                        y=tri_c[1]*hatch_noise_mult),
                    -1, 1, 0, 1)
        if rnd < 1/3:
            hatching = polygon_hatch(tri,
                                     hatch_dist,
                                     hatch_angle)
            hatchings += hatching
        elif rnd < 2/3:
            hatching = polygon_crosshatch(tri,
                                          hatch_dist,
                                          hatch_angle)
            hatchings += hatching


    # to_draw = []
    # backstroke = []
    # triangle[:, 1] += 10
    # for i in np.linspace(0, 1, 30):
    #     backstroke.append(triangle[0] + i * (triangle[1] - triangle[0]))
    # backstroke = np.array(backstroke)
    # to_draw = [triangle[:2], backstroke[::-1]]
    to_draw = [triangle] + triangles + hatchings

    pre_optim_stats = drawing_stats(to_draw)
    print('pre_optim_stats: {}'.format(pre_optim_stats))

    vis_drawing(to_draw,
                'k-', lw=0.5)
    plt.plot([0, W, W, 0, 0], [0, 0, H, H, 0], 'k:')
    plt.axis('equal')
    plt.gca().invert_yaxis()
    plt.show()

    to_draw = [] + optimize(triangles, verbose=False) + optimize(hatchings, verbose=False)
    optim_stats = drawing_stats(to_draw)
    print('optim_stats: {}'.format(optim_stats))

    with Plotter('/dev/ttyUSB0', 9600) as p:
        p.load_config('config.json')
        p.set_input_limits((0, 0), (W, 0),
                           (0, H), (W, H))
        p.draw_polylines(to_draw)
    return 0

def main():
    args = parse_arguments()
    return run(args)

if __name__ == '__main__':
    sys.exit(main())
