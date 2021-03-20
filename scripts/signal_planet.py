# -*- coding: utf-8 -*-
import sys
import argparse
import numpy as np
import vpype
import vpype_viewer
from primitives import shift, mask_drawing, circle
from repro import ReproSaver


def parse_arguments():
    parser = argparse.ArgumentParser(description='',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--nosave', help='', action='store_true')

    return parser.parse_args()


def circle_connections(N, R, max_delta, min_delta, init_angle=0):
    angles = np.cumsum(np.random.rand(N) * (max_delta - min_delta) + min_delta) + init_angle
    xs = R * np.cos(angles)
    ys = R * np.sin(angles)
    pts = [np.stack((xs, ys), axis=1)]
    return pts


def drop_parts(lines, space=2, passes=1):
    if passes < 1:
        return lines
    result = []
    for line in lines:
        try:
            split_i = np.random.randint(0, len(line) - 1)
            left = line[0:split_i, :]
            right = line[split_i + space:, :]
            result.append(left)
            result.append(right)
        except Exception:
            result.append(line)
    return drop_parts(result, space, passes - 1)


def draw_rays(r_min, r_max, angle_min, angle_max, N):
    result = []
    angles = np.linspace(angle_min, angle_max, N)
    for angle in angles:
        x_min, y_min = r_min * np.cos(angle), r_min * np.sin(angle)
        x_max, y_max = r_max * np.cos(angle), r_max * np.sin(angle)
        result.append(np.array([[x_min, y_min],
                                [x_max, y_max]]))
    return result


def run(args):
    saver = ReproSaver()
    saver.seed()
    R = 150
    phi = 1.6180339
    smaller = R / phi  # golden ratio
    pts = []
    pts.extend(circle_connections(1200, R, np.pi, -np.pi))
    rays = draw_rays(0, 7 * R * phi, 0, np.pi, 30)
    rays.extend(draw_rays(2.5 * R, 7 * R * phi,
                          np.pi / 60,
                          np.pi - np.pi / 60,
                          29))
    rays = shift(rays, np.array([1.7 * R, 0]))
    rays = mask_drawing(rays, circle((0, 0), 1.05 * R, N=50), invert=True)
    pts.extend(rays)
    # pts = mask_drawing(pts, circle((0, 0), R, N=50))
    # pts.extend(circle_connections(230, R, np.pi * 0.3, 0))
    # pts.extend(circle_connections(530, R, np.pi * 0.5, 0))
    # pts.extend(circle_connections(650, R, np.pi * 0.65, np.pi * 0.50))
    # pts.extend(circle_connections(700, R, np.pi * 0.95, np.pi * 0.72))
    # for i in range(1, 18):
    #     smooth_circle = circle_connections(360, R * 1.02**i, np.pi / 180, np.pi / 180)
    #     pts.extend(drop_parts(smooth_circle, space=1, passes=4))
    lines = to_vpype(pts)
    lines.crop(-2.3 * R - smaller, 2.3 * R + smaller, R + smaller, -R - smaller)  # left, bottom, right, top
    pts = from_vpype(lines)
    if not args.nosave:
        saver.add_svg(pts)
    document = vpype.Document(lines)
    # document.extend_page_size(None)
    with open('/tmp/signal_planet.svg', 'w') as fout:
        vpype.write_svg(fout, document)
    vpype_viewer.show(document)
    return 0


def to_vpype(paths):
    lc = vpype.LineCollection()
    for path in paths:
        lc.append(path[:, 0] + path[:, 1] * 1.j)
    return lc

def from_vpype(lines):
    results = []
    for line in lines:
        xs = np.real(line)
        ys = np.imag(line)
        results.append(np.stack((xs, ys), axis=1))

    return results


def main():
    args = parse_arguments()
    return run(args)


if __name__ == '__main__':
    sys.exit(main())
