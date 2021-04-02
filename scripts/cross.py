# -*- coding: utf-8 -*-
import sys
import argparse
import numpy as np
from vpype_integration import to_vpype
import vpype
import vpype_viewer
from primitives import mask_drawing, resize_and_center, shift, jitter_length
from repro import ReproSaver


def parse_arguments():
    parser = argparse.ArgumentParser(description='',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--nosave', help='', action='store_true')
    parser.add_argument('--type', help='', choices=['rand', 'parallel'], default='parallel')

    return parser.parse_args()


def random_line(H, W, min_angle=0, max_angle=2*np.pi):
    found = False
    while not found:
        start_x = np.random.randint(W)
        start_y = np.random.randint(H)
        end_x = np.random.randint(W)
        end_y = np.random.randint(H)

        line_angle = np.arctan2(end_y - start_y, end_x - start_x)
        if line_angle >= min_angle and line_angle <= max_angle:
            break

    return np.array([[start_x, start_y],
                     [end_x, end_y]])


def draw_cross(H, W, top_fraction=0.3, width_fraction=0.08):
    width = W * width_fraction

    top = 0
    middle_top = H * top_fraction - width
    middle_bottom = H * top_fraction + width
    bottom = H
    left = 0
    middle_left = 0.5 * W - width
    middle_right = 0.5 * W + width
    right = W
    return np.array([[left, middle_top],
                     [middle_left, middle_top],
                     [middle_left, top],
                     [middle_right, top],
                     [middle_right, middle_top],
                     [right, middle_top],
                     [right, middle_bottom],
                     [middle_right, middle_bottom],
                     [middle_right, bottom],
                     [middle_left, bottom],
                     [middle_left, middle_bottom],
                     [left, middle_bottom],
                     [left, middle_top],
                     ])

def draw_cross_parts(H, W, top_fraction=0.3, width_fraction=0.08):
    width = W * width_fraction

    top = 0
    middle_top = H * top_fraction - width
    middle_bottom = H * top_fraction + width
    bottom = H
    left = 0
    middle_left = 0.5 * W - width
    middle_right = 0.5 * W + width
    right = W
    horizontal = np.array([[left, middle_top],
                           [right, middle_top],
                           [right, middle_bottom],
                           [left, middle_bottom],
                           [left, middle_top]])
    vertical = np.array([[middle_left, top],
                         [middle_right, top],
                         [middle_right, bottom],
                         [middle_left, bottom],
                         [middle_left, top]])
    return [horizontal, vertical]


def run(args):
    saver = ReproSaver()
    saver.seed()

    H, W = 210, 148
    angle_range = 3

    horizontal_lines = []
    for i in range(1000):
        horizontal_lines.append(random_line(H, W, min_angle=np.radians(0 - angle_range / 2), max_angle=np.radians(0 + angle_range / 2)))

    vertical_lines = []
    for i in range(1000):
        vertical_lines.append(random_line(H, W, min_angle=np.radians(90 - angle_range / 2), max_angle=np.radians(90 + angle_range / 2)))

    cross = draw_cross(H, W)
    horizontal, vertical = draw_cross_parts(H, W)
    margin = 10
    (horizontal, vertical, cross) = resize_and_center([horizontal, vertical, cross], H, W,
                                                      margin, margin, margin, margin)
    horizontal_lines = mask_drawing(horizontal_lines, horizontal)
    vertical_lines = mask_drawing(vertical_lines, vertical)

    horizontal_lines = jitter_length(horizontal_lines, 0, 5)
    vertical_lines = jitter_length(vertical_lines, 0, 5)

    vertical_lines = mask_drawing(vertical_lines, horizontal, invert=True)

    cross_lines = horizontal_lines
    cross_lines.extend(vertical_lines)
    # cross_lines.append(cross)

    bg_lines = []
    for i in range(300):
        bg_lines.append(random_line(0.2 * H, W))

    bg_lines = shift(bg_lines,
                     (0, 0.8 * H))

    bg_lines = mask_drawing(bg_lines, cross, invert=True)

    pts = []
    pts.extend(cross_lines)
    # pts.extend(bg_lines)

    lines = to_vpype(pts)
    document = vpype.Document(lines)
    vpype_viewer.show(document)

    if not args.nosave:
        saver.add_svg(pts)
    return 0


def main():
    args = parse_arguments()
    return run(args)


if __name__ == '__main__':
    sys.exit(main())
