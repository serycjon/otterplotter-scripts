# -*- coding: utf-8 -*-
import sys
import argparse
import numpy as np
from vpype_integration import to_vpype
import vpype
import vpype_viewer
from primitives import mask_drawing, resize_and_center
from repro import ReproSaver


def parse_arguments():
    parser = argparse.ArgumentParser(description='',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--nosave', help='', action='store_true')

    return parser.parse_args()


def random_line(H, W):
    start_x = np.random.randint(W)
    start_y = np.random.randint(H)
    end_x = np.random.randint(W)
    end_y = np.random.randint(H)

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


def run(args):
    saver = ReproSaver()
    saver.seed()

    H, W = 210, 148

    cross_lines = []
    for i in range(1000):
        cross_lines.append(random_line(H, W))

    cross = draw_cross(H, W)
    margin = 10
    cross = resize_and_center([cross], H, W,
                              margin, margin, margin, margin)[0]
    cross_lines = mask_drawing(cross_lines, cross)

    bg_lines = []
    for i in range(600):
        bg_lines.append(random_line(H, W))

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
