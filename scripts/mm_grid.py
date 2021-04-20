# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import

import sys
import argparse

from primitives import grid, optimize
from geometry import rotate
try:
    from OtterPlotter import Plotter
except ImportError:
    from fake_plotter import FakePlotter as Plotter
import logging


def parse_arguments():
    parser = argparse.ArgumentParser(description='',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--rot', help='degrees of rotation', type=int, default=0)
    parser.add_argument('--grid_gap_mm', help='mm resolution of the grid', type=float, default=1)

    return parser.parse_args()

def run(args):
    y_margin_mm = 30
    x_margin_mm = 30
    H = 210 # A4
    W = 297 # A4
    grid_gap_mm = args.grid_gap_mm

    points = grid((x_margin_mm, y_margin_mm),
                  (W - x_margin_mm, H - y_margin_mm),
                  (grid_gap_mm, grid_gap_mm))
    points = optimize(points)
    if args.rot != 0:
        points = rotate(points, args.rot)

    with Plotter('/dev/ttyUSB0', 115200) as p:
        p.load_config('config.json')
        p.set_input_limits((0, 0), (W, 0),
                           (0, H), (W, H))
        p.draw_polylines(points)

    return 0

def main():
    args = parse_arguments()
    return run(args)

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    sys.exit(main())
