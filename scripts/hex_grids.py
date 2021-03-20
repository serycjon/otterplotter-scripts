# -*- coding: utf-8 -*-
import sys
import argparse

import numpy as np
from svg import export_svg
from primitives import hex_grid, rotate
from primitives import rounded_rect, drawing_bbox, mask_drawing
from opengl_utils import Drawer


def parse_arguments():
    parser = argparse.ArgumentParser(description='',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    return parser.parse_args()


def run(args):
    N = 40
    size = 6
    # rot = 5
    # paths = rotate(hex_grid(N, N, size), np.radians(rot))
    N_rows = N
    N_cols = int(np.round(np.sqrt(2) * N))
    paths = hex_grid(N_rows, N_cols, size)
    paths_2 = hex_grid(N_rows, N_cols, size * (34 / 36))
    # paths_3 = rotate(
    #     hex_grid(N, N, size),
    #     np.radians(2 * rot)
    # )
    border = rounded_rect(drawing_bbox(paths_2), 15)
    paths = mask_drawing(paths, border)
    paths_2 = mask_drawing(paths_2, border)
    gui = Drawer()
    gui.add_lines(paths, 'k')
    gui.add_lines(paths_2, 'k')
    gui.draw()
    export_svg(paths + paths_2, '/tmp/hex_grids.svg')
    return 0


def main():
    args = parse_arguments()
    return run(args)


if __name__ == '__main__':
    sys.exit(main())
