# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import

import sys
import argparse

import numpy as np
import matplotlib.pyplot as plt
try:
    from OtterPlotter import Plotter
except ImportError:
    from fake_plotter import FakePlotter as Plotter
from geometry import triangle_lengths, subdivide_triangle
from primitives import vis_drawing, resize_and_center, drawing_stats

def parse_arguments():
    parser = argparse.ArgumentParser(description='',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    return parser.parse_args()

def run(args):
    x_margin_mm = 20
    y_margin_mm = 20
    H = 210 # A4
    W = 297 # A4

    frac = 0.02
    subdiv_step = 2.0
    min_side = 1.0

    triangle = np.array([[0, 0], [20, 0], [10, (np.sqrt(3)/2) * 20], [0, 0]]) * 10
    triangle = resize_and_center([triangle], H, W,
                                 x_margin_mm, x_margin_mm,
                                 y_margin_mm, y_margin_mm)[0]

    to_draw = [triangle.copy()]
    i = 0
    while i < 5000:
        side_lengths = np.array(triangle_lengths(triangle))
        subdiv_frac = subdiv_step / side_lengths
        subdiv_frac[:] = frac
        if np.isnan(subdiv_frac[0]):
            break
        subdivision = subdivide_triangle(triangle,
                                         subdiv_frac[0],
                                         subdiv_frac[1],
                                         subdiv_frac[2])
        center_tri = subdivision[3]
        triangle = center_tri.copy()
        to_draw.append(triangle.copy())

        max_len = np.amax(side_lengths)
        if max_len < min_side:
            break
        i += 1

    stats = drawing_stats(to_draw)
    print('stats: {}'.format(stats))

    vis_drawing(to_draw,
                'k-', lw=0.5)
    plt.plot([0, W, W, 0, 0], [0, 0, H, H, 0], 'k:')
    plt.axis('equal')
    plt.gca().invert_yaxis()
    plt.show()

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
