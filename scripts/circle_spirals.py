# -*- coding: utf-8 -*-
import sys
import argparse

try:
    from OtterPlotter import Plotter
except ImportError:
    from fake_plotter import FakePlotter as Plotter
from primitives import vis_drawing, resize_and_center, optimize
from geometry import remap
import numpy as np
import matplotlib.pyplot as plt


def parse_arguments():
    parser = argparse.ArgumentParser(description='',
				     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    return parser.parse_args()

def run(args):
    N_per_layer = 8
    N_per_blob = 21
    N_layers = 12
    r_min_layer = 5.5
    r_layer_step = 1.6
    N_per_circle = 30
    r_circle_min = 0.05
    r_circle_max = 1.1

    paths = []
    for i_layer in range(N_layers):
        r_layer = r_min_layer + i_layer * r_layer_step
        center_angles = np.linspace(0, 2*np.pi,
                                    N_per_blob * N_per_layer,
                                    endpoint=False)
        # center_angles -= np.exp(remap(i_layer, 0, N_layers-1,
        #                               0, 0.6))
        # center_angles -= np.exp(i_layer*1.618 / N_layers)
        center_angles -= remap(i_layer, 0, N_layers-1,
                               0, np.radians(120))
        # center_angles += 1.618033 * i_layer
        for i_blob in range(N_per_layer):
            for i_circle in range(N_per_blob):
                center_x = r_layer * np.cos(center_angles[i_blob*N_per_blob + i_circle])
                center_y = r_layer * np.sin(center_angles[i_blob*N_per_blob + i_circle])
                # r_circle = remap(abs(i_circle - N_per_blob // 2),
                #                  0, N_per_blob // 2,
                #                  r_circle_max, r_circle_min)
                r_circle = np.sin(remap(i_circle, 0, N_per_blob,
                                        0, np.pi)) * r_circle_max + r_circle_min

                angle_start = np.random.rand() * 2*np.pi
                angle_end = angle_start + 2*np.pi
                angles = np.linspace(angle_start, angle_end, N_per_circle)
                sins = np.sin(angles)
                cosins = np.cos(angles)

                points = np.zeros((N_per_circle, 2))
                points[:, 0] = cosins * r_circle + center_x
                points[:, 1] = sins * r_circle + center_y
                paths.append(points)

    # plt.axis('equal')
    # vis_drawing(paths, 'k-', linewidth=0.1)
    # plt.gca().invert_yaxis()
    # plt.show()

    x_margin_mm = 10
    y_margin_mm = 10
    H = 210 # A4
    W = 297 # A4

    to_draw = resize_and_center(paths, H, W,
                                x_margin_mm, x_margin_mm,
                                y_margin_mm, y_margin_mm)
    vis_drawing(to_draw, 'r-', linewidth=0.1)
    to_draw = optimize(to_draw,
                       line_simplification_threshold=0.1)
    vis_drawing(to_draw, 'k-', linewidth=0.1)
    plt.plot([0, W, W, 0, 0], [0, 0, H, H, 0], 'k:')
    plt.axis('equal')
    plt.gca().invert_yaxis()
    plt.show()

    H = 210 # A4
    W = 297 # A4

    # baud = 115200
    baud = 9600
    with Plotter('/dev/ttyUSB0', baud) as p:
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
