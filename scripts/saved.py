# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import

import sys
import argparse

from OtterPlotter import Plotter
from primitives import vis_drawing, load_drawing, resize_and_center, optimize
import matplotlib.pyplot as plt

from getkey import getkey, keys


def parse_arguments():
    parser = argparse.ArgumentParser(description='',
				     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('path', help='drawing path')
    parser.add_argument('--simplify', help='line simplification threshold', type=float, default=0.0)
    parser.add_argument('--no_opt', help='disable optimization', action='store_true')
    
    return parser.parse_args()

def run(args):
    drawing = load_drawing(args.path)

    x_margin_mm = 10
    y_margin_mm = 10
    H = 210 # A4
    W = 297 # A4

    to_draw = resize_and_center(drawing, H, W,
                                x_margin_mm, x_margin_mm,
                                y_margin_mm, y_margin_mm)
    if type(to_draw) is not dict:
        # single layer
        to_draw = {'single_layer': to_draw}

    if not args.no_opt:
        to_draw = {layer_name: optimize(layer,
                                        line_simplification_threshold=args.simplify)
                for layer_name, layer in to_draw.items()}
        
    if 'black' in to_draw and 'red' in to_draw:
        vis_drawing(to_draw['black'], 'k-', lw=0.1)
        vis_drawing(to_draw['red'], 'r-', lw=0.3)
        plt.plot([0, W, W, 0, 0], [0, 0, H, H, 0], 'k:')
        plt.axis('equal')
        plt.gca().invert_yaxis()
        plt.show()

    all_keys = set([str(i) for i in range(10)] + [chr(c) for c in range(97, 123)])
    reserved_keys = set(['?', 'q'])

    layer_keys = sorted(list(all_keys - reserved_keys))
    key_layer_mapping = {layer_keys[i]: layer_name for i, layer_name in enumerate(to_draw.keys())}

    def help():
        print('? - this help')
        print('q - quit')
        for k in sorted(key_layer_mapping.keys()):
            print('{} - plot "{}" layer'.format(k, key_layer_mapping[k]))

    help()
    while True:
        key = getkey()
        print(key)
        if key == 'q':
            sys.exit(0)
        elif key == '?':
            help()
        elif key in key_layer_mapping:
            layer_name = key_layer_mapping[key]
            print('printing "{}"'.format(layer_name))
            layer = to_draw[layer_name]
            with Plotter('/dev/ttyUSB0', 115200) as p:
                p.load_config('config.json')
                p.set_input_limits((0, 0), (W, 0),
                                   (0, H), (W, H))
                p.draw_polylines(layer)
            print('LAYER FINISHED')
            help()

    return 0

def main():
    args = parse_arguments()
    return run(args)

if __name__ == '__main__':
    sys.exit(main())
