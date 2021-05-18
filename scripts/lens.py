# -*- coding: utf-8 -*-
import numpy as np
import sys
import argparse
import vpype_viewer
from vpype_integration import to_vpype_document, from_vpype_document
from primitives import square_grid, subsample_drawing, apply_per_layer, circle, mask_drawing
from primitives import rounded_rect, place_on_grid
from geometry import remap
import vpype_cli
from various_utils import with_debugger
from repro import ReproSaver
from collections import defaultdict


def parse_arguments():
    parser = argparse.ArgumentParser(description='',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--nosave', help='do not save results', action='store_true')
    parser.add_argument('--novis', help='', action='store_true')
    return parser.parse_args()


@with_debugger
def circle_lens(drawing, center, radius, strength=1, something=0.3):
    sub_drawing = subsample_drawing(drawing)
    lens_circle = circle(center, radius)
    thick_lens_circle = [lens_circle] + [circle(center, radius + 1)] + [circle(center, radius - 1)]

    def lense_path(path):
        np_center = np.array(center).reshape(1, 2)
        from_center = path - np_center
        dist_from_center = np.linalg.norm(from_center, axis=1)
        from_center[dist_from_center > 0, :] /= dist_from_center[dist_from_center > 0, np.newaxis]
        d = dist_from_center[:, np.newaxis]
        add = from_center * np.arctan(d / (radius * something)) * radius * strength
        return path + add

    def lensing_fn(layer):
        out = []
        for path in layer:
            out.append(lense_path(path))
        return out

    lensed = apply_per_layer(sub_drawing, lensing_fn)
    lensed = mask_drawing(lensed, lens_circle)
    lensed = lensed + lensed  # make the inside multipass
    background = mask_drawing(drawing, lens_circle, invert=True)
    return {'bg': background,
            'lensed': lensed,
            'lens': thick_lens_circle}


@with_debugger
def run(args):
    saver = ReproSaver()
    saver.seed()

    H, W = 900, 900
    side = 10
    sub_H, sub_W = 60, 60
    subgrid = square_grid(sub_H, sub_W, side)
    margin = 20
    main_rows, main_cols = 4, 4
    main_grid = place_on_grid({'black': subgrid}, (0 + margin, 0 + margin), (W - margin, H - margin),
                              main_rows, main_cols, margin)

    boundary = rounded_rect((0, 0, W, H), r=0, N_seg=1)
    layers = defaultdict(list)
    layers = main_grid
    # layers = {k: circle_lens(v, (400, 400), 200) for k, v in layers.items()}
    # layers['black'] = circle_lens(layers['black'], (400, 400), 200,
    #                               strength=0.2)
    # layers['black'] = circle_lens(layers['black'], (700, 600), 100,
    #                               strength=0.8)

    lenses = []
    for i in range(16):
        while True:
            radius = np.random.randint(H // 20, H // 3)
            center_x = np.random.randint(-radius, W + radius)
            center_y = np.random.randint(-radius, H + radius)
            center = (center_x, center_y)
            strength = remap(np.random.rand(),
                             0, 1,
                             0.2, 0.4)
            overlapping = False
            for l_center, l_radius, _ in lenses:
                c_dist = np.linalg.norm(np.array(center) - np.array(l_center))
                if c_dist < l_radius + radius:
                    overlapping = True
            if not overlapping:
                lenses.append((center, radius, strength))
                break

    for center, radius, strength in lenses:
        res = circle_lens(layers['black'], center, radius,
                          strength=strength, something=strength)
        layers['black'] = res['bg']
        layers['red'].extend(res['lens'])
        layers['green'].extend(res['lensed'])
    layers['black'] = mask_drawing(layers['black'], boundary)
    layers['blue'] = [boundary]
    layers['red'] = mask_drawing(layers['red'], boundary)
    layers['green'] = mask_drawing(layers['green'], boundary)
    document = to_vpype_document(layers)
    document = vpype_cli.execute("linesimplify linemerge linesort", document)
    layers = from_vpype_document(document)
    if not args.novis:
        vpype_viewer.show(document, view_mode=vpype_viewer.ViewMode.OUTLINE)
    if not args.nosave:
        saver.add_svg(layers)
    return 0


def main():
    args = parse_arguments()
    return run(args)


if __name__ == '__main__':
    sys.exit(main())
