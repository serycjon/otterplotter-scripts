# -*- coding: utf-8 -*-
import numpy as np
import sys
import argparse
import vpype_viewer
from vpype_integration import to_vpype_document
from primitives import grid, subsample_drawing, apply_per_layer, circle, mask_drawing
from primitives import rounded_rect
from geometry import remap
import vpype_cli
from various_utils import with_debugger


def parse_arguments():
    parser = argparse.ArgumentParser(description='',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    return parser.parse_args()


@with_debugger
def circle_lens(drawing, center, radius, strength=1, something=0.3):
    sub_drawing = subsample_drawing(drawing)
    lens_circle = circle(center, radius)

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
    background = mask_drawing(drawing, lens_circle, invert=True)
    return background + lensed + [lens_circle]


def run(args):
    H, W = 900, 900
    margin = 30
    stripe_n = 4
    gap = 7
    stripe_w = (W / stripe_n) - 2 * margin - gap
    density = stripe_w / 30
    grids = []
    for i in range(stripe_n):
        grids += grid((margin + i * (stripe_w + 2 * margin + gap), margin),
                      (margin + i * (stripe_w + 2 * margin + gap) + stripe_w, H - margin),
                      (density, density))

    boundary = rounded_rect((0, 0, W, H), r=0, N_seg=1)
    layers = {'black': grids}
    # layers = {k: circle_lens(v, (400, 400), 200) for k, v in layers.items()}
    # layers['black'] = circle_lens(layers['black'], (400, 400), 200,
    #                               strength=0.2)
    # layers['black'] = circle_lens(layers['black'], (700, 600), 100,
    #                               strength=0.8)

    lenses = []
    for i in range(13):
        while True:
            center_x = np.random.randint(0, W)
            center_y = np.random.randint(0, H)
            center = (center_x, center_y)
            radius = np.random.randint(H // 20, H // 5)
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
        layers['black'] = circle_lens(layers['black'], center, radius,
                                      strength=strength, something=strength)
    layers['black'] = mask_drawing(layers['black'], boundary) + [boundary]
    document = to_vpype_document(layers)
    document = vpype_cli.execute("linemerge linesort", document)
    vpype_viewer.show(document, view_mode=vpype_viewer.ViewMode.OUTLINE)
    return 0


def main():
    args = parse_arguments()
    return run(args)


if __name__ == '__main__':
    sys.exit(main())
