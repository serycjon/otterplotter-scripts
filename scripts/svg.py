# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import

import os
import sys
import argparse
import tqdm
import copy

import svgpathtools as svgpt
import numpy as np
import matplotlib.pyplot as plt

from OtterPlotter import Plotter
from primitives import vis_drawing, resize_and_center, rotate
from primitives import optimize, load_page_conf
from primitives import rounded_rect, drawing_bbox, mask_drawing
from various_utils import with_debugger

import logging
logger = logging.getLogger(__name__)


def parse_arguments():
    parser = argparse.ArgumentParser(description='',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--path', help='svg path', required=True)
    parser.add_argument('--simplify', help='line simplification threshold [mm]', type=float, default=0.0)
    parser.add_argument('--join', help='line joining threshold [mm]', type=float, default=0.0)
    parser.add_argument('--drop', help='line dropping threshold [mm]', type=float, default=0.0)
    parser.add_argument('--format', help='path to a format json', required=True)
    parser.add_argument('--margins', help='page margins [mm]', type=float, default=10.0)
    parser.add_argument('--rotate', help='rotate drawing 90 deg', action='store_true')
    parser.add_argument('--multipass', help='number of passes', default=1, type=int)
    parser.add_argument('--noopt', help='disable optimization', action='store_true')
    parser.add_argument('--opconfig', help='OtterPlotter calibration path', default='config.json')
    parser.add_argument('--border_crop', help='', action='store_true')
    parser.add_argument('--border_round', help='border rounding radius', default=15, type=int)

    return parser.parse_args()


def main():
    args = parse_arguments()
    return run(args)


def cplx_to_xy(cplx):
    return (cplx.real, cplx.imag)


def xy_to_cplx(xy):
    return (xy[0] + xy[1]*1j)


def export_svg(paths, filename):
    svg_paths = []
    view_box = drawing_bbox(paths)
    for path in paths:
        svg_path = []
        for i in range(len(path)):
            pt = path[i]
            svg_path.append('{},{}'.format(pt[0], pt[1]))
        svg_path = ' '.join(svg_path)
        svg_paths.append('<polyline points="{}" style="fill:none;stroke:black;" />'.format(svg_path))
    header = '<?xml version="1.0" encoding="UTF-8" standalone="no"?>\n'
    document_params = 'version="1.1"\nxmlns="http://www.w3.org/2000/svg"'
    document_params += f'\nviewBox="{view_box[0]} {view_box[1]} {view_box[2]} {view_box[3]}"'
    document = '<svg {}>\n{}\n</svg>'.format(document_params,
                                             '\n'.join(svg_paths))
    document = header + document
    with open(filename, 'w') as fout:
        fout.write(document)


def svg_as_string(paths):
    svg_paths = []
    view_box = drawing_bbox(paths)
    for path in paths:
        svg_path = []
        for i in range(len(path)):
            pt = path[i]
            svg_path.append('{},{}'.format(pt[0], pt[1]))
        svg_path = ' '.join(svg_path)
        svg_paths.append('<polyline points="{}" style="fill:none;stroke:black;" />'.format(svg_path))
    # header = '<?xml version="1.0" encoding="UTF-8" standalone="no"?>\n'
    document_params = 'version="1.1"\nxmlns="http://www.w3.org/2000/svg"'
    document_params += f'\nviewBox="{view_box[0]} {view_box[1]} {view_box[2]} {view_box[3]}"'
    document = '<svg {}>\n{}\n</svg>'.format(document_params,
                                             '\n'.join(svg_paths))
    # document = header + document
    return document


@with_debugger
def run(args):
    svg_path = args.path
    parsed_paths, attributes, svg_attributes = svgpt.svg2paths(svg_path, return_svg_attributes=True)
    logger.info('SVG loaded')

    paths = []
    for path in parsed_paths:
        if path.iscontinuous():
            paths.append(path)
        else:
            for p in path.continuous_subpaths():
                paths.append(p)

    np_paths = []
    step = 1
    for path_i, path in enumerate(tqdm.tqdm(paths, desc='paths')):
        np_path = [cplx_to_xy(path.start)]
        for part in path:
            if type(part) is svgpt.path.Line:
                start = cplx_to_xy(part.start)
                end = cplx_to_xy(part.end)
                if start != np_path[-1]:
                    np_path.append(start)
                np_path.append(end)
            else:
                length = part.length()
                steps = int(np.round(length / step))
                # steps = 20
                if steps == 0:
                    continue
                fraction_step = 1 / steps

                for i in range(steps + 1):
                    try:
                        pt = path.point(fraction_step * i)
                        pt = cplx_to_xy(pt)
                        np_path.append(pt)
                    except Exception:
                        pass

        np_paths.append(np.array(np_path))
    logger.info('SVG converted')

    to_draw = np_paths
    if args.rotate:
        to_draw = rotate(to_draw)

    page_conf = load_page_conf(args.format)

    x_margin_mm = args.margins
    y_margin_mm = args.margins
    H = page_conf['H']  # 210 # A4
    W = page_conf['W']  # 297 # A4

    to_draw = resize_and_center(to_draw, H, W,
                                x_margin_mm, x_margin_mm,
                                y_margin_mm, y_margin_mm)
    orig = copy.deepcopy(to_draw)

    if args.border_crop:
        border = rounded_rect(drawing_bbox(to_draw), args.border_round)
        vis_drawing([border], 'b-', linewidth=0.5)

        logger.info("Masking drawing")
        to_draw = mask_drawing(to_draw, border)

    to_draw = multi_pass(to_draw, args.multipass)

    if not args.noopt:
        logger.info("Starting optimization")
        to_draw = optimize(to_draw,
                           path_join_threshold=args.join,
                           line_simplification_threshold=args.simplify,
                           path_drop_threshold=args.drop)

    logger.info('plotting')
    vis_drawing(to_draw, 'r-', linewidth=0.5)
    vis_drawing(orig, 'k-', linewidth=0.5)
    plt.plot([0, W, W, 0, 0], [0, 0, H, H, 0], 'k:')
    plt.axis('equal')
    # plt.gca().invert_yaxis()
    plt.show()

    with Plotter('/dev/ttyUSB0', 9600) as p:
        p.load_config(args.opconfig)
        p.set_input_limits((0, 0), (W, 0),
                           (0, H), (W, H))
        p.set_speed(289)
        p.draw_polylines(to_draw)

    return 0


def multi_pass(paths, count):
    results = []
    for i in range(count):
        for path in paths:
            results.append(path.copy())
    return results


if __name__ == '__main__':
    format = "[%(asctime)s] %(levelname)s:%(name)s:%(message)s"
    logging.basicConfig(level=logging.INFO, format=format)
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    # import ipdb
    try:
        sys.exit(main())
    except Exception as e:
        print(e)
        # ipdb.post_mortem()

