# -*- coding: utf-8 -*-
import sys
import argparse

import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt
import skimage.measure

from OtterPlotter import Plotter
from primitives import vis_drawing, resize_and_center, rotate
from primitives import rounded_rect, drawing_bbox, mask_drawing
from primitives import load_page_conf, optimize
from img_tools import resize_to_px_count, bgr2cmyk

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)
logging.getLogger("matplotlib").setLevel(logging.WARNING)

import tqdm


def parse_arguments():
    parser = argparse.ArgumentParser(description='',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--img', help='image path', required=True)
    parser.add_argument('--hatch_step_px', help='', type=int, default=5)
    parser.add_argument('--N_hatches', help='', type=int, default=8)
    parser.add_argument('--format', help='path to a format json', required=True)
    parser.add_argument('--margins', help='page margins [mm]', type=float, default=10.0)
    parser.add_argument('--rotate', help='rotate drawing 90 deg', action='store_true')
    parser.add_argument('--opconfig', help='OtterPlotter calibration path', default='config.json')
    return parser.parse_args()


def hatch_at_value(gray_img, value, step_px=5,
                   simplify_tolerance=0, dark=True):
    if dark:
        mask = gray_img < value
    else:
        mask = gray_img >= value

    distance = distance_transform_edt(mask)
    d_max = np.amax(distance)
    distance_int = np.int32(np.round(distance))

    hatches = []
    cont_vis = np.zeros(mask.shape, dtype=np.uint8)
    for i in np.arange(1, d_max, step_px):
        try:
            contours = skimage.measure.find_contours(distance, i)
        except KeyError:
            continue
        if simplify_tolerance > 0:
            simplified = []
            for contour in contours:
                new = skimage.measure.approximate_polygon(contour, simplify_tolerance)
                if len(new) > 4:
                    simplified.append(new)
            contours = simplified

        hatches += contours
        # py_polygons = [poly[:, ::-1].tolist() for poly in polygons]
        cont_vis[distance_int == i] = 255
    return cont_vis, hatches


def multi_hatch(img, hatch_step_px, N_hatches=8, contour_tolerance=1, dark=True):
    levels = np.linspace(0, 255, N_hatches + 2)[1:-1]

    hatching = []
    for level_i, level in enumerate(tqdm.tqdm(levels)):
        _, hatching_lines = hatch_at_value(img, level,
                                           step_px=hatch_step_px,
                                           simplify_tolerance=contour_tolerance,
                                           dark=dark)
        hatching.extend(hatching_lines)

    return hatching


def run(args):
    img = cv2.imread(args.img)
    img = resize_to_px_count(img, 1000**2)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur_type = 'bilateral'
    # blur_type = 'gauss'
    if blur_type == 'gauss':
        sigma = 8
        if sigma > 0:
            gray = cv2.GaussianBlur(gray, ksize=(0, 0), sigmaX=sigma)
    elif blur_type == 'bilateral':
        sigmaColor = 20
        sigmaSpace = 5
        blurred = cv2.bilateralFilter(img, d=-1, sigmaColor=sigmaColor, sigmaSpace=sigmaSpace)

        # gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
        # sigma = 3
        # gray = cv2.GaussianBlur(gray, ksize=(0, 0), sigmaX=sigma)

        # cv2.imshow("cv: gray", gray)
        # while True:
        #     c = cv2.waitKey(0)
        #     if c == ord('q'):
        #         sys.exit(1)

    cmyk = bgr2cmyk(blurred)
    to_draw = {}
    N_hatches = args.N_hatches
    for i, name in enumerate("CMYK"):
        to_draw[name] = multi_hatch(cmyk[:, :, i], args.hatch_step_px, contour_tolerance=1, N_hatches=N_hatches, dark=False)
        to_draw[name] = [x[:, ::-1] for x in to_draw[name]]  # swap x, y

    if args.rotate:
        to_draw = rotate(to_draw)

    page_conf = load_page_conf(args.format)

    x_margin_mm = args.margins
    y_margin_mm = args.margins
    H = page_conf['H']  # 210 # A4
    W = page_conf['W']  # 297 # A4

    # to_draw = rotate(to_draw)
    to_draw = resize_and_center(to_draw, H, W,
                                x_margin_mm, x_margin_mm,
                                y_margin_mm, y_margin_mm)
    border = rounded_rect(drawing_bbox(to_draw), 15)
    vis_drawing([border], 'b:', linewidth=0.5)
    to_draw = mask_drawing(to_draw, border)
    to_draw = optimize(to_draw,
                       line_simplification_threshold=0.1,
                       path_drop_threshold=2.0,
                       path_join_threshold=1.0)
    vis_drawing(to_draw, layer_options={'C': (['c-'], dict(linewidth=0.1, alpha=0.5)),
                                        'M': (['m-'], dict(linewidth=0.1, alpha=0.5)),
                                        'Y': (['y-'], dict(linewidth=0.1, alpha=0.5)),
                                        'K': (['k-'], dict(linewidth=0.1, alpha=0.5)),
                                        })
    plt.plot([0, W, W, 0, 0], [0, 0, H, H, 0], 'k:')
    plt.show()

    with Plotter('/dev/ttyUSB0', 9600) as p:
        p.load_config(args.opconfig)
        p.set_input_limits((0, 0), (W, 0),
                           (0, H), (W, H))
        p.draw_polylines(to_draw)

    # cv2.imshow("cv: canvas", canvas)
    # while True:
    #     c = cv2.waitKey(0)
    #     if c == ord('q'):
    #         break
    # plt.contour(gray, levels=levels,
    #             colors='k', linewidths=0.1)
    # plt.axis('equal')
    # plt.gca().invert_yaxis()
    # plt.show()
    return 0


def main():
    args = parse_arguments()
    try:
        return run(args)
    except Exception as e:
        import traceback
        import ipdb
        print(traceback.format_exc())
        print(e)
        ipdb.post_mortem()


if __name__ == '__main__':
    sys.exit(main())
