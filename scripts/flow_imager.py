# -*- coding: utf-8 -*-
import os
import sys
import argparse
import tqdm

import numpy as np
import cv2
from opensimplex import OpenSimplex
from skimage.draw import line as sk_line
from OtterPlotter import Plotter
from primitives import vis_drawing, resize_and_center, optimize

import matplotlib.pyplot as plt


def parse_arguments():
    parser = argparse.ArgumentParser(description='',
				     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--img', help='path to image', required=True)
    
    return parser.parse_args()

def run(args):
    img = cv2.imread(args.img)
    img = cv2.resize(img, (800, 600), cv2.INTER_AREA)
    ## create a flow field
    H, W = img.shape[:2]

    x_noise = OpenSimplex(240)
    y_noise = OpenSimplex(32)
    field = np.zeros((H, W, 2))
    for y in range(H):
        for x in range(W):
            mult = 0.0015
            x_val = x_noise.noise2d(x=mult*x, y=mult*y)
            y_val = y_noise.noise2d(x=mult*x, y=mult*y)
            field[y, x, :] = (x_val, y_val)

    # draw_field(field, N_particles=2000)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64) / 256
    sigma = 3
    gray = cv2.GaussianBlur(gray, ksize=(0, 0), sigmaX=sigma)
    # cv2.imshow("cv: gray", gray)
    # c = cv2.waitKey(0)
    # if c == ord('q'):
    #     import sys
    #     sys.exit(1)
    points = draw_field(field, guide=1-gray, N_particles=10000)
            
    # ## draw the image with the flow field

    # cv2.namedWindow('cv: img', cv2.WINDOW_NORMAL)
    # cv2.imshow("cv: img", img)
    # # cv2.imshow("cv: field", np.linalg.norm(field, axis=2))
    # cv2.imshow("cv: field", field[..., 0])
    # cv2.resizeWindow('cv: img', 800, 600)
    # c = cv2.waitKey(0)
    # if c == ord('q'):
    #     import sys
    #     sys.exit(1)

    x_margin_mm = 10
    y_margin_mm = 10
    H = 210 # A4
    W = 297 # A4

    to_draw = resize_and_center(points, H, W,
                                x_margin_mm, x_margin_mm,
                                y_margin_mm, y_margin_mm)
    vis_drawing(to_draw, 'r-', linewidth=0.5)
    to_draw = optimize(to_draw,
                       line_simplification_threshold=0.1)
    vis_drawing(to_draw, 'k-', linewidth=0.5)
    plt.plot([0, W, W, 0, 0], [0, 0, H, H, 0], 'k:')
    plt.axis('equal')
    plt.gca().invert_yaxis()
    plt.show()

    H = 210 # A4
    W = 297 # A4

    with Plotter('/dev/ttyUSB0', 115200) as p:
        p.load_config('config.json')
        p.set_input_limits((0, 0), (W, 0),
                           (0, H), (W, H))
        p.draw_polylines(to_draw)
    return 0

def draw_field(field, guide=None, N_particles=50000, vis=False):
    ''' draw vector field

    args:
    field - (H, W, 2) np array
    guide - (H, W) np array image '''
    lines = []

    if guide is None:
        guide_counter = np.ones(field.shape[:2])
    else:
        guide_counter = guide.copy()

    def sample_field_point(probs, n=1):
        N_points = probs.size
        lin_coords = np.random.choice(N_points, size=n, replace=False, p=probs.flatten())
        coords = np.unravel_index(lin_coords, probs.shape)
        coords = np.array(coords).T
        if n==1:
            coords = coords[0]
        return coords

    guide_scale = 1
    guide_counter = cv2.resize(guide_counter, dsize=(0, 0), fx=guide_scale, fy=guide_scale)
    probs = guide_counter.copy() ** 1.4
    probs /= np.sum(probs)
    max_prob = np.amax(probs)
    orig_guide = guide_counter.copy()

    rot_noise = OpenSimplex(117)
    rot_noise_mult = 0.555


    start_pts = []

    H, W = field.shape[:2]
    for part_i in tqdm.tqdm(range(N_particles)):
        if np.random.rand() > 0.5:
            sign = -1
        else:
            sign = 1

        line = []
        pos = sample_field_point(probs)
        prob = probs[pos[0], pos[1]]
        # pos /= guide_scale
        pos = pos[::-1].astype(np.float64) # convert to (x, y)
        start_pts.append(pos.copy())

        # perturbation_angle = np.random.randn()
        # perturbation_angle /= 3
        perturbation_angle = rot_noise.noise2d(x=rot_noise_mult*pos[0], y=rot_noise_mult*pos[1])
        perturbation_angle /= 1.6
        c, s = np.cos(perturbation_angle), np.sin(perturbation_angle)
        perturbation_rot = np.array([[c, -s], [s, c]])

        line.append(pos.copy())
        particle_paint_left = ((max_prob - prob) / max_prob)**0.5 * 30 * guide_scale
        # print(f"particle_paint_left: {particle_paint_left}")
        stop_line = False
        vals = []
        while particle_paint_left > 0:
            field_pos = np.round(pos)
            force = field[int(field_pos[1]),
                          int(field_pos[0]), :]
            old_pos = pos.copy()
            # if np.linalg.norm(force) < 0.0001:
            #     force += 1
            # to_add = sign * force
            to_add = sign * force / np.linalg.norm(force)
            to_add = np.matmul(perturbation_rot, to_add.T).T
            pos += to_add
            # pos = np.round(pos)
            if np.logical_or(np.any(pos < 0),
                            np.any(pos >= (W, H))):
                break

            segment = bresenham(old_pos, np.round(pos))
            prev_xy = None
            for xy in segment:
                xy = np.floor(guide_scale*np.array(xy)).astype(np.int32)
                if np.all(xy == prev_xy):
                    continue
                prev_xy = xy.copy()
                try:
                    val = orig_guide[xy[1], xy[0]].copy()
                    vals.append(val)
                    if np.abs(vals[-1] - vals[0]) > (30.0 / 256):
                        stop_line = True
                        break
                    particle_paint_left -= 1
                    if particle_paint_left <= 0:
                        stop_line = True
                        break
                except IndexError:
                    stop_line = True
                    break
            if np.all(force == 0):
                # print('noforce')
                break

            line.append(pos.copy())

            if stop_line:
                # print('stop_line')
                break


        line = np.array(line)
        if len(line) >= 2:
            lines.append(line)
        if vis:
            plt.plot(line[:, 0], line[:, 1], 'k-', lw=0.1, alpha=0.5)

    if vis:
        plt.axis('image')
        plt.gca().invert_yaxis()
        # plt.figure()
        # start_pts = np.array(start_pts)
        # plt.plot(start_pts[:, 0], start_pts[:, 1], 'k.', markersize=0.3)
        # plt.axis('image')
        # plt.gca().invert_yaxis()
        # plt.imshow(guide_counter)
        # plt.colorbar()
        plt.show()
    return lines

def bresenham(start, end):
    # being start and end two points (x1,y1), (x2,y2)
    a = np.int32(start)
    b = np.int32(end)
    discrete_line = list(zip(*sk_line(*a, *b)))
    return discrete_line


def main():
    args = parse_arguments()
    return run(args)

if __name__ == '__main__':
    sys.exit(main())
