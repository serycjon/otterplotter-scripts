# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import

import cv2
import numpy as np
import scipy.interpolate
import sys
import argparse

from plotter import p2e, e2p
from primitives import vis_drawing, resize_and_center
from primitives import optimize, drawing_stats

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def parse_arguments():
    parser = argparse.ArgumentParser(description='',
				     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    return parser.parse_args()

def spiral_points(n_coils, radius, point_distance):
    ''' https://stackoverflow.com/a/13901170/1705970 '''
    max_angle = n_coils * 2 * np.pi

    radius_step = radius / max_angle

    pts = [(0, 0)]

    angle = point_distance / radius_step
    while angle <= max_angle:
        r = radius_step * angle
        x = np.cos(angle) * r
        y = np.sin(angle) * r
        pts.append((x, y))

        angle += point_distance / r
    return np.array(pts)

def run(args):
    img = cv2.imread('elevation/crop2.TIF', flags=(cv2.IMREAD_GRAYSCALE | cv2.IMREAD_ANYDEPTH))
    img = np.float64(img)
    elevation = img.copy()
    amin = np.amin(img)
    amax = np.amax(img)
    img = (img - amin) / (amax-amin)
    img = cv2.resize(img, None, fx=0.25, fy=0.25)
    # cv2.imshow("cv: img", img)
    # c = cv2.waitKey(0)

    H, W = elevation.shape[:2]
    # elevation = np.zeros((H, W), dtype=np.float64)
    # elevation[8:12, 8:12] = -1
    x = np.arange(W)
    y = np.arange(H)
    xs, ys = np.meshgrid(x, y)
    zs = elevation[ys, xs]
    zs -= amin
    zs /= amax-amin
    zs *= 600

    center = (W/2, H/2)
    point_distance = 5
    coil_distance = 20
    radius = H/2
    n_coils = 2*radius / coil_distance
    spiral = spiral_points(n_coils=int(n_coils),
                           radius=radius,
                           point_distance=point_distance)
    spiral += center
    interp = scipy.interpolate.RectBivariateSpline(y, x, zs)
    spiral_z = interp(spiral[:, 1], spiral[:, 0], grid=False)

    points = np.vstack((spiral[:, 0] - center[0],
                        spiral[:, 1] - center[1],
                        spiral_z))

    f = 1
    K = np.array([[f, 0, 0],
                  [0, f, 0],
                  [0, 0, 1]])
    R = np.array([[1, 0, 0],
                  [0, 1, 0],
                  [0, 0, 1]])
    R = R_x(-45)
    C = np.array([0, 0, 2000]).reshape(3, 1)
    t = -np.dot(R, C)
    P = np.dot(K, np.concatenate((R, t), axis=1))

    projected = p2e(np.dot(P, e2p(points)))

    x_margin_mm = 10
    y_margin_mm = 10
    H = 210 # A4
    W = 297 # A4

    print('projected.shape: {}'.format(projected.shape))
    to_draw = [projected.T]

    to_draw = resize_and_center(to_draw, H, W,
                                x_margin_mm, x_margin_mm,
                                y_margin_mm, y_margin_mm)
    to_draw = optimize(to_draw)

    # plt.figure()
    vis_drawing(to_draw, 'k-', linewidth=0.1)
    plt.gca().invert_yaxis()

    # plt.scatter(spiral[:, 0], spiral[:, 1],
    #             s=1, marker='.', c=spiral_z)
    plt.axis('equal')

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(spiral[:, 1],
    #            spiral[:, 0],
    #            spiral_z, s=1, marker='.')
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    # ax.set_zlabel('z')
    plt.show()

    return 0

def R_x(angle):
    s = np.sin(np.radians(angle))
    c = np.cos(np.radians(angle))
    return np.array([[1, 0, 0],
                     [0, c, -s],
                     [0, s, c]])
def R_y(angle):
    s = np.sin(np.radians(angle))
    c = np.cos(np.radians(angle))
    return np.array([[c, 0, s],
                     [0, 1, 0],
                     [-s, 0, c]])
def R_z(angle):
    s = np.sin(np.radians(angle))
    c = np.cos(np.radians(angle))
    return np.array([[c, -s, 0],
                     [s, c, 0],
                     [0, 0, 1]])

def main():
    args = parse_arguments()
    return run(args)

if __name__ == '__main__':
    sys.exit(main())
