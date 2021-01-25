# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import

import sys
import argparse

from OtterPlotter import Plotter
import numpy as np
import matplotlib.pyplot as plt

def parse_arguments():
    parser = argparse.ArgumentParser(description='',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    return parser.parse_args()

def point_line_distance(x0,y0,x1,y1,x2,y2):
    return abs((x2-x1)*(y1-y0)-(x1-x0)*(y2-y1)) / np.sqrt((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1))

def resize_and_center(points, H, W,
                      l_x_margin, r_x_margin,
                      t_y_margin, b_y_margin):
    """ points (N, 2) """
    src_min_x, src_min_y = np.amin(points, axis=0)
    src_max_x, src_max_y = np.amax(points, axis=0)
    src_H = src_max_y - src_min_y
    src_W = src_max_x - src_min_x

    points = points - np.array((src_min_x, src_min_y))

    dst_min_x = l_x_margin
    dst_max_x = W - r_x_margin
    dst_min_y = t_y_margin
    dst_max_y = H - b_y_margin

    dst_H = dst_max_y - dst_min_y
    dst_W = dst_max_x - dst_min_x

    scale = dst_H / src_H
    # take the fitting scale
    if src_W * scale > dst_W:
        scale = dst_W / src_W

    shift_x = (W / 2) - (src_W / 2) * scale
    shift_y = (H / 2) - (src_H / 2) * scale

    shift = np.array((shift_x, shift_y))

    points = points * scale + shift
    return points


def simplify_reumann_witkam(tolerance,p,step=-1):
    mask = np.ones(len(p),dtype='bool')
    first = 0
    second = 1
    third = 2

    marker = np.array([p[first],p[second],p[third]],dtype='double')

    if step == -1:
        maxstep = len(p)
    else:
        maxstep = min(step,len(p))

    for i in range(0,min(maxstep,len(p)-2)):

        dist = point_line_distance(p[third,0],p[third,1],p[first,0],p[first,1],p[second,0],p[second,1])
        #print dist
        if dist <= tolerance:
            mask[third] = False
            third = third+1
        else:
            first = second
            second = third
            third = third+1
        marker = np.array([p[first],p[second]],dtype='double')

    return p[mask]
    # return mask,marker

def isometric(xs, s1=-1, s2=1):
    """ xs: (3, N) points """
    alpha = s1*np.arcsin(np.tan(np.radians(30)))
    sa, ca = np.sin(alpha), np.cos(alpha)
    beta = s2*np.radians(45)
    sb, cb = np.sin(beta), np.cos(beta)
    isometric = np.dot(
        np.array(
            [[1, 0, 0],
             [0, ca, sa],
             [0, -sa, ca]]),
        np.array(
            [[cb, 0, -sb],
             [0, 1, 0],
             [sb, 0, cb]]))

    iso_proj = np.dot(isometric, xs)[:2, :]
    return iso_proj

def run(args):
    # att = Clifford(-1.7, 1.3, -0.1, -1.21)
    # att = DequanLi(40, 1.833, 0.16, 0.65, 55, 20,
    #                x=0.349,
    #                y=0.0,
    #                z=-0.16,
    #                dt=0.0001)

    att = halvorsen(1.4, x=1, y=0, z=0, dt=0.0049)
    points = []
    burn_in = 1000
    [next(att)  for _ in range(burn_in)]

    points = [next(att) for _ in range(750000)]
    points = np.array(points)
    # from mpl_toolkits.mplot3d import Axes3D
    # ax = plt.axes(projection="3d")
    # ax.plot3D(points[:, 0],
    #           points[:, 1],
    #           points[:, 2],
    #           lw=0.01)
    # plt.figure()

    points = isometric(points.T).T

    y_margin_mm = 30
    x_margin_mm = 30
    H = 210 # A4
    W = 297 # A4

    points = resize_and_center(points, H, W,
                               x_margin_mm, x_margin_mm,
                               y_margin_mm, y_margin_mm)

    N_pre_simplification = points.shape[0]
    # points = simplify_reumann_witkam(0.07, points)
    N_post_simplification = points.shape[0]
    print('line simplified from {} to {} points ({}%)'.format(
        N_pre_simplification,
        N_post_simplification,
        int(np.round(100 * N_post_simplification / N_pre_simplification))))

    plt.plot(points[:, 0],
             points[:, 1],
             '-',
             linewidth=0.02)
    plt.plot(points[0:500, 0],
             points[0:500, 1],
             'r-')
    plt.plot(points[0, 0],
             points[0, 1],
             'rx')
    plt.axis('equal')
    plt.show()


    with Plotter('/dev/ttyUSB0', 9600) as p:
        p.load_config('config.json')
        p.set_input_limits((0, 0), (W, 0),
                           (0, H), (W, H))
        p.draw_polyline(points, n_decimal=1)

    return 0

def halvorsen(a, x=0, y=0, z=0, dt=0.0005):
    while True:
        yield np.array((x, y, z))

        dx = -a*x - 4*y - 4*z - y**2
        dy = -a*y - 4*z - 4*x - z**2
        dz = -a*z - 4*x - 4*y - x**2

        x += dx * dt
        y += dy * dt
        z += dz * dt

class DequanLi():
    def __init__(self, a, c, d, e, k, f, x=0, y=0, z=0, dt=0.1):
        self.a = a
        self.c = c
        self.d = d
        self.e = e
        self.k = k
        self.f = f

        self.x = x
        self.y = y
        self.z = z

        self.dt = dt

    def __iter__(self):
        return self

    def __next__(self):
        dx = self.a * (self.y - self.x) + self.d * self.x * self.z
        dy = self.k * self.x + self.f * self.y - self.x * self.z
        dz = self.c  * self.z + self.x * self.y - self.e * self.x**2

        self.x += dx * self.dt
        self.y += dy * self.dt
        self.z += dz * self.dt

        return np.array((self.y, self.z))

class Clifford():
    def __init__(self, a, b, c, d):
        self.a = a
        self.b = b
        self.c = c
        self.d = d

        self.x = 0.1
        self.y = 0.1

    def __iter__(self):
        return self

    def __next__(self):
        x = np.sin(self.a * self.y) + self.c*np.cos(self.a * self.x)
        y = np.sin(self.b * self.x) + self.d*np.cos(self.b * self.y)

        self.x = x
        self.y = y

        return np.array((x, y))

def main():
    args = parse_arguments()
    return run(args)

if __name__ == '__main__':
    sys.exit(main())
