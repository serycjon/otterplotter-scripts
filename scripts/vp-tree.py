# -*- coding: utf-8 -*-
import sys
import argparse
import numpy as np
from collections import deque

import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from shapely.geometry import Polygon, box
from primitives import circle
from primitives import vis_drawing
from repro import ReproSaver


def parse_arguments():
    parser = argparse.ArgumentParser(description='',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--nosave', help='', action='store_true')

    return parser.parse_args()


def run(args):
    saver = ReproSaver()
    saver.seed()
    points = np.random.rand(1000, 2) * 800

    def dist_fn(vp, db):
        return np.sqrt(np.sum(np.square(db - vp.reshape(1, 2)), axis=1))

    vp_tree = VPTree(points, dist_fn)
    # plt.scatter(points[:, 0], points[:, 1])
    paths = vp_tree.plot(color='k', lw=0.5)
    if not args.nosave:
        saver.add_svg(paths)
    vis_drawing(paths, 'k')
    plt.axis('equal')
    plt.show()
    return 0


def main():
    args = parse_arguments()
    return run(args)


class VPTree:
    def __init__(self, points, distance_fn, restricted_to=None):
        self.dist = distance_fn
        self.restricted_to = restricted_to
        if restricted_to is None:
            # construct bbox from points
            min_x, min_y = np.amin(points, axis=0)
            max_x, max_y = np.amax(points, axis=0)
            self.restricted_to = box(min_x, min_y, max_x, max_y)

        if points.shape[0] <= 1:
            self.inside_subtree = None
            self.outside_subtree = None
            self.vantage_point = None
            self.radius = 0
            self.leaf = True
        else:
            self.leaf = False
            self.vantage_point = points[np.random.choice(len(points)), :]
            dists = distance_fn(self.vantage_point, points)
            self.radius = np.median(dists)

            inside_data = points[dists < self.radius, :]
            outside_data = points[dists >= self.radius, :]

            circle_poly = Polygon(circle(self.vantage_point, self.radius))

            self.inside_subtree = VPTree(inside_data, distance_fn, self.restricted_to.intersection(circle_poly))
            self.outside_subtree = VPTree(outside_data, distance_fn, self.restricted_to.difference(circle_poly))

    def plot(self, depth=0, **plot_kwargs):
        # if depth > 2:
        #     return
        result = []
        if not self.leaf:
            # plt.scatter(self.vantage_point[0], self.vantage_point[1])
            # plt.gca().add_artist(Circle(self.vantage_point, radius=self.radius, fill=False))
            if depth > 0:
                if self.restricted_to.type == 'MultiPolygon':
                    for geom in self.restricted_to.geoms:
                        result.append(np.array(geom.exterior.coords))
                else:
                    result.append(np.array(self.restricted_to.exterior.coords))

            result.extend(self.inside_subtree.plot(depth + 1, **plot_kwargs))
            result.extend(self.outside_subtree.plot(depth + 1, **plot_kwargs))

        return result


if __name__ == '__main__':
    sys.exit(main())
