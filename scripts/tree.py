# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import

import os
import pickle
import sys
import argparse
import numpy as np
import tqdm

from clifford import g2

from geometry import remap
from plotter import e2p, p2e

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def parse_arguments():
    parser = argparse.ArgumentParser(description='',
				     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    return parser.parse_args()

class SpaceTree():
    """ Modeling Trees with a Space Colonization Algorithm
Adam Runions, Brendan Lane, and Przemyslaw Prusinkiewicz
http://www.algorithmicbotany.org/papers/colonization.egwnp2007.large.pdf """
    def __init__(self, radius_of_influence=1, growth_D=0.2, kill_distance=0.01, N_attractors=1000,
                 flat_tree=False):
        self.radius_of_influence = radius_of_influence
        self.growth_D = growth_D
        self.kill_distance = kill_distance
        self.attraction_points = self.generate_attraction_points(N_attractors)
        if flat_tree:
            self.attraction_points = [(x[0], x[1], 0) for x in self.attraction_points]
        self.attraction_active = [True for x in self.attraction_points]
        self.nodes = [(0, 0.8, 0)]
        self.parent = [None]

        for step in tqdm.tqdm(range(5000)):
            if not np.any(self.attraction_active):
                break
            attracted_by = [[] for node in self.nodes]
            for attractor_i in range(len(self.attraction_points)):
                closest_node = self.get_closest_node(attractor_i, self.radius_of_influence)
                if closest_node is not None:
                    attracted_by[closest_node].append(attractor_i)

            new_nodes = []
            new_parents = []
            for node_i in range(len(self.nodes)):
                if len(attracted_by[node_i]) == 0:
                    continue

                node = self.nodes[node_i]
                growth_dir = np.array([0, 0, 0], dtype=np.float64)
                for att_i in attracted_by[node_i]:
                    att = self.attraction_points[att_i]
                    to_att = (att[0] - node[0],
                            att[1] - node[1],
                            att[2] - node[2])
                    to_att /= np.linalg.norm(to_att)
                    growth_dir += to_att
                growth_dir /= np.linalg.norm(growth_dir)

                new_node = (node[0] + self.growth_D * growth_dir[0],
                            node[1] + self.growth_D * growth_dir[1],
                            node[2] + self.growth_D * growth_dir[2])
                new_nodes.append(new_node)
                new_parents.append(node_i)

            self.nodes += new_nodes
            self.parent += new_parents

            self.deactivate_reached_attractors()

    def deactivate_reached_attractors(self):
        for attractor_i in range(len(self.attraction_points)):
            closest_node = self.get_closest_node(attractor_i, self.kill_distance)
            if closest_node is not None:
                self.attraction_active[attractor_i] = False

    def draw_2d(self, *args, **kwargs):
        f = 1
        K = np.eye(3)
        K[0, 0] = f
        K[1, 1] = f
        u0 = 0
        v0 = 0
        K[0, 2] = u0
        K[1, 2] = v0
        R = np.eye(3)
        t = np.array([0, 0, -5]).reshape(3, 1)
        Rt = np.concatenate((R, t), axis=1)
        P = np.dot(K, Rt)

        pts = np.array(self.nodes).T
        pts = e2p(pts)
        
        nodes_2d = p2e(np.dot(P, pts)).T
        # nodes_2d = np.array(self.nodes)

        for node_i, node in enumerate(nodes_2d):
            parent = self.parent[node_i]
            if parent is None:
                continue

            end = nodes_2d[parent]
            start_ga = node[0]*g2.e1 + node[1]*g2.e2
            end_ga = end[0]*g2.e1 + end[1]*g2.e2
            print('start_ga: {}'.format(start_ga))
            print('end_ga: {}'.format(end_ga))
            line_ga = start_ga ^ end_ga
            print('line_ga: {}'.format(line_ga))
            plt.plot([node[0], end[0]],
                     [node[1], end[1]],
                     *args, **kwargs)
        
    def draw(self, ax=None, *args, **kwargs):
        if ax is None:
            ax = plt.gca()

        for node_i, node in enumerate(self.nodes):
            parent = self.parent[node_i]
            if parent is None:
                continue

            end = self.nodes[parent]
            ax.plot([node[0], end[0]],
                    [node[1], end[1]],
                    [node[2], end[2]],
                    *args, **kwargs)

    def get_closest_node(self, attractor_id, max_dist_thr):
        if not self.attraction_active[attractor_id]:
            return None
        attractor_xyz = self.attraction_points[attractor_id]

        min_dist = float('inf')
        min_i = None
        for node_i, node in enumerate(self.nodes):
            dist = np.sqrt((attractor_xyz[0]-node[0])**2 +
                           (attractor_xyz[1]-node[1])**2 +
                           (attractor_xyz[2]-node[2])**2)
            if dist < min_dist and dist <= max_dist_thr:
                min_dist = dist
                min_i = node_i

        return min_i
        
    def generate_attraction_points(self, N):
        """ drop shape from:
https://math.stackexchange.com/a/481988/660780 """
        i = 0
        points = []
        while i < N:
            x, y, z = np.random.rand(3)
            x = remap(x, 0, 1, -1, 1)
            y = remap(y, 0, 1, -1, 1)
            z = remap(z, 0, 1, -1, 0)
            in_drop_shape = (x**2 + y**2 + z**4 - z**2 <= 0)
            if in_drop_shape:
                points.append((x, remap(z, -1, 0, 1, 2), y))
                i += 1
        return points

def run(args):
    if os.path.exists('cache.pkl'):
        with open('cache.pkl', 'rb') as f:
            tree = pickle.load(f)
    else:
        tree = SpaceTree(radius_of_influence=1,
                        growth_D=1*0.02,
                        kill_distance=5*0.02,
                        N_attractors=5000,
                        flat_tree=True)
        with open('cache.pkl', 'wb') as f:
            pickle.dump(tree, f)
    # fig = plt.figure()
    # ax = Axes3D(fig)
    # pts = np.array(tree.attraction_points)
    # ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
    #            alpha=0.1)
    # tree.draw(ax, 'k-')
    tree.draw_2d('k-')
    plt.axis('equal')
    plt.gca().invert_yaxis()
    plt.show()
    return 0

def main():
    args = parse_arguments()
    return run(args)

if __name__ == '__main__':
    sys.exit(main())
