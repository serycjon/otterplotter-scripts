# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import

import sys
import argparse

from OtterPlotter import Plotter
from primitives import circle, vis_drawing, drawing_stats, resize_and_center
from primitives import optimize
from attractors import simplify_reumann_witkam
import numpy as np
import matplotlib.pyplot as plt
from opensimplex import OpenSimplex
import tqdm

def parse_arguments():
    parser = argparse.ArgumentParser(description='',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    return parser.parse_args()

def run_particle(particle, forces, N_steps=100, step_size=1):
    path = [particle.copy()]
    current = particle.copy()
    velocity = np.array((0, 0), dtype=np.float64)
    for i in range(N_steps):
        F = np.array((0, 0), dtype=np.float64)
        for force in forces:
            F += force(current)

        max_F = 1
        F = np.clip(F, -max_F, max_F)

        velocity += F / 10000
        current += velocity * step_size
        path.append(current.copy())

    return np.array(path)

def gravity(a_xy, b_xy, m_a=1, m_b=1):
    """ gravity force to a_xy """
    towards_b = b_xy - a_xy
    dist = np.linalg.norm(towards_b)
    if (dist == 0):
        return np.zeros(2, np.float64)
    force = (m_a * m_b) / dist**2
    force_vector = force * (towards_b / dist)
    return force_vector

def border_repulsion(xy, tl_xy, br_xy):
    dlx = None


def run(args):
    particles = circle((0, 0), 1, N=360)
    forces = []

    x_noise = OpenSimplex(42)
    y_noise = OpenSimplex(211)
    np.random.seed(64)

    dist = lambda x: np.linalg.norm(x)

    forces.append(lambda x: 0.6*np.array(
        [x_noise.noise2d(x=x[0], y=x[1]),
         y_noise.noise2d(x=x[0], y=x[1])]))
    forces.append(lambda x: 0.05*gravity(x, np.array((2, 2))))
    forces.append(lambda x: 0.05*gravity(x, np.array((1.5, 0))))
    forces.append(lambda x: 0.05*gravity(x, np.array((1.5, 1.5))))
    forces.append(lambda x: -x if dist(x) > 3 else np.array((0.0, 0.0)))
    to_draw = []
    for p in tqdm.tqdm(particles):
        path = run_particle(p, forces, N_steps=np.random.randint(999, 1000), step_size=1)
        to_draw.append(path)

    x_margin_mm = 30
    y_margin_mm = 30
    H = 210 # A4
    W = 297 # A4

    to_draw = resize_and_center(to_draw, H, W,
                                x_margin_mm, x_margin_mm,
                                y_margin_mm, y_margin_mm)

    simplified = []
    for p in to_draw:
        simplified.append(simplify_reumann_witkam(0.5, p))

    to_draw = optimize(simplified)

    vis_drawing(to_draw, 'b-', lw=0.1)
    plt.axis('equal')
    plt.gca().invert_yaxis()
    plt.show()

    # return 1
    with Plotter('/dev/ttyUSB0', 9600) as p:
        p.load_config('config.json')
        p.set_input_limits((0, 0), (W, 0),
                           (0, H), (W, H))
        p.draw_polylines(to_draw)

    return 0

def main():
    args = parse_arguments()
    return run(args)

if __name__ == '__main__':
    sys.exit(main())
