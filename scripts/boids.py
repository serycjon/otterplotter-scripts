# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import

import sys
import argparse

import numpy as np
from sklearn.neighbors import KDTree
import cv2

from primitives import vis_drawing, save_drawing
import matplotlib.pyplot as plt

class Boid:
    def __init__(self, pos, velocity,
                 max_speed, sense_r, mass,
                 separation_weight,
                 align_weight,
                 cohesion_weight):
        self.pos = pos
        self.velocity = velocity
        self.max_speed = max_speed
        self.sense_r = sense_r
        self.mass = mass
        self.separation_weight = separation_weight
        self.align_weight = align_weight
        self.cohesion_weight = cohesion_weight

    def compute_update(self, kdtree, other_boids, force_fields):
        query = self.pos.reshape(1, 2)
        neighbor_ids = kdtree.query_radius(query, self.sense_r)[0]
        neighbors = [other_boids[i] for i in neighbor_ids if not other_boids[i] is self]

        separation = self.separation_weight * self.calculate_separation_force(neighbors)
        align = self.align_weight * self.calculate_align_force(neighbors)
        cohesion = self.cohesion_weight * self.calculate_cohesion_force(neighbors)

        force = separation + align + cohesion
        for force_field in force_fields:
            force += force_field(self.pos, self.velocity)

        self.new_velocity = self.velocity + force / self.mass

    def calculate_separation_force(self, neighbors):
        result = np.array([0.0, 0.0])
        for neighbor in neighbors:
            away_vector = self.pos - neighbor.pos
            dist = np.linalg.norm(away_vector)
            strength = 1 - (dist / self.sense_r)
            force = (strength / dist) * away_vector
            result += force

        return result

    def calculate_cohesion_force(self, neighbors):
        result = np.array([0.0, 0.0])
        if len(neighbors) == 0:
            return result

        for neighbor in neighbors:
            result += neighbor.pos
        mean_pos = result / len(neighbors)
        to_mean = mean_pos - self.pos
        length = np.linalg.norm(to_mean)
        if length > 0:
            to_mean /= length
        return to_mean

    def calculate_align_force(self, neighbors):
        result = np.array([0.0, 0.0])
        for neighbor in neighbors:
            result += neighbor.velocity
        length = np.linalg.norm(result)
        if length > 0:
            result /= length
        return result

    def apply_update(self):
        self.velocity = self.new_velocity
        speed = np.linalg.norm(self.velocity)
        if speed > self.max_speed:
            self.velocity = (self.max_speed / speed) * self.velocity
        self.pos = self.pos + self.velocity

def step_simulation(boids, force_fields):
    positions = np.array([boid.pos for boid in boids])
    kdtree = KDTree(positions)

    for boid in boids:
        boid.compute_update(kdtree, boids, force_fields)

    for boid in boids:
        boid.apply_update()

def draw_boid(canvas, boid):
    if boid.mass > 3:
        color = (0, 0, 255)
    else:
        color = (255, 255, 255)
    cv2.circle(canvas, (int(round(boid.pos[0])), int(round(boid.pos[1]))),
               3, color, -1)

def parse_arguments():
    parser = argparse.ArgumentParser(description='',
				     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', help='random seed', type=int)
    parser.add_argument('--n_iter', help='number of steps', default=1000, type=int)
    parser.add_argument('--cv_vis', help='show CV visualization', action='store_true')
    
    return parser.parse_args()

def run(args):
    cv_vis = args.cv_vis
    if args.seed is not None:
        seed = args.seed
    else:
        import time
        seed = int(round(time.time() * 1000))

    np.random.seed(np.mod(seed, 2**32 - 1))
    N_boids = 300
    separation_weight = 1
    align_weight = 0.6
    cohesion_weight = 2
    sense_r = 60
    default_mass = 1
    max_speed = 2

    H, W = 600, 800
    def canvas_force_field(pos, vel):
        canvas_margin = 70
        x, y = pos
        x_force = 0
        y_force = 0

        power = 0.4

        # left
        distance = (x - 0)
        if distance < canvas_margin:
            strength = (canvas_margin - distance)**power
            x_force += strength
        # right
        distance = (W - x)
        if distance < canvas_margin:
            strength = (canvas_margin - distance)**power
            x_force -= strength
        # top
        distance = (y - 0)
        if distance < canvas_margin:
            strength = (canvas_margin - distance)**power
            y_force += strength
        # bottom
        distance = (H - y)
        if distance < canvas_margin:
            strength = (canvas_margin - distance)**power
            y_force -= strength

        force = np.array((x_force, y_force))
        return 0.1*force

    def gen_void_force_field(center, size):
        def void_force_field(pos, vel):
            to_center = center - pos
            dist = np.linalg.norm(to_center)
            strength = size*20 / (dist + 0.001)
            if dist > size*2:
                strength = 0
            force = -(strength / dist) * to_center
            return force
        return void_force_field

    def gen_pos():
        overshoot = 10
        x = np.random.randint(0-overshoot, W+overshoot)
        y = np.random.randint(0-overshoot, H+overshoot)
        return np.array([x, y], dtype=np.float64)

    def gen_vel():
        x = np.random.randint(-4, 4)
        y = np.random.randint(-4, 4)
        return np.array([x, y], dtype=np.float64)

    def gen_mass():
        min_mass = 0.2
        max_mass = 3
        fat = 1
        if np.random.rand() < 0.15:
            fat = 20
            min_mass *= fat
            max_mass *= fat

        mass = np.random.randn() + default_mass * fat
        return np.clip(mass, min_mass, max_mass)

    boids = [Boid(gen_pos(), gen_vel(),
                  max_speed,
                  sense_r, gen_mass(),
                  separation_weight,
                  align_weight,
                  cohesion_weight) for i in range(N_boids)]

    boid_traces = [[boid.pos] for boid in boids]
    force_fields = [canvas_force_field, gen_void_force_field(np.array((W/2, H/2)),
                                                             60)]
    for i in range(4):
        pos = gen_pos()
        size = np.random.randint(5, 30)
        force_fields.append(gen_void_force_field(pos, size))

    iter = 0
    while True:
        canvas = np.zeros((H, W, 3), dtype=np.uint8)
        [draw_boid(canvas, boid) for boid in boids]
        step_simulation(boids, force_fields)
        for i in range(len(boids)):
            boid_traces[i].append(boids[i].pos)
        if cv_vis:
            cv2.imshow("cv: canvas", canvas)
            c = cv2.waitKey(25)
            if c == ord('q'):
                break
        if args.n_iter is not None and iter >= args.n_iter:
            break
        iter += 1

    trace_canvas = np.ones_like(canvas) * 255
    layers = {'black': [],
              'red': []}
    for trace in boid_traces:
        color = (0, 0, 0)
        layer = 'black'
        if np.random.rand() < 0.05:
            color = (0, 0, 255)
            layer = 'red'
        layers[layer].append(np.array(trace))
        for i in range(1, len(trace)):
            start = (int(round(trace[i-1][0])),
                     int(round(trace[i-1][1])))
            end = (int(round(trace[i][0])),
                   int(round(trace[i][1])))
            cv2.line(trace_canvas, start, end, color, 1)
    if cv_vis:
        cv2.imshow("cv: trace_canvas", trace_canvas)
        while True:
            c = cv2.waitKey(0)
            if c == ord('q'):
                break

    save_drawing('drawings/boids.pkl', layers)
    vis_drawing(layers['black'], 'k-', lw=0.1)
    vis_drawing(layers['red'], 'r-', lw=0.3)
    plt.axis('equal')
    plt.gca().invert_yaxis()
    plt.show()
    return 0

def main():
    args = parse_arguments()
    return run(args)

if __name__ == '__main__':
    sys.exit(main())
