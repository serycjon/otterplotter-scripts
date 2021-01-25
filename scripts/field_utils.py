# -*- coding: utf-8 -*-
import os
import sys
import argparse
import numpy as np
from collections import deque
from opensimplex import OpenSimplex
from kdtree import get_nearest, add_point, make_kd_tree, rebalance
from img_tools import clip_to_img, resize_to_max
from caching import CachedComputation
from geometry import remap
from various_utils import with_debugger
import cv2

import matplotlib.pyplot as plt
import tqdm
from svg import export_svg
from repro import ReproSaver
from opengl_utils import Drawer

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("asyncio").setLevel(logging.WARNING)


eps = 1e-6


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--nosave', help='do not save results', action='store_true')
    parser.add_argument('--mult', help='simplex noise scaling factor', type=float, default=0.006)
    parser.add_argument('--img', help='path to guide image')
    parser.add_argument('--sep', help='streamline separation', type=float, default=6)

    return parser.parse_args()


def dist_fn(x, y):
    return np.linalg.norm(x - y)


@CachedComputation(os.path.join("cache", "gen_flow_field.pkl"))
def gen_flow_field(H, W, x_mult=1, y_mult=None):
    if y_mult is None:
        y_mult = x_mult
    x_noise = OpenSimplex(np.random.randint(9393931))
    y_noise = OpenSimplex(np.random.randint(9393931))
    field = np.zeros((H, W, 2), dtype=np.float64)
    for y in range(H):
        for x in range(W):
            x_val = x_noise.noise2d(x=x_mult * x, y=x_mult * y)
            y_val = y_noise.noise2d(x=y_mult * x, y=y_mult * y)
            norm = np.sqrt(x_val ** 2 + y_val ** 2)
            if norm > eps:
                x_val /= norm
                y_val /= norm
            else:
                x_val, y_val = 0, 0
            field[y, x, :] = (x_val, y_val)

    return field


@with_debugger
def run(args):
    saver = ReproSaver()
    saver.seed()
    # H, W = 2100, 2970
    # d_sep = 4
    H, W = 600, 800
    # d_sep = 16
    d_sep = args.sep
    seedpoints_per_path = 40
    mult = args.mult

    if args.img is not None:
        img = cv2.imread(args.img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = resize_to_max(gray, 800)
        H, W = gray.shape
    else:
        gray = None

    logging.info('Generating noise field')
    field = gen_flow_field(H, W, x_mult=mult)

    # def d_sep_fn(pos):
    #     return d_sep

    def d_sep_fn(pos):
        x, y = clip_to_img(np.round(pos), gray).astype(np.int32)
        val = gray[y, x] / 255
        val = val**2
        return remap(val, 0, 1, 0.8, 10)

    logging.debug('Drawing uniform density streamlines')
    paths = draw_field_uniform(field, d_sep_fn,
                               seedpoints_per_path=seedpoints_per_path,
                               guide=gray)

    gui = Drawer()
    gui.add_lines(paths, 'k')
    gui.draw()
    # vis_field(field)
    for path in paths:
        plt.plot([x for x, _ in path],
                 [y for _, y in path], 'k-', lw=0.3)
    # field_2 = np.zeros_like(field)
    # field_2[:, :, 0] = field[:, :, 1]
    # field_2[:, :, 1] = -field[:, :, 0]
    # paths = draw_field_uniform(field_2, d_sep)
    # # vis_field(field)
    # for path in paths:
    #     plt.plot([x for x, _ in path],
    #              [y for _, y in path], 'b-', lw=0.3)

    export_svg(paths, '/tmp/uniform_flow.svg')
    # plt.axis('image')
    # plt.gca().invert_yaxis()
    # plt.show()

    if not args.nosave:
        saver.add_svg(paths)

    return 0


def rotate_field(field, degrees):
    s, c = np.sin(np.radians(degrees)), np.cos(np.radians(degrees))
    R = np.array([[c, -s],
                  [s, c]])
    return np.matmul(R, field.reshape(-1, 2).T).T.reshape(field.shape)


def draw_field_uniform(field, d_sep_fn, d_test_fn=None,
                       seedpoints_per_path=10,
                       guide=None):
    # field_2 = rotate_field(field, 90)
    # field_3 = rotate_field(field, 60)
    # field_4 = rotate_field(field, 120)
    # field_2[:, :, 0] = -field[:, :, 1]
    # field_2[:, :, 1] = field[:, :, 0]
    # fields = [field, field_3, field_4]
    fields = [VectorField(field),
              # VectorField(rotate_field(field, 60)),
              # VectorField(rotate_field(field, 120)),
              # VectorField(rotate_field(field, 180)),
              # VectorField(rotate_field(field, 240)),
              # VectorField(rotate_field(field, 300)),
              ]

    field_i = 0

    if d_test_fn is None:
        def d_test_fn(*args, **kwargs):
            return d_sep_fn(*args, **kwargs) / 2

    H, W = field.shape[:2]

    def should_stop(new_pos, kdtree, path, d_sep_fn):
        if not inside(np.round(new_pos), H, W):
            return True
        if kdtree is not None:
            point = new_pos.copy()
            nearest = get_nearest(kdtree, point, dim=2,
                                  dist_func=dist_fn,
                                  return_distances=True)
            dist, pt = nearest
            if dist < d_sep_fn(new_pos):
                return True
        # compute streamline length
        length = 0
        cur = path[0]

        for pt in path:
            length += np.linalg.norm(cur - pt)
            cur = pt

        if length > 10:
            return True

        # look for loops
        # candidate = np.round(new_pos).astype(np.int64).reshape(1, 2)
        # for pt in reversed(path):
        #     if np.all(candidate == np.round(pt).astype(np.int64)):
        #         return True
        return False

    kdtree = make_kd_tree([np.array([-10, -10])], dim=2)
    paths = []
    rebalance_every = 500
    save_every = 100
    seed_pos = np.array((W / 2, H / 2))
    seedpoints = [seed_pos]
    seedpoints = deque(seedpoints)
    pbar = tqdm.tqdm()
    logging.info('Drawing noise field')
    try:
        while True:
            # try to find a suitable seedpoint in the queue
            try:
                while True:
                    seed_pos = seedpoints.pop()
                    if not inside(np.round(seed_pos), H, W):
                        continue

                    dist, _ = get_nearest(kdtree, seed_pos, dim=2,
                                          dist_func=dist_fn,
                                          return_distances=True)
                    if dist < d_sep_fn(seed_pos):
                        continue

                    break
            except IndexError:
                # no more seedpoints
                logging.debug('no more seedpoints')
                # raise
                break

            # field_to_use = fields[field_i]
            # field_i = (field_i + 1) % len(fields)

            # def select_field(_, _):
            #     return field_to_use

            start_field = np.random.randint(len(fields))

            def select_field(path_len, direction):
                same_field_len = 10

                idx = int(direction * path_len // same_field_len) + start_field
                idx = idx % len(fields)
                return fields[idx]

            class MemorySelector():
                def __init__(self, fields):
                    self.same_field_len = 10
                    self.cur_len = 0
                    self.idx = np.random.randint(len(fields))
                    self.fields = fields

                def select_field(self, path_len, direction):
                    if (path_len - self.cur_len) > self.same_field_len:
                        self.cur_len = path_len
                        idx_delta = np.random.randint(-1, 1 + 1)
                        self.idx = (self.idx + idx_delta) % len(self.fields)

                    return self.fields[self.idx]

            selector = MemorySelector(fields)

            path = compute_streamline(selector.select_field, seed_pos, kdtree, d_test_fn, d_sep_fn,
                                      should_stop_fn=should_stop)
            if len(path) <= 2:
                # nothing found
                # logging.debug('streamline ended immediately')
                continue

            for pt in path:
                add_point(kdtree, pt, dim=2)
            paths.append(path)
            if len(paths) % rebalance_every == 0:
                kdtree = rebalance(kdtree, dim=2)
            if len(paths) % save_every == 0:
                export_svg(paths, '/tmp/uniform_flow.svg')

            new_seedpoints = generate_seedpoints(path, d_sep_fn,
                                                 seedpoints_per_path)
            order = np.arange(len(new_seedpoints))
            np.random.shuffle(order)
            seedpoints.extend([new_seedpoints[i] for i in order])
            pbar.update(1)
    except KeyboardInterrupt:
        pass

    pbar.close()
    return paths


class VectorField():
    def __init__(self, field_array):
        self.field = field_array

    def __getitem__(self, pos):
        ''' pos should be (x, y) '''
        round_pos = np.round(pos[:2]).astype(np.int64)
        round_pos = fit_inside(round_pos, self.field)

        return self.field[round_pos[1], round_pos[0], :]

    @property
    def shape(self):
        return self.field.shape


def fit_inside(xy, img):
    return np.clip(xy,
                   np.array([0, 0], xy.dtype),
                   np.array([img.shape[1] - 1, img.shape[0] - 1], xy.dtype))


def compute_streamline(field_getter, seed_pos, kdtree, d_test_fn, d_sep_fn,
                       should_stop_fn):
    direction_sign = 1  # first go with the field
    pos = seed_pos.copy()
    paths = []
    path = [pos.copy()]
    path_length = 0
    stop_tracking = False
    self_kdtree = make_kd_tree([(-20, -20)], dim=2)
    while True:
        field = field_getter(path_length, direction_sign)
        rk_force = runge_kutta(field, pos, d_test_fn(pos)) * direction_sign
        new_pos = pos + d_test_fn(pos) * rk_force

        # test validity
        if should_stop_fn(new_pos, kdtree, path, d_sep_fn):
            stop_tracking = True

        # prevent soft looping
        if get_nearest(self_kdtree, new_pos, dim=2,
                       dist_func=dist_fn, return_distances=True)[0] < d_sep_fn(pos):
            stop_tracking = True
        lookback = 15
        if len(path) >= 2 * lookback:
            add_point(self_kdtree, path[-lookback], dim=2)

        # fallback
        if len(path) >= 600:
            stop_tracking = True

        if not stop_tracking:
            path.append(new_pos.copy())
            path_length += np.linalg.norm(pos - new_pos)

        if stop_tracking:
            paths.append(path)
            if direction_sign == 1:
                # go to the other side from the seed
                direction_sign = -1
                pos = seed_pos.copy()
                path = [pos.copy()]
                path_length = 0
                # self_kdtree = make_kd_tree([(-20, -20)], dim=2)
                stop_tracking = False
            else:
                # both directions finished
                break
        else:
            pos = new_pos
    singleline = list(reversed(paths[1]))
    singleline.extend(paths[0])

    return singleline


def generate_seedpoints(path, d_sep_fn, N_seedpoints=10):
    # go along the path and create points perpendicular in d_sep distance
    seeds = []
    seedpoint_positions = np.linspace(0, len(path) - 1, N_seedpoints)
    seedpoint_ids = np.unique(np.round(seedpoint_positions)).tolist()

    cur_xy = path[0]
    direction = path[1] - path[0]
    direction /= max(np.linalg.norm(direction), eps)
    normal = np.array((direction[1], -direction[0]))
    margin = 1.1
    seeds.append(cur_xy + margin * d_sep_fn(cur_xy) * normal)
    seeds.append(cur_xy - margin * d_sep_fn(cur_xy) * normal)

    for i in range(1, len(path)):
        if i not in seedpoint_ids:
            continue
        last_xy = cur_xy.copy()
        cur_xy = path[i]
        direction = cur_xy - last_xy
        direction /= max(np.linalg.norm(direction), eps)
        normal = np.array((direction[1], -direction[0]))
        seeds.append(cur_xy + margin * d_sep_fn(cur_xy) * normal)
        seeds.append(cur_xy - margin * d_sep_fn(cur_xy) * normal)

    return seeds


def vis_field(field, step=10):
    H, W = field.shape[:2]
    sample_xs = np.arange(W, step=step)
    sample_ys = np.arange(H, step=step)

    length = step / 2

    for y in sample_ys:
        for x in sample_xs:
            direction = field[y, x, :]
            direction /= max(np.linalg.norm(direction), eps)
            sample_pt = np.array([x, y])
            start = sample_pt - length * (direction / 2)
            stop = sample_pt + length * (direction / 2)
            plt.plot([start[0], stop[0]], [start[1], stop[1]],
                     'k-', lw=0.1)


def inside(xy_pt, H, W):
    return (xy_pt[0] >= 0 and
            xy_pt[1] >= 0 and
            xy_pt[0] < W and
            xy_pt[1] < H)


def runge_kutta(field, pos, h):
    k1 = field[pos]

    k2_pos = pos + (h / 2) * k1
    k2 = field[k2_pos]

    k3_pos = pos + (h / 2) * k2
    k3 = field[k3_pos]

    k4_pos = pos + h * k3
    k4 = field[k4_pos]

    # Runge-Kutta for the win
    rk = (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    return rk


def main():
    args = parse_arguments()
    return run(args)


if __name__ == '__main__':
    sys.exit(main())
