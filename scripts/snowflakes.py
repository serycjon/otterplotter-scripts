# -*- coding: utf-8 -*-
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt

from primitives import vis_drawing, resize_and_center, rotate, circle, mask_drawing, reflect, shift
from primitives import optimize, load_page_conf
try:
    from OtterPlotter import Plotter
except ImportError:
    from fake_plotter import FakePlotter as Plotter
from repro import ReproSaver


def parse_arguments():
    parser = argparse.ArgumentParser(description='',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--nosave', help='do not save results', action='store_true')
    parser.add_argument('--format', help='path to a format json', required=True)
    parser.add_argument('--opconfig', help='OtterPlotter calibration path', default='config.json')
    parser.add_argument('--rotate', help='rotate drawing 90 deg', action='store_true')

    return parser.parse_args()


def snowflakify(paths, n=6, do_prune=True):
    if do_prune:
        paths = prune(paths, n)
    result = []
    angles = np.arange(n) * (2 * np.pi / n)
    for angle in angles:
        result.extend(rotate(paths, angle))
    return result


def prune(paths, n):
    angle = 360 / n
    mask = circle((0, 0), 10, N=2, angle_start=-angle / 2, angle_end=angle / 2).tolist()
    mask = np.array([[0, 0]] + mask + [[0, 0]])
    # return [mask]
    return mask_drawing(paths, mask)


def branch(line, where, angle, length):
    branch_point = where * line[1] + (1 - where) * line[0]

    line_direction = line[1] - line[0]
    line_length = np.linalg.norm(line_direction)
    line_direction /= line_length

    CCW_direction = rotate([line_direction], angle)[0]
    CCW_line = np.array([branch_point, branch_point + length * line_length * CCW_direction])

    CW_direction = rotate([line_direction], -angle)[0]
    CW_line = np.array([branch_point, branch_point + length * line_length * CW_direction])
    return CCW_line, CW_line


def execute_branching(line, branching_spec):
    if branching_spec is None:
        return [line]
    else:
        current_branches = []
        for spec in branching_spec:
            _, new_branch = branch(line, spec[0], np.radians(spec[1]), spec[2])
            complete_branch = execute_branching(new_branch, spec[3])
            current_branches.extend(complete_branch)
            current_branches.extend(reflect(complete_branch, line))
        return [line] + current_branches


def generate_branching(depth=0):
    n_branches = np.random.randint(1, 4)
    if n_branches == 0 or depth > 1:
        return None

    res = []
    for i in range(n_branches):
        where = np.random.rand()
        angle = np.random.randint(30, 65)
        length = np.random.randint(4, 10) / 10
        branch = (where, angle, length, generate_branching(depth=depth + 1))
        res.append(branch)

    return res


def snowflake():
    branching = generate_branching()
    all_paths = execute_branching(np.array([[0, 0], [1.0, 0]]), branching)

    paths = snowflakify(all_paths)
    # paths = rotate(paths, np.radians(np.random.randint(0, 360)))
    return resize_and_center(paths, 1, 1, 0, 0, 0, 0)


def run(args):
    saver = ReproSaver()
    saver.seed()

    snowflakes = []
    rows, columns = 5, 7
    # disabled = [(0, 0), (0, 1)]
    disabled = []
    a = 1.2
    row_height = a * (np.sqrt(3) / 2)
    for r in range(rows):
        for c in range(columns):
            if (r, c) in disabled:
                continue
            row_shift = 0 if (r % 2) == 0 else (a / 2)
            delta = (row_shift + c * a, r * row_height)
            snowflakes.append(shift(snowflake(), delta))

    paths = []
    for sflake in snowflakes:
        paths.extend(optimize(sflake,
                              path_join_threshold=0.01,
                              line_simplification_threshold=0,
                              path_drop_threshold=0.01,
                              verbose=False))

    page_conf = load_page_conf(args.format)
    H = page_conf['H']  # 210 # A4
    W = page_conf['W']  # 297 # A4
    x_margin_mm, y_margin_mm = 10, 10
    to_draw = resize_and_center(paths, H, W,
                                x_margin_mm, x_margin_mm,
                                y_margin_mm, y_margin_mm)
    if args.rotate:
        to_draw = rotate(to_draw)
    # paths = optimize(paths,
    #                  path_join_threshold=0.01,
    #                  line_simplification_threshold=0,
    #                  path_drop_threshold=0.01)

    if not args.nosave:
        saver.add_svg(paths)

    vis_drawing(to_draw, 'k-', lw=0.5)
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


def main():
    args = parse_arguments()
    return run(args)

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
