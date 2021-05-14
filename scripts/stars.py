# -*- coding: utf-8 -*-
import sys
import copy
import argparse
import numpy as np
import vpype
import vpype_viewer
from geometry import remap
from repro import ReproSaver
from vpype_integration import to_vpype_document, from_vpype_document
import vpype_cli
from various_utils import with_debugger
from collections import defaultdict

from numpy import linspace, geomspace


def parse_arguments():
    parser = argparse.ArgumentParser(description='',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--nosave', help='', action='store_true')
    parser.add_argument('--rnd', help='do a random one', action='store_true')
    parser.add_argument('--novis', help='disable display', action='store_true')

    return parser.parse_args()


def rev_geomspace(start, stop, num):
    return remap(np.geomspace(start, stop, num),
                 start, stop,
                 stop, start)


def star_line(N_star, start_r, end_r, end_skip,
              start_skip=1, mirror=True):
    star_angles = np.linspace(0, 2 * np.pi, N_star + 1)[:-1]

    start_ids = np.arange(0, N_star, start_skip)
    end_ids = (start_ids + end_skip) % N_star

    start_angles = star_angles[start_ids]
    end_angles = star_angles[end_ids]

    if mirror:
        mirror_end_ids = (start_ids - end_skip) % N_star
        mirror_end_angles = star_angles[mirror_end_ids]
        start_angles = np.concatenate((start_angles, start_angles))
        end_angles = np.concatenate((end_angles, mirror_end_angles))

    lines = []
    for start_angle, end_angle in zip(start_angles, end_angles):
        lines.append(np.array(
            [(start_r * np.cos(start_angle),
              start_r * np.sin(start_angle)),
             (end_r * np.cos(end_angle),
              end_r * np.sin(end_angle))]))
    return lines


all_colors = ['black', 'blue', 'red', 'green']


def random_star(R=600, n_colors=4):
    colors = all_colors[:n_colors]
    description = []
    N_star = np.random.randint(4, 20)
    N_layers = np.random.randint(3, 5)

    range_fns = ['linspace', 'geomspace', 'rev_geomspace']

    for layer_i in range(N_layers):
        layer = {}
        layer['R'] = R
        layer['N_lines'] = np.random.randint(1, 30)
        layer['N_star'] = N_star

        layer['start_low'] = np.random.randint(1, 100) / 100
        layer['start_high'] = np.random.randint(1, 100) / 100

        layer['end_low'] = np.random.randint(1, 100) / 100
        layer['end_high'] = np.random.randint(1, 100) / 100

        layer['start_fn'] = range_fns[np.random.randint(len(range_fns))]
        layer['end_fn'] = range_fns[np.random.randint(len(range_fns))]

        layer['start_reversed'] = np.random.randint(2) > 0
        layer['end_reversed'] = np.random.randint(2) > 0

        layer['end_skip'] = np.random.randint(1, int(N_star / 2))
        layer['start_skip'] = np.random.randint(1, int(N_star / 2))
        layer['mirror'] = np.random.randint(2) > 0

        layer['color'] = str(np.random.choice(colors))
        description.append(layer)

    return description


def mutate_float(lb, ub, sigma):
    def do_it(x):
        return np.clip(np.random.randn() * sigma,
                       lb, ub)
    return do_it


def identity(x):
    return x


def invert(x):
    return not x


def mutate_int(lb, ub, sigma):
    def do_it(x):
        return int(np.clip(
            np.round(np.random.randn() * sigma),
            lb, ub))
    return do_it


def mutate_param(name, value, layer):
    actions = {
        'R': identity,
        'N_star': identity,
        'N_lines': mutate_int(1, 30, 1),
        'start_low': mutate_float(0.01, 1, 0.1),
        'start_high': mutate_float(0.01, 1, 0.1),
        'start_reversed': invert,
        'end_low': mutate_float(0.01, 1, 0.1),
        'end_high': mutate_float(0.01, 1, 0.1),
        'end_reversed': invert,
        'end_skip': mutate_int(1, layer['N_star'], 1),
        'start_skip': mutate_int(1, layer['N_star'], 1),
        'mirror': invert,
        'color': lambda _: str(np.random.choice(all_colors)),
    }
    return actions.get(name, identity)(value)


def mutate_star(description):
    new_star = copy.deepcopy(description)
    action = np.random.choice(['drop_layer', 'add_layer', 'modify_param'], p=[0.1, 0.1, 0.8])
    if action == 'modify_param':
        layer_i = np.random.choice(len(description))
        param_name = np.random.choice(
            ['R', 'N_lines', 'N_star', 'start_low', 'start_high',
             'end_low', 'end_high', 'start_reversed', 'end_reversed',
             'end_skip', 'start_skip', 'mirror', 'color'])
        new_star[layer_i][param_name] = mutate_param(param_name,
                                                     new_star[layer_i][param_name],
                                                     new_star[layer_i])

    return new_star


@with_debugger
def construct_star(description):
    out_layers = defaultdict(list)
    for layer in description:
        start_fn, end_fn = globals()[layer['start_fn']], globals()[layer['end_fn']]

        start_range = start_fn(layer['start_low'], layer['start_high'], layer['N_lines'])
        if layer['start_reversed']:
            start_range = reversed(start_range)

        end_range = end_fn(layer['end_low'], layer['end_high'], layer['N_lines'])
        if layer['end_reversed']:
            end_range = reversed(end_range)

        for start_fr, end_fr in zip(start_range, end_range):
            start_r = layer['R'] * start_fr
            end_r = layer['R'] * end_fr
            out_layers[layer['color']].extend(star_line(layer['N_star'], start_r, end_r,
                                                        layer['end_skip'], layer['start_skip'],
                                                        layer['mirror']))
    return out_layers


def run(args):
    saver = ReproSaver('star_results')
    saver.seed()

    if args.rnd:
        description = random_star()
        print(f"description: {description}")
        layers = construct_star(description)
    else:
        N_star = 12
        R = 600
        paths = []
        start_fractions = np.linspace(0.4, 1, 25)
        end_fractions = reversed(start_fractions)
        for start_fraction, end_fraction in zip(start_fractions,
                                                end_fractions):
            start_r = R * start_fraction
            end_r = R * end_fraction
            paths.extend(star_line(N_star, start_r, end_r, 1))

        # vlastovka
        start_fractions = np.geomspace(0.4, 0.65, 20)
        end_fractions = reversed(np.geomspace(0.3, 0.9, 20))
        for start_fraction, end_fraction in zip(start_fractions,
                                                end_fractions):
            start_r = R * start_fraction
            end_r = R * end_fraction
            paths.extend(star_line(N_star, start_r, end_r, 1))

        # stred
        start_fractions = rev_geomspace(0.1, 1, 30)
        end_fractions = start_fractions
        # end_fractions = np.linspace(0.1, 1, 30)
        for start_fraction, end_fraction in zip(start_fractions,
                                                end_fractions):
            start_r = R * start_fraction
            end_r = R * end_fraction
            paths.extend(star_line(N_star, start_r, end_r, 5, mirror=False))

        # stred 2
        start_fractions = rev_geomspace(0.13, 0.7, 20)
        end_fractions = start_fractions
        # end_fractions = np.linspace(0.1, 1, 30)
        for start_fraction, end_fraction in zip(start_fractions,
                                                end_fractions):
            start_r = R * start_fraction
            end_r = R * end_fraction
            paths.extend(star_line(N_star, start_r, end_r, 5, mirror=False))

        paths.extend(star_line(N_star, R, R, 6, mirror=False))
        layers = {'black': paths}

    document = to_vpype_document(layers)
    document = vpype_cli.execute("linemerge linesort", document)
    layers = from_vpype_document(document)
    if not args.nosave:
        saver.add_svg(layers)
    with open('/tmp/stars.svg', 'w') as fout:
        vpype.write_svg(fout, document, color_mode='layer')
    if not args.novis:
        vpype_viewer.show(document)
    return 0


def main():
    args = parse_arguments()
    return run(args)


if __name__ == '__main__':
    sys.exit(main())
