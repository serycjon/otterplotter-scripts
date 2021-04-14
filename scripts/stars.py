# -*- coding: utf-8 -*-
import sys
import argparse
import numpy as np
import vpype
import vpype_viewer
from primitives import shift, mask_drawing, circle
from geometry import remap
from repro import ReproSaver
from vpype_integration import to_vpype, from_vpype


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


def random_star(R=600):
    paths = []
    N_star = np.random.randint(4, 20)
    N_layers = np.random.randint(3, 5)

    range_fns = [np.linspace, np.geomspace, rev_geomspace]

    for layer in range(N_layers):
        N_lines = np.random.randint(1, 30)

        start_low = np.random.randint(1, 100) / 100
        start_high = np.random.randint(1, 100) / 100

        end_low = np.random.randint(1, 100) / 100
        end_high = np.random.randint(1, 100) / 100

        start_fn = range_fns[np.random.randint(len(range_fns))]
        end_fn = range_fns[np.random.randint(len(range_fns))]

        start_range = start_fn(start_low, start_high, N_lines)
        end_range = end_fn(end_low, end_high, N_lines)

        if np.random.randint(2) > 0:
            start_range = reversed(start_range)
        if np.random.randint(2) > 0:
            end_range = reversed(end_range)

        end_skip = np.random.randint(1, int(N_star / 2))
        start_skip = np.random.randint(1, int(N_star / 2))
        mirror = np.random.randint(2) > 0

        for start_fr, end_fr in zip(start_range, end_range):
            start_r = R * start_fr
            end_r = R * end_fr
            paths.extend(star_line(N_star, start_r, end_r, end_skip, start_skip, mirror))

    return paths


def run(args):
    saver = ReproSaver('star_results')
    saver.seed()

    if args.rnd:
        paths = random_star()
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

    lines = to_vpype(paths)
    if not args.nosave:
        saver.add_svg(paths)
    document = vpype.Document(lines)
    with open('/tmp/stars.svg', 'w') as fout:
        vpype.write_svg(fout, document)
    if not args.novis:
        vpype_viewer.show(document)
    return 0


def main():
    args = parse_arguments()
    return run(args)


if __name__ == '__main__':
    sys.exit(main())
