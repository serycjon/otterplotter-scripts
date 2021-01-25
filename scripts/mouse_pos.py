# -*- coding: utf-8 -*-
import sys
import argparse
import time
import subprocess
import numpy as np

import matplotlib.pyplot as plt


def parse_arguments():
    parser = argparse.ArgumentParser(description='',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--delay', help='sleep time in [s]', type=float, default=1)
    parser.add_argument('--export', help='out path')
    parser.add_argument('--plot', help='', action='store_true')

    return parser.parse_args()


def run(args):
    if args.plot:
        coords = np.loadtxt(args.export)

        start = 40000
        plt.plot(coords[start:, 0], coords[start:, 1], lw=0.1)
        plt.show()
        sys.exit(0)
        
    coords = []
    i = 0
    while True:
        output = subprocess.check_output(["xdotool", "getmouselocation"])
        output = output.decode('utf-8').strip()
        info = dict([x.split(':') for x in output.split(' ')])
        x = int(info['x'])
        y = int(info['y'])
        coords.append((x, y))
        i += 1

        if i > 300:
            i = 0
            if args.export is not None:
                np.savetxt(args.export, coords)
        time.sleep(args.delay)
    return 0


def main():
    args = parse_arguments()
    return run(args)


if __name__ == '__main__':
    sys.exit(main())
