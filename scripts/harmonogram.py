# -*- coding: utf-8 -*-

import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt


def parse_arguments():
    parser = argparse.ArgumentParser(description='',
				     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    return parser.parse_args()

def generate_equations(freq, phase, amplitude, damping):
    assert len(phase) == len(freq)
    assert len(amplitude) == len(freq)
    assert len(damping) == len(freq)
    _freq = np.array(freq)
    _phase = np.array(phase)
    _amplitude = np.array(amplitude)
    _damping = np.array(damping)
    def equations(t):
        x = np.sum(_amplitude * np.sin(t * _freq + _phase) * np.power(np.e, -_damping * t))
        return x

    return equations

def run(args):
    x_eqn = generate_equations([2, 1.3], [0, 0], [1, 1.2], [0.0002, .0001])
    y_eqn = generate_equations([2/3, 2.6, 1.3], [0.1, 1, 1.1], [0.4, 0.9, 0.1], [0.0001, 0.001, 0.01])

    pts = []
    t = 0
    step = 0.01
    t_max = 10000
    while True:
        x = x_eqn(t)
        y = y_eqn(t)
        pts.append((x, y))
        t += step
        if t > t_max:
            break

    pts = np.array(pts)
    plt.plot(pts[:, 0], pts[:, 1], 'k-', lw=0.1)
    plt.show()
    return 0

def main():
    args = parse_arguments()
    return run(args)

if __name__ == '__main__':
    sys.exit(main())
