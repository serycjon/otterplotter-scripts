import vpype
import numpy as np


def to_vpype(paths):
    lc = vpype.LineCollection()
    for path in paths:
        lc.append(path[:, 0] + path[:, 1] * 1.j)
    return lc


def from_vpype(lines):
    results = []
    for line in lines:
        xs = np.real(line)
        ys = np.imag(line)
        results.append(np.stack((xs, ys), axis=1))

    return results
