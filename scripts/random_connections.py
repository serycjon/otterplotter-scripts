# -*- coding: utf-8 -*-
import sys
import argparse
import numpy as np
import vpype
import vpype_viewer


def parse_arguments():
    parser = argparse.ArgumentParser(description='',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    return parser.parse_args()


def circle_connections(N, R, max_delta, min_delta, init_angle=0):
    angles = np.cumsum(np.random.rand(N) * (max_delta - min_delta) + min_delta) + init_angle
    xs = R * np.cos(angles)
    ys = R * np.sin(angles)
    pts = [np.stack((xs, ys), axis=1)]
    return pts


def drop_parts(lines, space=2, passes=1):
    if passes < 1:
        return lines
    result = []
    for line in lines:
        try:
            split_i = np.random.randint(0, len(line) - 1)
            left = line[0:split_i, :]
            right = line[split_i + space:, :]
            result.append(left)
            result.append(right)
        except Exception:
            result.append(line)
    return drop_parts(result, space, passes - 1)


def run(args):
    R = 150
    smaller = R / 1.6180339  # golden ratio
    pts = []
    pts.extend(circle_connections(230, R, np.pi * 0.3, 0))
    pts.extend(circle_connections(530, R, np.pi * 0.5, 0))
    pts.extend(circle_connections(650, R, np.pi * 0.65, np.pi * 0.50))
    pts.extend(circle_connections(700, R, np.pi * 0.95, np.pi * 0.72))
    for i in range(1, 18):
        smooth_circle = circle_connections(360, R * 1.02**i, np.pi / 180, np.pi / 180)
        pts.extend(drop_parts(smooth_circle, space=1, passes=4))
    lines = to_vpype(pts)
    lines.crop(R - smaller, 0, R, -R)
    document = vpype.Document(lines)
    # document.extend_page_size(None)
    with open('/tmp/circles.svg', 'w') as fout:
        vpype.write_svg(fout, document)
    vpype_viewer.show(document)
    return 0


def to_vpype(paths):
    lc = vpype.LineCollection()
    for path in paths:
        lc.append(path[:, 0] + path[:, 1] * 1.j)
    return lc


def main():
    args = parse_arguments()
    return run(args)


if __name__ == '__main__':
    sys.exit(main())
