# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import

import os
import sys
import argparse
import tqdm

import numpy as np
import cv2

from geometry import remap
from opensimplex import OpenSimplex

def parse_arguments():
    parser = argparse.ArgumentParser(description='',
				     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--pic', help='path to picture', required=True)
    
    return parser.parse_args()

def run(args):
    MIN_LEN = 10
    MAX_LEN = 70

    dx_noise = OpenSimplex(42)
    dx_noise_mult = 0.010
    dy_noise = OpenSimplex(211)
    dy_noise_mult = 0.010
    
    img = cv2.imread(args.pic)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    H, W = gray.shape[:2]
    pos = np.random.randint(0, min(H, W), size=2).astype(np.float64)
    path = [pos.copy()]

    i = 0
    ITER = 50000
    pbar = tqdm.tqdm(total=ITER)
    while i < ITER:
        # direction = np.random.rand(2) - 0.5
        dx = dx_noise.noise2d(x=pos[0]*dx_noise_mult,
                              y=pos[1]*dx_noise_mult) 
        dy = dy_noise.noise2d(x=pos[0]*dy_noise_mult,
                              y=pos[1]*dy_noise_mult) 
        direction = np.array([dy, dx])
        direction /= np.linalg.norm(direction)
        try:
            value = gray[int(np.round(pos[0])),
                         int(np.round(pos[1]))]
        except IndexError:
            value = 255
        length = remap(value, 0, 255, MIN_LEN, MAX_LEN)
        length = MAX_LEN
        new_pos = pos + direction * length
        if np.any(new_pos < 0) or np.any(new_pos >= (H, W)):
            direction = np.array([H/2, W/2]) - pos
            direction /= np.linalg.norm(direction)
        pos += direction * length

        path.append(pos.copy())
        i += 1
        pbar.update()
    pbar.close()

    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    path = np.round(path).astype(np.int32)
    print('path.shape: {}'.format(path.shape))
    # print('path: {}'.format(path))
    for i in range(1, path.shape[0]):
        cv2.line(vis, tuple(path[i-1, ::-1]), tuple(path[i, ::-1]),
                 (0, 0, 255), 1)
        
    # cv2.imshow("cv: img", img)
    # cv2.imshow("cv: gray", gray)
    cv2.imshow("cv: vis", vis)
    while True:
        c = cv2.waitKey(0)
        if c == ord('q'):
            break
    return 0

def main():
    args = parse_arguments()
    return run(args)

if __name__ == '__main__':
    sys.exit(main())
