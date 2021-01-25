# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import

import os
import sys
import argparse
import numpy as np
import cv2
from skimage.morphology import skeletonize

def parse_arguments():
    parser = argparse.ArgumentParser(description='',
				     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    return parser.parse_args()

def run(args):
    srdce = cv2.imread(os.path.join('srdce',
                                    'SKM_C224e20021214210_BW.png'),
                       0)
    skeleton = skeletonize(1 - (srdce/255))
    skeleton = 1 - np.float32(skeleton)
    print('srdce.shape: {}'.format(srdce.shape))
    cv2.imshow("cv: srdce", srdce)
    cv2.imshow("cv: skeleton", skeleton)
    c = cv2.waitKey(0)
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
