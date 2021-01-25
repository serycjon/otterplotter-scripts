# -*- coding: utf-8 -*-
import sys
import argparse

import numpy as np
import cv2
from scipy import signal

import tqdm
import matplotlib.pyplot as plt


def parse_arguments():
    parser = argparse.ArgumentParser(description='',
				     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--img', help='path to image', required=True)
    parser.add_argument('--sub_mode', help='canvas subtraction mode', choices=['line', 'endpoint'], default='line')
    parser.add_argument('--sense', help='sensing mode', choices=['connected', 'pool'], default='pool')
    parser.add_argument('--nosave', '-N', help='don\'t save', action='store_true')
    parser.add_argument('--N_lines', help='number of walks', type=int, default=800)
    parser.add_argument('--sub', help='canvas subtraction amount', type=float, default=0.02)
    parser.add_argument('--wsr', help='walk stop ratio', type=float, default=0.75)
    return parser.parse_args()

def run(args):
    img = cv2.imread(args.img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    scale = 0.3
    gray = cv2.resize(gray, dsize=(0, 0), fx=scale, fy=scale)
    H, W = gray.shape

    to_cover = np.float32(gray) / 255
    to_cover = 1 - to_cover

    sensing_window_sz = 9
    sensing_kernel_sz = 3
    thickness = 3
    subtract = args.sub
    step_size = 2
    all_lines = []
    N_lines = args.N_lines
    max_walk = 400
    walk_stop_ratio = args.wsr
    for line_i in tqdm.tqdm(range(N_lines)):
        ind = np.unravel_index(np.argmax(to_cover, axis=None),
                               to_cover.shape)
        pos_xy = np.array((ind[1], ind[0])).astype(np.float64)
        vel_xy = np.array((0, 0))

        line = [pos_xy.copy()]
        to_cover[int(pos_xy[1]), int(pos_xy[0])] -= subtract
        last_best_val = -float('inf')
        for step_i in range(max_walk):
            if args.sense == 'connected':
                dir_xy, best_val = sense_connected(to_cover, pos_xy + vel_xy)
            else:
                dir_xy, best_val = sense_pool(to_cover, pos_xy + vel_xy,
                                              window_sz=sensing_window_sz,
                                              kernel_sz=sensing_kernel_sz)
            if best_val <= 0:
                break
            if best_val < last_best_val * walk_stop_ratio:
                break
            last_best_val = best_val

            vel_xy = vel_xy + dir_xy
            old_xy = pos_xy.copy()
            pos_xy += vel_xy
            if args.sub_mode == 'line':
                subtraction_canvas = np.zeros_like(to_cover)
                cv2.line(subtraction_canvas,
                         (int(np.round(old_xy[0])), int(np.round(old_xy[1]))),
                         (int(np.round(pos_xy[0])), int(np.round(pos_xy[1]))),
                         subtract, thickness)
                to_cover -= subtraction_canvas
                line.append(pos_xy.copy())
                lookup_xy = np.round(pos_xy).astype(np.int32)
                if in_bounds(lookup_xy, to_cover):
                    pass
                else:
                    break
            else:
                lookup_xy = np.round(pos_xy).astype(np.int32)
                if in_bounds(lookup_xy, to_cover):
                    to_cover[lookup_xy[1], lookup_xy[0]] -= subtract
                    line.append(pos_xy.copy())
                else:
                    break
        
        line = np.array(line)
        all_lines.append(line)
        plt.plot(line[:, 0], line[:, 1], 'k-', lw=0.1)
    plt.tight_layout()
    plt.gca().set_aspect('equal', 'box')
    plt.gca().invert_yaxis()
    # from datetime import datetime
    # time_string = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    # plt.savefig(f'results/{time_string}.png', bbox_inches='tight', pad_inches=0,
    #             dpi=600)
    from repro import ReproSaver
    saver = ReproSaver()
    saver.seed_numpy()
    if not args.nosave:
        saver.add_plt_image()
    plt.figure()
    plt.imshow(to_cover)
    plt.show()
    return 0

def sense_connected(to_cover, pos_xy):
    H, W = to_cover.shape
    dirs_xy = np.array([(1, 0), (1, 1), (0, 1), (-1, 1),
                        (-1, 0), (-1, -1), (0, -1), (1, -1)])
    lookup_xy = np.round(pos_xy[np.newaxis, :] + dirs_xy).astype(np.int32)
    in_bounds = np.logical_and(lookup_xy >= 0,
                               lookup_xy < np.array(((W, H))))
    in_bounds = np.all(in_bounds, axis=1)
    lookup_xy = lookup_xy[in_bounds, :]
    dirs_xy = dirs_xy[in_bounds, :]
    values = to_cover[lookup_xy[:, 1], lookup_xy[:, 0]]

    bias_direction = np.array([1.0, 1.0])
    bias_direction /= np.linalg.norm(bias_direction)
    additional_score = -np.abs(np.sum(dirs_xy * bias_direction[np.newaxis, :],
                                      axis=1))
    values = values + 0.05*additional_score
    if len(values) == 0:
        return None, -float('inf')
    best = np.argmax(values)
    best_val = values[best]
    dir_xy = dirs_xy[best, :]

    return dir_xy, best_val

def sense_pool(to_cover, pos_xy,
               window_sz, kernel_sz):
    round_pos = np.round(pos_xy).astype(np.int32)

    # sample window around round_pos from to_cover (with 0 padding)
    assert window_sz % 2 != 0
    tl_xy = round_pos - (window_sz - 1) // 2
    br_xy = round_pos + (window_sz - 1) // 2

    H, W = to_cover.shape

    top_pad = max(-tl_xy[1], 0)
    left_pad = max(-tl_xy[0], 0)
    bottom_pad = max(br_xy[1] - (H-1), 0)
    right_pad = max(br_xy[0] - (W-1), 0)

    # these should be in bounds
    src_tl_xy = (tl_xy + np.array((left_pad, top_pad))).astype(np.int32)
    src_br_xy = (br_xy - np.array((right_pad, bottom_pad))).astype(np.int32)
    if not in_bounds(src_tl_xy, to_cover) or not in_bounds(src_br_xy, to_cover):
        return None, -float('inf')

    window = np.zeros((window_sz, window_sz), dtype=np.float32)
    # dst = window[top_pad:(window_sz - bottom_pad), left_pad:(window_sz - right_pad)]
    # src = to_cover[src_tl_xy[1]:src_br_xy[1]+1,
    #                src_tl_xy[0]:src_br_xy[0]+1]
    window[top_pad:(window_sz - bottom_pad),
           left_pad:(window_sz - right_pad)] = to_cover[src_tl_xy[1]:src_br_xy[1]+1,
                                                        src_tl_xy[0]:src_br_xy[0]+1]

    kernel = np.ones((kernel_sz, kernel_sz)) / kernel_sz**2
    sensed = signal.convolve2d(window, kernel, mode='valid') 
    ind = np.unravel_index(np.argmax(sensed, axis=None),
                           sensed.shape)
    sensed_value = sensed[ind[0], ind[1]]
    sensed_xy = np.array((ind[1], ind[0]))
    window_xy = sensed_xy + (kernel_sz - 1) / 2  # valid convolution padding
    window_center_xy = np.array(((window_sz - 1) / 2, (window_sz - 1) / 2))
    direction_xy = window_xy - window_center_xy
    return direction_xy, sensed_value


def in_bounds(pos_xy, canvas):
    return np.all(pos_xy >= 0) and np.all(pos_xy < np.array(canvas.shape[::-1]))


def main():
    args = parse_arguments()
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
