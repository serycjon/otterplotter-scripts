# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import

import sys
import argparse
import serial
import io
import time

import matplotlib.pyplot as plt

import numpy as np

def parse_arguments():
    parser = argparse.ArgumentParser(description='',
				     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--port', help='', default='/dev/ttyUSB0')
    parser.add_argument('--baud', help='', type=int, default=9600)
    
    return parser.parse_args()

def run(args):
    with serial.Serial(args.port, args.baud, timeout=1) as ser:
        print('ser.port: {}'.format(ser.port))
        print('ser.baudrate: {}'.format(ser.baudrate))
        print('connected to {}'.format(args.port))
        try:
            time.sleep(2)
            # draw_circle(ser, center=(333, 333), radius=50, N_points=3)
            # draw_circle(ser, center=(333, 333), radius=50, N_points=4)
            # draw_circle(ser, center=(333, 333), radius=50, N_points=5)
            # draw_circle(ser, center=(333, 333), radius=50, N_points=6)
            # draw_circle(ser, center=(333, 333), radius=50, N_points=100)

            # draw_grid(ser, (215, 270), (400, 415), linedist_xy=(5, 5)) 

            draw_spiral(ser, (330, 326), r=0, R=70, line_dist=1, N_points_per_round=6)
            # draw_spiral(ser, (330, 326), r=0, R=50, line_dist=1, N_points_per_round=100)
            # draw_spiral(ser, (336, 323), r=0, R=50, line_dist=1)
            # draw_spiral(ser, (336, 329), r=0, R=50, line_dist=1)
            print('done :)')
        except KeyboardInterrupt:
            print('terminating now')
        finally:
            time.sleep(0.3)
            ser.write(b'stop\n')
            time.sleep(0.3)
            ser.write(b'reset\n')
            time.sleep(0.3)
            penup(ser)
            time.sleep(0.3)
            ser.write(b'start\n')
            # time.sleep(0.3)
            # ser.write(b'stop')
            # time.sleep(0.2)
            # ser.write(b'disable')
    return 0

def penup(ser):
    ser.write(b'pu\n')
    time.sleep(0.3)

def pendown(ser):
    ser.write(b'pd\n')
    time.sleep(0.3)

def goto(ser, x, y):
    target_command = 'txy={:03d}{:03d}\n'.format(int(np.round(x)),
                                                int(np.round(y)))
    print(target_command, end='')
    ser.write(str.encode(target_command))

    time.sleep(0.3)
    ser.write(b'start\n')
    # ser.write('blablabla\n'.encode('ascii', 'ignore'))
    # print('start')
    # ser.flush()
    line = ser.readline()
    while line != b"ready\r\n":
        # print('line: {}'.format(line))
        line = ser.readline()

def draw_circle(ser, center, radius, N_points=50):
    pts = []
    for angle in np.linspace(0, 2*np.pi, N_points+1):
        x = center[0] + radius * np.cos(angle)
        y = center[1] + radius * np.sin(angle)
        pts.append((x, y))

    pts = np.array(pts)
    # plt.plot(pts[:, 0], pts[:, 1])
    # plt.show()

    penup(ser)
    for i in range(pts.shape[0]):
        time.sleep(0.1)
        x = pts[i, 0]
        y = pts[i, 1]
        goto(ser, x, y)
        if i==0:
            time.sleep(0.3)
            pendown(ser)

def draw_spiral(ser, center_xy, r, R, line_dist, N_points_per_round=100):
    pts = []
    n_circles = int(np.round((R - r) / line_dist))
    n_pts = n_circles * N_points_per_round
    radii = np.linspace(r, R, n_pts)
    angles = np.linspace(0, 2*np.pi*n_circles, n_pts)

    for radius, angle in zip(radii, angles):
        x = center_xy[0] + radius * np.cos(angle)
        y = center_xy[1] + radius * np.sin(angle)
        pts.append((x, y))

    pts = np.array(pts)
    penup(ser)
    time.sleep(0.1)
    goto(ser, pts[0, 0], pts[0, 1])
    time.sleep(0.3)
    pendown(ser)

    for i in range(pts.shape[0]):
        time.sleep(0.1)
        x = pts[i, 0]
        y = pts[i, 1]
        goto(ser, x, y)

def draw_line(ser, src_xy, dst_xy, initial_penup=True):
    if initial_penup:
        penup(ser)
        time.sleep(0.3)
    time.sleep(0.1)
    goto(ser, src_xy[0], src_xy[1])
    time.sleep(0.1)
    pendown(ser)
    time.sleep(0.3)
    goto(ser, dst_xy[0], dst_xy[1])
    time.sleep(0.1)

def draw_grid(ser, tl_xy, br_xy, linedist_xy):
    x_coords = np.arange(tl_xy[0], br_xy[0], linedist_xy[0])
    y_coords = np.arange(tl_xy[1], br_xy[1], linedist_xy[1])

    # horizontal lines
    for i, y in enumerate(y_coords):
        if i%2 == 0:
            start_x = tl_xy[0]
            stop_x = br_xy[0]
        else:
            start_x = br_xy[0]
            stop_x = tl_xy[0]

        draw_line(ser, (start_x, y), (stop_x, y))

    # vertical lines
    for i, x in enumerate(x_coords):
        if i%2 == 0:
            start_y = tl_xy[1]
            stop_y = br_xy[1]
        else:
            start_y = br_xy[1]
            stop_y = tl_xy[1]

        draw_line(ser, (x, start_y), (x, stop_y))

def main():
    args = parse_arguments()
    return run(args)

if __name__ == '__main__':
    sys.exit(main())
