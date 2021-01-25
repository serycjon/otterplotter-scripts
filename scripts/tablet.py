# -*- coding: utf-8 -*-
import sys
import argparse

import numpy as np
import cv2

from OtterPlotter import Plotter

def parse_arguments():
    parser = argparse.ArgumentParser(description='',
				     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    return parser.parse_args()

def run(args):
    gui = GUI()
    gui.loop()

    return 0

max_dupe_diff = 10

class GUI():
    def __init__(self, W=2970//3, H=2100//3):
        self.canvas = 255 * np.ones((H, W))
        cv2.namedWindow('canvas')
        cv2.setMouseCallback('canvas', self.handler)
        self.down = False
        self.use_servo = True
        self.started = True
        self.tracking = False
        self.last_pos = None
        try:
            self.plotter = Plotter('/dev/ttyUSB0', 115200)
            self.plotter.load_config('config.json')
            self.plotter.set_input_limits((0, 0), (W, 0),
                                          (0, H), (W, H))
        except:
            print('no plotter connected')
            self.plotter = None

    def handler(self, event, x, y, flags, param):
        if not self.started:
            return

        if self.last_pos is None:
            self.last_pos = (x, y)
        
        if event == cv2.EVENT_LBUTTONDOWN:
            self.down = True
            if (self.plotter is not None) and self.use_servo:
                self.plotter.goto(x, y, draw=False, skip_dupes=False)
                print('goto')
                self.plotter.pendown()
            self.last_pos = (x, y)
            print('down')

        elif event == cv2.EVENT_LBUTTONUP:
            self.down = False
            if (self.plotter is not None) and self.use_servo:
                self.plotter.penup()
                cv2.line(self.canvas, self.last_pos, (x, y), 0, 2)
            self.last_pos = (x, y)
            print('up')

        elif event == cv2.EVENT_MBUTTONDOWN:
            self.tracking = True
        elif event == cv2.EVENT_MBUTTONUP:
            self.tracking = False

        elif event == cv2.EVENT_MOUSEMOVE:
            # print(x, y)
            if self.down:
                cv2.line(self.canvas, self.last_pos, (x, y), 0, 2)

            if self.plotter is not None:
                if self.down or self.tracking:
                    self.plotter.goto(x, y, draw=self.down, max_dupe_diff=max_dupe_diff)

            self.last_pos = (x, y)
        else:
            print(f"event: {event}")

    def loop(self):
        while True:
            cv2.imshow("canvas", self.canvas)
            c = cv2.waitKey(20)
            if c == ord('q'):
                break
            elif c == ord(' '):
                self.started = not self.started

        if self.plotter is not None:
            self.plotter.finalize()

def main():
    args = parse_arguments()
    return run(args)

if __name__ == '__main__':
    sys.exit(main())
