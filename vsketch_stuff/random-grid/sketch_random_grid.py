import vsketch
import numpy as np


class RandomGridSketch(vsketch.SketchClass):
    # Sketch parameters:
    # radius = vsketch.Param(2.0)

    N_interp_steps = vsketch.Param(5)
    N_rows = vsketch.Param(10)
    N_cols = vsketch.Param(15)
    noise_pos_mult = vsketch.Param(1.0)
    noise_val_mult = vsketch.Param(1.0)

    def draw(self, vsk: vsketch.Vsketch) -> None:
        vsk.size("a5", landscape=True)
        vsk.scale("cm")

        all_points = np.zeros((self.N_rows, self.N_cols, 2))
        ys = np.linspace(0, 10, self.N_rows)
        xs = np.linspace(0, 15, self.N_cols)
        for row, y in enumerate(ys):
            for col, x in enumerate(xs):
                pt_x = x + self.noise_val_mult * vsk.noise(x * self.noise_pos_mult, y * self.noise_pos_mult)
                pt_y = y + self.noise_val_mult * vsk.noise(xs[-1] + x * self.noise_pos_mult, ys[-1] + y * self.noise_pos_mult)
                # vsk.point(pt_x, pt_y)
                all_points[row, col, :] = (pt_x, pt_y)

        for row in range(self.N_rows - 1):
            for col in range(self.N_cols - 1):
                # horizontal
                if row == self.N_rows - 2:
                    interp_positions = np.linspace(0, 1, self.N_interp_steps + 1, endpoint=True)
                else:
                    interp_positions = np.linspace(0, 1, self.N_interp_steps, endpoint=False)

                for interp_pos in interp_positions:
                    left_x = vsk.lerp(all_points[row, col, 0], all_points[row + 1, col, 0], interp_pos)
                    right_x = vsk.lerp(all_points[row, col + 1, 0], all_points[row + 1, col + 1, 0], interp_pos)
                    left_y = vsk.lerp(all_points[row, col, 1], all_points[row + 1, col, 1], interp_pos)
                    right_y = vsk.lerp(all_points[row, col + 1, 1], all_points[row + 1, col + 1, 1], interp_pos)
                    vsk.line(left_x, left_y, right_x, right_y)

                # vertical
                if col == self.N_cols - 2:
                    interp_positions = np.linspace(0, 1, self.N_interp_steps + 1, endpoint=True)
                else:
                    interp_positions = np.linspace(0, 1, self.N_interp_steps, endpoint=False)

                for interp_pos in interp_positions:
                    top_x = vsk.lerp(all_points[row, col, 0], all_points[row, col + 1, 0], interp_pos)
                    bottom_x = vsk.lerp(all_points[row + 1, col, 0], all_points[row + 1, col + 1, 0], interp_pos)
                    top_y = vsk.lerp(all_points[row, col, 1], all_points[row, col + 1, 1], interp_pos)
                    bottom_y = vsk.lerp(all_points[row + 1, col, 1], all_points[row + 1, col + 1, 1], interp_pos)
                    vsk.line(top_x, top_y, bottom_x, bottom_y)

    def finalize(self, vsk: vsketch.Vsketch) -> None:
        vsk.vpype("linemerge --tolerance 0.2mm linesimplify reloop linesort")


if __name__ == "__main__":
    RandomGridSketch.display()
