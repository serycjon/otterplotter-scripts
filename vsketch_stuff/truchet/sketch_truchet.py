import numpy as np
import vsketch
from shapely.geometry import Polygon, LineString, Point
from shapely.errors import TopologicalError
import vpype
import vpype_cli

import traceback
import ipdb


def with_debugger(orig_fn):
    def new_fn(*args, **kwargs):
        try:
            return orig_fn(*args, **kwargs)
        except Exception as e:
            print(traceback.format_exc())
            print(e)
            ipdb.post_mortem()

    return new_fn


class TruchetSketch(vsketch.SketchClass):
    # Sketch parameters:
    # radius = vsketch.Param(2.0)
    show_patterns = vsketch.Param(False)
    show_grid = vsketch.Param(False)
    show_boundary = vsketch.Param(False)
    separate_long = vsketch.Param(False)
    use_probs = vsketch.Param(False)
    use_custom_probs = vsketch.Param(False)
    pattern_size = vsketch.Param(5.0)
    N_rows = vsketch.Param(20)
    N_cols = vsketch.Param(32)
    center_x = vsketch.Param(0)
    center_y = vsketch.Param(0)
    max_dist = vsketch.Param(30)
    sigma = vsketch.Param(1.0)
    N = vsketch.Param(3)
    prob_1 = vsketch.Param(0.2)
    prob_2 = vsketch.Param(0.2)
    prob_3 = vsketch.Param(0.2)
    prob_4 = vsketch.Param(0.2)
    prob_5 = vsketch.Param(0.2)
    prob_6 = vsketch.Param(0.2)

    @with_debugger
    def draw(self, vsk: vsketch.Vsketch) -> None:
        vsk.size("a5", landscape=True)
        vsk.scale("mm")

        a = self.pattern_size

        patterns = patterns_v4_simple_param(a, self.N)
        sqr = square(0, 0, a, a)

        pattern_lengths = [np.sum([path_length(path) for path in pat])
                           for pat in patterns]
        pattern_ids = np.argsort(pattern_lengths)
        if self.use_probs:
            patterns = [patterns[idx] for idx in pattern_ids]
        transformations = [
            rotate_pattern(a, 0), rotate_pattern(a, 90),
            rotate_pattern(a, 180), rotate_pattern(a, 270),
            rotate_pattern(a, 0, swap=True), rotate_pattern(a, 90, swap=True),
            rotate_pattern(a, 180, swap=True), rotate_pattern(a, 270, swap=True)]

        H = self.N_rows
        W = self.N_cols
        sun_drawing = []
        drawing = []
        grid_drawing = []
        if not self.show_patterns:
            for row in range(H):
                for col in range(W):
                    if self.use_probs:
                        dist = np.sqrt((row - self.center_y)**2 + (col - self.center_x)**2)
                        probs = discrete_trunc_normal_distribution(len(patterns),
                                                                   np.clip(remap(float(dist),
                                                                                 0, self.max_dist,
                                                                                 0, len(patterns)),
                                                                           0, len(patterns)),
                                                                   self.sigma)
                    elif self.use_custom_probs:
                        custom_probs = [self.prob_1, self.prob_2, self.prob_3, self.prob_4, self.prob_5, self.prob_6]
                        probs = np.zeros(len(patterns))
                        for i in range(len(patterns)):
                            if i >= len(custom_probs):
                                break
                            probs[i] = custom_probs[i]
                        probs /= np.sum(probs)
                        
                    else:
                        probs = None

                    selected_pattern_id = np.random.choice(len(patterns), p=probs)
                    pattern = patterns[selected_pattern_id]
                    selected_transformation_id = np.random.choice(len(transformations))
                    # selected_transformation_id = 0
                    transformation = transformations[selected_transformation_id]
                    if False and selected_pattern_id == 0:
                        sun_drawing.extend(pattern_to_grid(transformation(pattern), a, row, col))
                    else:
                        drawing.extend(pattern_to_grid(transformation(pattern), a, row, col))
                    if self.show_grid:
                        grid_drawing.extend(pattern_to_grid([sqr], a, row, col))
            if self.show_boundary:
                grid_drawing.append(square(0, 0, W * a, H * a))
                pass
        else:
            for row in range(H):
                for col in range(W):
                    pattern_id = col + row * W
                    if pattern_id >= len(patterns):
                        break
                    pattern = patterns[pattern_id]

                    drawing.extend(pattern_to_grid(pattern, 1.1 * a, row, col))
                    drawing.extend(pattern_to_grid([sqr], 1.1 * a, row, col))
        # vsk.polygon(sqr)

        filtered_document = vpype.Document(to_vpype_lines(drawing))
        # min_length = a / (self.N * 7.1)
        # filtered_document = vpype_cli.execute(f"filter --min-length {min_length}mm", filtered_document)
        processed_document = vpype_cli.execute("linesort linemerge",
                                               filtered_document)
        processed = from_vpype_document(processed_document)
        lengths = [path_length(path) for path in processed]
        ordering = list(reversed(np.argsort(lengths)))
        for i, path_i in enumerate(ordering):
            path = processed[path_i]
            if self.separate_long and i < 1:
                vsk.stroke(3)
            elif self.separate_long and i < 2:
                vsk.stroke(2)
            else:
                vsk.stroke(1)
            vsk.polygon(path)

        for path in grid_drawing:
            vsk.stroke(4)
            vsk.polygon(path)

    def finalize(self, vsk: vsketch.Vsketch) -> None:
        vsk.vpype("linemerge linesimplify reloop linesort")


def circle(c_xy, r, N=100, angle_start=0, angle_end=360):
    angles = np.linspace(angle_start, angle_end, N)
    sins = np.sin(np.radians(angles))
    cosins = np.cos(np.radians(angles))

    points = np.zeros((N, 2))
    points[:, 0] = cosins * r + c_xy[0]
    points[:, 1] = sins * r + c_xy[1]

    return points


def path_length(path):
    pos = path[0]
    length = 0
    for point in path:
        length += np.linalg.norm(point - pos)
        pos = point
    return length


def square(tl_x, tl_y, w, h):
    pts = [(tl_x, tl_y), (tl_x + w, tl_y),
           (tl_x + w, tl_y + h), (tl_x, tl_y + h),
           (tl_x, tl_y)]
    return np.array(pts)


def rotate_pattern(size, angle, swap=False):
    def transform(pattern):
        rotated = rotate(pattern, angle, center=(size / 2, size / 2))
        if not swap:
            return rotated
        else:
            result = []
            for path in rotated:
                result.append(path[:, ::-1])
            return result
    return transform


def rotate(drawing, degrees, center=(0, 0)):
    R = rotation_matrix(degrees)

    rotated = []
    for line in drawing:
        centered = line - np.array(center).reshape(1, 2)
        rotated_line = np.matmul(R, centered.T).T
        uncentered = rotated_line + np.array(center).reshape(1, 2)

        rotated.append(uncentered)
    return rotated


def rotation_matrix(degrees):
    s, c = np.sin(np.radians(degrees)), np.cos(np.radians(degrees))
    R = np.array([[c, -s],
                  [s, c]])
    return R


def pattern_to_grid(pattern, a, row, col):
    x_shift = a * col
    y_shift = a * row
    shift = np.array((x_shift, y_shift)).reshape(1, 2)

    result = []
    for path in pattern:
        result.append(path + shift)
    return result


def mask_drawing(drawing, mask_poly, invert=False,
                 buffers=None, probs=None):
    shapely_mask = Polygon(mask_poly)
    masks = [shapely_mask]
    if buffers is not None:
        masks = [shapely_mask.buffer(buff) for buff in buffers]

    def mask_layer(drawing):
        result = []
        if buffers is not None:
            mask = np.random.choice(masks, size=1,
                                    replace=False,
                                    p=probs)[0]
        else:
            mask = shapely_mask
        for path in drawing:
            shapely_path = LineString(path)

            try:
                if invert:
                    intersection = shapely_path.difference(mask)
                else:
                    intersection = shapely_path.intersection(mask)
            except TopologicalError:
                continue
            if type(intersection) is LineString:
                intersection = [intersection]  # simulate multilinestring
            elif type(intersection) is Point:
                continue

            for thing in intersection:
                if type(thing) is LineString:
                    masked_line = np.array(thing.coords)
                    if len(masked_line) > 0:
                        result.append(np.array(thing.coords))

        return result

    result = mask_layer(drawing)
    return result


def to_vpype_lines(paths):
    lc = vpype.LineCollection()
    for path in paths:
        lc.append(path[:, 0] + path[:, 1] * 1.j)
    return lc


def from_vpype_document(doc):
    def from_vpype_lines(lines):
        results = []
        for line in lines:
            xs = np.real(line)
            ys = np.imag(line)
            results.append(np.stack((xs, ys), axis=1))

        return results
    assert len(doc.ids()) == 1
    for layer_id in doc.ids():
        paths = from_vpype_lines(doc[layer_id])
        return paths


def discrete_trunc_normal_distribution(N, mean_pos, sigma):
    xs = np.arange(N).astype(np.float64)
    xs -= mean_pos
    pdf = np.exp(-0.5 * np.square(xs / sigma))
    pdf /= np.sum(pdf)
    if np.any(np.isnan(pdf)):
        return discrete_trunc_normal_distribution(N, mean_pos, sigma * 1.2)
    return pdf


def remap(x,
          src_min, src_max,
          dst_min, dst_max):
    x_01 = (x - src_min) / float(src_max - src_min)
    x_dst = x_01 * (dst_max - dst_min) + dst_min

    return x_dst


def patterns_v1(a):
    # pattern 1
    pat_1 = []
    pat_1.append(circle((0, 0), a / 4))
    pat_1.append(circle((0, 0), 2 * a / 4))
    big_circle = circle((0, 0), 3 * a / 4)
    pat_1.append(big_circle)
    pat_1.append(circle((0, a), a / 4))
    pat_1.extend(mask_drawing([circle((0, a), 2 * a / 4)], big_circle, invert=True))
    pat_1.extend(mask_drawing([circle((a, 0), a / 4)], big_circle, invert=True))
    pat_1.extend(mask_drawing([circle((a, 0), 2 * a / 4)], big_circle, invert=True))
    pat_1.extend(mask_drawing([circle((a, 0), 3 * a / 4)], big_circle, invert=True))
    pat_1.append(circle((a, a), a / 4))
    sqr = square(0, 0, a, a)
    pat_1 = mask_drawing(pat_1, sqr)

    # pattern 2
    pat_2 = []
    pat_2.append(circle((0, 0), a / 4))
    pat_2.append(circle((0, 0), 2 * a / 4))
    big_circle = circle((0, 0), 3 * a / 4)
    pat_2.append(big_circle)
    pat_2.append(circle((0, a), a / 4))
    pat_2.extend(mask_drawing([circle((a * 5 / 8, a), a / 8)], big_circle, invert=True))
    pat_2.append(circle((a, 0), a / 4))
    pat_2.extend(mask_drawing([circle((a, a * 5 / 8), a / 8)], big_circle, invert=True))
    pat_2.extend(mask_drawing([circle((0, a), 2 * a / 4)], big_circle, invert=True))
    sqr = square(0, 0, a, a)
    pat_2 = mask_drawing(pat_2, sqr)

    # pattern 3
    pat_3 = []
    pat_3.append(circle((0, 0), a / 4))
    pat_3.append(circle((0, 0), 2 * a / 4))
    big_circle = circle((0, 0), 3 * a / 4)
    pat_3.append(big_circle)
    pat_3.append(circle((0, a), a / 4))
    pat_3.extend(mask_drawing([circle((a * 5 / 8, a), a / 8)], big_circle, invert=True))
    pat_3.append(circle((a, 0), a / 4))
    pat_3.extend(mask_drawing([circle((a, a * 5 / 8), a / 8)], big_circle, invert=True))
    pat_3.append(circle((a, a), a / 4))
    sqr = square(0, 0, a, a)
    pat_3 = mask_drawing(pat_3, sqr)

    # pattern 4
    pat_4 = []
    pat_4.append(circle((0, 0), a / 4))
    pat_4.append(circle((0, 0), 2 * a / 4))
    big_circle = circle((0, 0), 3 * a / 4)
    pat_4.append(big_circle)
    pat_4.append(circle((a, a), a / 4))
    pat_4.append(circle((a, a), 2 * a / 4))
    pat_4.extend(mask_drawing([circle((a, a), 3 * a / 4)], big_circle, invert=True))
    pat_4.extend(mask_drawing([circle((a * 5 / 8, a), a / 8)], big_circle, invert=True))
    pat_4.extend(mask_drawing([circle((a, a * 5 / 8), a / 8)], big_circle, invert=True))
    sqr = square(0, 0, a, a)
    pat_4 = mask_drawing(pat_4, sqr)

    # pattern 5
    pat_5 = []
    pat_5.append(circle((0, 0), a / 4))
    pat_5.append(circle((0, 0), 2 * a / 4))
    big_circle = circle((0, 0), 3 * a / 4)
    pat_5.append(big_circle)
    pat_5.append(circle((a, a), a / 4))
    pat_5.append(circle((a, a), 2 * a / 4))
    pat_5.extend(mask_drawing([circle((a, 0), 2 * a / 4)], big_circle, invert=True))
    pat_5.extend(mask_drawing([circle((a, 0), a / 4)], big_circle, invert=True))

    pat_5.extend(mask_drawing([circle((0, a), 2 * a / 4)], big_circle, invert=True))
    pat_5.extend(mask_drawing([circle((0, a), a / 4)], big_circle, invert=True))
    sqr = square(0, 0, a, a)
    pat_5 = mask_drawing(pat_5, sqr)

    # pattern 6
    pat_6 = []
    pat_6.append(circle((0, 0), a / 4))
    pat_6.append(circle((0, 0), 2 * a / 4))
    big_circle = circle((0, 0), 3 * a / 4)
    pat_6.append(big_circle)
    pat_6.append(circle((0, a), a / 4))
    pat_6.extend(mask_drawing([circle((a * 5 / 8, a), a / 8)], big_circle, invert=True))
    pat_6.append(circle((a, 0), a / 4))
    pat_6.extend(mask_drawing([circle((a, a * 5 / 8), a / 8)], big_circle, invert=True))
    pat_6.append(circle((a, a), a / 4))
    pat_6.append(circle((11 / 16 * a, 11 / 16 * a), a / 8))
    sqr = square(0, 0, a, a)
    pat_6 = mask_drawing(pat_6, sqr)

    # pattern 7
    pat_7 = []
    pat_7.append(circle((0, 0), a / 4))
    pat_7.append(circle((0, 0), 2 * a / 4))
    big_circle = circle((0, 0), 3 * a / 4)
    pat_7.append(big_circle)
    pat_7.append(circle((0, a), a / 4))
    pat_7.extend(mask_drawing([circle((0, a), 2 * a / 4)], big_circle, invert=True))
    pat_7.extend(mask_drawing([circle((a, 0), a / 4)], big_circle, invert=True))
    pat_7.extend(mask_drawing([circle((a, 0), 2 * a / 4)], big_circle, invert=True))
    pat_7.extend(mask_drawing([circle((a, 0), 3 * a / 4)], big_circle, invert=True))
    pat_7.append(circle((0, 3 * a / 8), a / 8))
    pat_7.append(circle((5 * a / 8, 0), a / 8))
    pat_7.append(circle((a, a), a / 4))
    sqr = square(0, 0, a, a)
    pat_7 = mask_drawing(pat_7, sqr)

    patterns = [pat_1, pat_2, pat_3, pat_4, pat_5, pat_6, pat_7]
    return patterns


def patterns_v2(a):
    # pattern 1
    pat_1 = []
    pat_1.append(circle((0, 0), a / 4))
    pat_1.append(circle((0, 0), 2 * a / 4))
    big_circle = circle((0, 0), 3 * a / 4)
    pat_1.append(big_circle)
    pat_1.append(circle((0, a), a / 4))
    pat_1.extend(mask_drawing([circle((0, a), 2 * a / 4)], big_circle, invert=True))
    pat_1.extend(mask_drawing([circle((a, 0), a / 4)], big_circle, invert=True))
    pat_1.extend(mask_drawing([circle((a, 0), 2 * a / 4)], big_circle, invert=True))
    pat_1.extend(mask_drawing([circle((a, 0), 3 * a / 4)], big_circle, invert=True))
    pat_1.append(circle((a, a), a / 4))
    sqr = square(0, 0, a, a)
    pat_1 = mask_drawing(pat_1, sqr)

    # pat_2
    pat_2 = []
    pat_2.append(np.array([[1 / 4 * a, 0], [1 / 4 * a, a]]))
    pat_2.append(np.array([[1 / 4 * a, 1 / 4 * a], [a, 1 / 4 * a]]))
    pat_2.append(np.array([[1 / 4 * a, 2 / 4 * a], [a, 2 / 4 * a]]))
    pat_2.append(np.array([[1 / 4 * a, 3 / 4 * a], [a, 3 / 4 * a]]))
    pat_2.append(circle((0, 0), a / 4))
    pat_2.append(circle((0, 5 / 8 * a), a / 8))
    pat_2.append(circle((5 / 8 * a, 0), a / 8))
    pat_2.append(circle((5 / 8 * a, a), a / 8))
    # pat_2.append(np.array([[3 / 4 * a, 0], [3 / 4 * a, a]]))
    pat_2 = mask_drawing(pat_2, sqr)

    # pat_3
    pat_3 = []
    # pat_3.append(np.array([[2 / 4 * a, 0], [2 / 4 * a, a]]))
    # pat_3.append(np.array([[0, 2 / 4 * a], [a, 2 / 4 * a]]))
    # pat_3.append(circle((0, 0), a / 4))
    # pat_3.append(circle((a, 0), a / 4))
    # pat_3.append(circle((a, a), a / 4))
    # pat_3.append(circle((0, a), a / 4))
    Q = np.sqrt(2) / 2
    pat_3.append(np.array([[Q * (a / 4), Q * (a / 4)],
                           [a - Q * (a / 4), a - Q * (a / 4)]]))
    # pat_3.append(np.array([[0, 2 / 4 * a], [a, 2 / 4 * a]]))
    pat_3.append(circle((0, 0), a / 4))
    pat_3.append(circle((a, 0), a / 4))
    pat_3.append(circle((a, a), a / 4))
    pat_3.append(circle((0, a), a / 4))

    pat_3.append(circle((a, 0), 2 * a / 4))
    pat_3.append(circle((0, a), 2 * a / 4))
    pat_3 = mask_drawing(pat_3, sqr)

    # pat_4
    pat_4 = []
    pat_4.append(circle((0, 0), a / 4))
    pat_4.append(circle((a, 0), a / 4))
    pat_4.append(circle((a, a), a / 4))
    pat_4.append(circle((0, a), a / 4))

    pat_4.append(circle((0, 0), 2 * a / 4))
    pat_4.append(circle((a, 0), 2 * a / 4))
    pat_4.append(circle((a, a), 2 * a / 4))
    pat_4.append(circle((0, a), 2 * a / 4))

    pat_4.append(circle((a / 2, a / 2), a / 6))
    pat_4 = mask_drawing(pat_4, sqr)

    patterns = [pat_1, pat_2, pat_3, pat_4]
    return patterns


def patterns_v3_ana(a):
    sqr = square(0, 0, a, a)

    # pattern 1
    pat_1 = []
    pat_1.append(circle((0, 0), a / 4))
    pat_1.append(circle((a, 0), a / 4))
    pat_1.append(circle((a, a), a / 4))
    pat_1.append(circle((0, a), a / 4))

    pat_1.append(circle((0, 0), a / 2))
    pat_1.append(circle((a, 0), a / 2))
    pat_1.append(circle((a, a), a / 2))
    pat_1.append(circle((0, a), a / 2))
    pat_1 = mask_drawing(pat_1, sqr)

    pat_2 = []
    pat_2.append(circle((0, 5 / 8 * a), a / 8))
    pat_2.append(np.array([[0, a / 4], [a / 4, 0]]))
    pat_2.append(np.array([[a / 4, a], [a / 2, 0]]))
    pat_2.append(circle((5 / 8 * a, a), a / 8))
    pat_2.append(np.array([[3 * a / 4, a], [3 * a / 4, 0]]))
    pat_2.append(circle((a, a / 2), a / 4))
    pat_2.append(circle((a, 5 / 8 * a), a / 8))
    pat_2.append(circle((a, 3 / 8 * a), a / 8))
    pat_2 = mask_drawing(pat_2, sqr)

    pat_3 = []
    pat_3.append(circle((0, 0), a / 4))
    pat_3.append(circle((a, a), a / 4))
    pat_3.append(np.array([[0, a / 4], [3 * a / 4, a]]))
    pat_3.append(np.array([[0, 2 * a / 4], [2 * a / 4, a]]))
    pat_3.append(np.array([[0, 3 * a / 4], [a / 4, a]]))
    pat_3.append(np.array([[a / 4, 0], [a, 3 * a / 4]]))
    pat_3.append(np.array([[2 * a / 4, 0], [a, 2 * a / 4]]))
    pat_3.append(np.array([[3 * a / 4, 0], [a, 1 * a / 4]]))
    pat_3 = mask_drawing(pat_3, sqr)

    pat_4 = []
    pat_4.append(circle((a / 2, 0), a / 4))
    pat_4.append(circle((a / 2, a), a / 4))
    pat_4.append(np.array([[a / 2, 0], [a / 2, a]]))
    pat_4.append(np.array([[0, a / 4], [a, 3 / 4 * a]]))
    pat_4.append(np.array([[0, 3 * a / 4], [a, 1 / 4 * a]]))
    pat_4.append(circle((a, 3 * a / 8), a / 8))
    pat_4.append(circle((a, 5 * a / 8), a / 8))
    pat_4.append(circle((0, 3 * a / 8), a / 8))
    pat_4.append(circle((0, 5 * a / 8), a / 8))
    pat_4 = mask_drawing(pat_4, sqr)

    pat_5 = []
    pat_5.append(np.array([[a / 4, 0], [0, a / 4]]))
    pat_5.append(np.array([[a / 4, 0], [0, 2 * a / 4]]))
    pat_5.append(np.array([[a / 4, 0], [0, 3 * a / 4]]))
    pat_5.append(np.array([[2 * a / 4, 0], [0, 3 * a / 4]]))
    pat_5.append(np.array([[0, 3 / 4 * a], [a / 4, a]]))

    pat_5.append(np.array([[3 / 4 * a, 0], [a, a / 4]]))
    pat_5.append(np.array([[3 / 4 * a, 0], [a, 2 * a / 4]]))
    pat_5.append(np.array([[3 / 4 * a, 0], [a, 3 * a / 4]]))
    pat_5.append(np.array([[3 / 4 * a, 0], [0, 3 * a / 4]]))

    pat_5.append(np.array([[2 / 4 * a, a], [a, 3 * a / 4]]))
    pat_5.append(np.array([[3 / 4 * a, a], [a, 3 * a / 4]]))
    pat_5.append(np.array([[1 / 4 * a, a], [a, 3 * a / 4]]))

    pat_5 = mask_drawing(pat_5, sqr)

    pat_6 = []
    pat_6.append(circle((0, 0), a / 4))
    pat_6.append(circle((0, 0), 2 * a / 4))
    pat_6.append(circle((0, 0), 3 * a / 4))
    pat_6.append(circle((a, a), a / 4))
    pat_6.append(circle((a, a), 2 * a / 4))
    pat_6.append(circle((a, a), 3 * a / 4))
    pat_6.append(np.array([[3 * a / 4, 0], [a, a / 4]]))
    pat_6.append(np.array([[0, 3 * a / 4], [a / 4, a]]))

    pat_6 = mask_drawing(pat_6, sqr)

    pat_7 = []
    pat_7.append(circle((0, 3 / 8 * a), a / 8))
    pat_7.append(circle((a, 5 / 8 * a), a / 8))
    pat_7.append(np.array([[0, 3 * a / 4], [a / 4, 0]]))
    pat_7.append(np.array([[0, 3 * a / 4], [2 * a / 4, 0]]))
    pat_7.append(np.array([[0, 3 * a / 4], [3 * a / 4, 0]]))
    pat_7.append(np.array([[a, a / 4], [3 * a / 4, a]]))
    pat_7.append(np.array([[a, a / 4], [2 * a / 4, a]]))
    pat_7.append(np.array([[a, a / 4], [1 * a / 4, a]]))
    pat_7 = mask_drawing(pat_7, sqr)

    pat_8 = []
    pat_8.append(circle((0, 0), a / 2))
    pat_8.append(circle((a, 0), a / 2))
    pat_8.append(np.array([[0, a / 4], [a / 4, 0]]))
    pat_8.append(np.array([[0, 3 * a / 4], [a / 4, a]]))
    pat_8.append(np.array([[3 * a / 4, 0], [a, a / 4]]))
    pat_8.append(np.array([[3 * a / 4, a], [a, 3 * a / 4]]))
    pat_8.append(np.array([[a / 2, 0], [a / 2, a]]))
    pat_8.append(np.array([[a / 2, 0], [3 * a / 4, a]]))
    pat_8.append(np.array([[a / 2, 0], [1 * a / 4, a]]))
    pat_8 = mask_drawing(pat_8, sqr)

    pat_9 = []
    pat_9.append(np.array([[0, a / 4], [a / 4, 0]]))
    pat_9.append(np.array([[0, 3 * a / 4], [a / 4, a]]))
    pat_9.append(np.array([[3 * a / 4, 0], [a, a / 4]]))
    pat_9.append(np.array([[3 * a / 4, a], [a, 3 * a / 4]]))
    pat_9.append(np.array([[0, a / 2], [a / 4, 0]]))
    pat_9.append(np.array([[3 * a / 4, 0], [a, a / 2]]))
    pat_9.append(np.array([[0, 3 * a / 4], [a / 2, 0]]))
    pat_9.append(np.array([[a, 3 * a / 4], [a / 2, 0]]))
    pat_9.append(np.array([[a / 2, 0], [3 * a / 4, a]]))
    pat_9.append(np.array([[a / 2, 0], [1 * a / 4, a]]))
    pat_9.append(circle((0, 3 / 8 * a), a / 8))
    pat_9.append(circle((a, 3 / 8 * a), a / 8))
    pat_9.append(circle((a / 2, a), a / 4))
    pat_9 = mask_drawing(pat_9, sqr)

    patterns = [pat_1, pat_2, pat_3, pat_4, pat_5, pat_6, pat_7, pat_8, pat_9]
    return patterns

def patterns_v4_simple_param(a, N):
    sqr = square(0, 0, a, a)

    # pattern 1
    pat_1 = []
    width = a / N
    rs = np.linspace(width / 2, a - width / 2, N)
    for r in rs:
        pat_1.append(circle((0, 0), r))

    masking_circle = circle((0, 0), rs[-1])

    for r in rs:
        pat_1.extend(mask_drawing([circle((a, a), r)], masking_circle, invert=True))

    pat_1 = mask_drawing(pat_1, sqr)

    pat_2 = []
    rs = rs.tolist()
    # rs.append(a)
    rect_mask = np.array([[rs[0], -1], [rs[0], a + 1],
                          [rs[-1], a + 1], [rs[-1], -1],
                          [rs[0], -1]])
    for r in rs:
        pat_2.append(np.array([[r, 0], [r, a]]))
        pat_2.extend(mask_drawing([np.array([[0, r], [a, r]])], rect_mask, invert=True))


    patterns = [pat_1, pat_2]
    return patterns

if __name__ == "__main__":
    TruchetSketch.display()
