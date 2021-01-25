# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import

import tqdm
import numpy as np
import matplotlib.pyplot as plt
import pickle
import json
from shapely.geometry import Polygon, LineString
from shapely.errors import TopologicalError


def circle(c_xy, r, N=100, angle_start=0, angle_end=360):
    angles = np.linspace(angle_start, angle_end, N)
    sins = np.sin(np.radians(angles))
    cosins = np.cos(np.radians(angles))

    points = np.zeros((N, 2))
    points[:, 0] = cosins * r + c_xy[0]
    points[:, 1] = sins * r + c_xy[1]

    return points


def mask_drawing(drawing, mask_poly):
    shapely_mask = Polygon(mask_poly)

    def mask_layer(drawing):
        result = []
        for path in drawing:
            shapely_path = LineString(path)

            try:
                intersection = shapely_path.intersection(shapely_mask)
            except TopologicalError:
                continue
            if type(intersection) is LineString:
                intersection = [intersection]  # simulate multilinestring

            for thing in intersection:
                if type(thing) is LineString:
                    masked_line = np.array(thing.coords)
                    if len(masked_line) > 0:
                        result.append(np.array(thing.coords))

        return result

    result = apply_per_layer(drawing, mask_layer)
    return result


def apply_per_layer(multilayer_drawing, fn, ensure_layer=False):
    if type(multilayer_drawing) is not dict:
        single_layer = True
        layers = {'layer': multilayer_drawing}
    else:
        single_layer = False
        layers = multilayer_drawing

    result = {layer_name: fn(layer)
              for layer_name, layer in layers.items()}

    if single_layer and not ensure_layer:
        return result['layer']
    else:
        return result


def merge_layers(multilayer_drawing):
    if type(multilayer_drawing) is not dict:
        return multilayer_drawing

    merged = []
    for layer_name, layer in multilayer_drawing.items():
        merged.extend(layer)
    return merged


def rounded_rect(xywh, r, N_seg=10):
    tl_x, tl_y, w, h = xywh
    assert w > 2 * r
    assert h > 2 * r

    # top-right
    pts = []
    # pts = [(tl_x + r, tl_y), (tl_x + w - r, tl_y)]
    corner = circle((tl_x + w - r, tl_y + r), r, N=N_seg, angle_start=-90, angle_end=0)
    pts.extend(corner.tolist())

    # right-bottom
    # pts.extend([(tl_x + w, tl_y + r), (tl_x + w, tl_y + h - r)])
    corner = circle((tl_x + w - r, tl_y + h - r), r, N=N_seg, angle_start=0, angle_end=90)
    pts.extend(corner.tolist())

    # bottom-left
    # pts.extend([(tl_x + w - r, tl_y + h), (tl_x + r, tl_y + h)])
    corner = circle((tl_x + r, tl_y + h - r), r, N=N_seg, angle_start=90, angle_end=180)
    pts.extend(corner.tolist())

    # left-top
    # pts.extend([(tl_x, tl_y + h - r), (tl_x, tl_y + r)])
    corner = circle((tl_x + r, tl_y + r), r, N=N_seg, angle_start=180, angle_end=270)
    pts.extend(corner.tolist())

    pts.append(pts[0])

    pts = np.array(pts)
    return pts


def drawing_bbox(drawing, padding=0):
    min_x, min_y = np.inf, np.inf
    max_x, max_y = -np.inf, -np.inf
    for line in merge_layers(drawing):
        line_min_x, line_min_y = np.amin(line, axis=0)
        line_max_x, line_max_y = np.amax(line, axis=0)
        min_x = min(min_x, line_min_x)
        min_y = min(min_y, line_min_y)
        max_x = max(max_x, line_max_x)
        max_y = max(max_y, line_max_y)

    min_x -= padding
    min_y -= padding
    max_x += padding
    max_y += padding

    tl_x, tl_y = min_x, min_y
    w, h = max_x - min_x, max_y - min_y

    return (tl_x, tl_y, w, h)


def grid(tl_xy, br_xy, dist_xy):
    xs = np.arange(tl_xy[0], br_xy[0], dist_xy[0])
    ys = np.arange(tl_xy[1], br_xy[1], dist_xy[1])

    lines = []

    # horizontal lines
    for i, y in enumerate(ys):
        if i % 2 == 0:
            start_x = tl_xy[0]
            stop_x = br_xy[0]
        else:
            start_x = br_xy[0]
            stop_x = tl_xy[0]

        lines.append(np.array([[start_x, y], [stop_x, y]]))

    # vertical lines
    for i, x in enumerate(xs):
        if i % 2 == 0:
            start_y = tl_xy[1]
            stop_y = br_xy[1]
        else:
            start_y = br_xy[1]
            stop_y = tl_xy[1]

        lines.append(np.array([[x, start_y], [x, stop_y]]))

    return lines


def load_page_conf(format_path):
    with open(format_path, 'r') as fin:
        conf = json.loads(fin.read())
    return conf


def rotate(drawing, radians=None):
    ''' rotate CCW, right angle if radians not specified '''
    def rotate_layer(lines):
        if radians is not None:
            R = rotation_matrix(radians)
        rotated = []
        for line in lines:
            if radians is None:
                rotated_line = np.stack((line[:, 1],
                                        -line[:, 0]),
                                        axis=1)
            else:
                rotated_line = np.matmul(R, line.T).T

            rotated.append(rotated_line)
        return rotated

    return apply_per_layer(drawing, rotate_layer)


def reflect(lines, mirror_line):
    def reflect_point(pt, mirror_line):
        shifted_pt = pt - mirror_line[0]
        line_direction = mirror_line[1] - mirror_line[0]
        line_direction /= np.linalg.norm(line_direction)
        proj_pt = np.dot(shifted_pt, line_direction) * line_direction
        shifted_mirrored_pt = 2 * proj_pt - shifted_pt
        return shifted_mirrored_pt + mirror_line[0]

    def reflect_lines(lines):
        res = []
        for line in lines:
            res.append(np.array([reflect_point(pt, mirror_line) for pt in line]))
        return res

    return apply_per_layer(lines, reflect_lines)


def shift(lines, shift):
    def shift_layer(lines):
        res = []
        for line in lines:
            res.append(np.array([pt + shift for pt in line]))

        return res

    return apply_per_layer(lines, shift_layer)


def resize_and_center(lines, H, W,
                      l_x_margin, r_x_margin,
                      t_y_margin, b_y_margin):
    """ lines [(N, 2), (M, 2), ...] """
    if type(lines) is not dict:
        single_layer = True
        layers = {'layer': lines}
    else:
        single_layer = False
        layers = lines

    src_min_x, src_min_y = np.inf, np.inf
    src_max_x, src_max_y = -np.inf, -np.inf
    for layer, lines in layers.items():
        for line in lines:
            line_min_x, line_min_y = np.amin(line, axis=0)
            line_max_x, line_max_y = np.amax(line, axis=0)
            src_min_x = min(src_min_x, line_min_x)
            src_min_y = min(src_min_y, line_min_y)
            src_max_x = max(src_max_x, line_max_x)
            src_max_y = max(src_max_y, line_max_y)

    src_H = src_max_y - src_min_y
    src_W = src_max_x - src_min_x

    dst_min_x = l_x_margin
    dst_max_x = W - r_x_margin
    dst_min_y = t_y_margin
    dst_max_y = H - b_y_margin

    dst_H = dst_max_y - dst_min_y
    dst_W = dst_max_x - dst_min_x

    scale = dst_H / src_H
    # take the fitting scale
    if src_W * scale > dst_W:
        scale = dst_W / src_W

    shift_x = (W / 2) - (src_W / 2) * scale
    shift_y = (H / 2) - (src_H / 2) * scale

    shift = np.array((shift_x, shift_y))

    results = {}
    for layer, lines in layers.items():
        layer_results = []
        for line in lines:
            points = line.copy()
            points = points - np.array((src_min_x, src_min_y))
            points = points * scale + shift
            layer_results.append(points)
        results[layer] = layer_results
    if single_layer:
        return results['layer']
    else:
        return results


def count_segments(polylines):
    count = 0
    for line in polylines:
        count += len(line) - 1

    return count


def reverse_line(polyline):
    return polyline[::-1, :]


def line_dist(point, polyline):
    polyline_start = polyline[0]
    polyline_end = polyline[-1]
    dist_start = np.linalg.norm(point - polyline_start)
    dist_end = np.linalg.norm(point - polyline_end)
    if dist_start < dist_end:
        return (dist_start, False)
    else:
        return (dist_end, True)

# def closest_polyline(point, polylines):
#     starts = [x[0, :] for x in polylines]
#     ends = [x[-1, :] for x in polylines]
#     start_dists = np.linalg.norm(starts - point[np.newaxis, :], axis=1)
#     end_dists = np.linalg.norm(ends - point[np.newaxis, :], axis=1)
#     min_start_i = np.argmin(start_dists)
#     min_start_dist = start_dists[min_start_i]
#     min_end_i = np.argmin(end_dists)
#     min_end_dist = end_dists[min_end_i]

#     if min_start_dist <= min_end_dist:
#         return min_start_i, False
#     else:
#         return min_end_i, True


class FalsePBar():
    def update(self, n):
        pass

    def close(self):
        pass


def optimize(drawing, verbose=True,
             path_join_threshold=0.1,
             line_simplification_threshold=0.1,
             path_drop_threshold=0.1):
    if verbose:
        stats = drawing_stats(drawing)
        print('before optimization: {}'.format(stats))

    def optimize_layer(polylines):
        optimized = []
        if verbose:
            pbar = tqdm.tqdm(desc="optimization", total=len(polylines))
        else:
            pbar = FalsePBar()
        optimized.append(polylines[0])
        pos = polylines[0][-1]
        pbar.update(1)
        end_points = np.array([(polyline[0, :], polyline[-1, :]) for polyline in polylines])  # (N, 2 endpoints, 2 coords)
        already_used = np.zeros(len(end_points)) > 1  # all false
        already_used[0] = True

        for iteration in range(len(end_points) - 1):
            sq_dists = np.sum(np.square(pos.reshape(1, 1, 2) - end_points),
                              axis=2)
            sq_dists[already_used, :] = np.inf
            best_i, reversed_p = np.unravel_index(np.argmin(sq_dists),
                                                  sq_dists.shape)

            to_append = polylines[best_i]
            if reversed_p:
                to_append = reverse_line(to_append)
            optimized.append(to_append)
            pbar.update(1)
            pos = to_append[-1]
            already_used[best_i] = True

        pbar.close()
        # order_stats = drawing_stats(optimized)

        optimized = join_straight_lines(optimized, threshold=line_simplification_threshold)
        # join_straight_stats = drawing_stats(optimized)

        optimized = join_path_ends(optimized, threshold=path_join_threshold)
        # join_ends_stats = drawing_stats(optimized)

        optimized = drop_short_lines(optimized, threshold=path_drop_threshold)
        # drop_short_stats = drawing_stats(optimized)

        return optimized
    optimized = apply_per_layer(drawing, optimize_layer)

    if verbose:
        stats = drawing_stats(optimized)
        print('After optimization: {}'.format(stats))

    return optimized


def drop_short_lines(polylines, threshold):
    return [polyline for polyline in polylines if polyline_length(polyline) >= threshold]


def project_to_line(X, A, B):
    line_vector = B - A
    line_direction = line_vector / np.linalg.norm(line_vector)
    point_vector = X - A
    projection = A + np.dot(line_direction, point_vector) * line_direction
    return projection


def join_path_ends(paths, threshold=1e-4):
    result = [paths[0]]
    for path in paths[1:]:
        previous_last = result[-1][-1, :]
        current_first = path[0, :]
        dist = np.linalg.norm(current_first - previous_last)
        if dist < threshold:
            result[-1] = np.concatenate((result[-1],
                                         path),
                                        axis=0)
        else:
            result.append(path)
    return result


def join_straight_lines(paths, threshold=1e-4):
    if threshold is None or threshold == 0:
        return paths

    result = []
    for path in paths:
        simplified = [path[0, :].copy()]
        prev_start = path[0, :].copy()
        prev_end = path[1, :].copy()
        line_end = prev_end.copy()
        for i in range(2, path.shape[0]):
            vertex = path[i, :]
            line_projection = project_to_line(vertex,
                                              prev_start,
                                              line_end)
            dist = np.linalg.norm(vertex - line_projection)
            if dist < threshold:
                prev_end = vertex.copy()
            else:
                simplified.append(prev_end.copy())
                prev_start = prev_end.copy()
                prev_end = vertex.copy()
                line_end = vertex.copy()

        simplified.append(prev_end.copy())
        result.append(np.array(simplified))
    return result


def polyline_length(line):
    pos = line[0]
    length = 0
    for pt in line:
        length += np.linalg.norm(pt - pos)
        pos = pt
    return length


def drawing_stats(drawing):
    def layer_stats(lines):
        n_segments = 0
        length_penup = 0
        length_pendown = 0

        pos = lines[0][0]
        for line in lines:
            length_penup += np.linalg.norm(pos - line[0])
            length_pendown += polyline_length(line)
            n_segments += len(line) - 1
            pos = line[-1]

        return {
            'n_paths': len(lines),
            'n_segments': n_segments,
            'length_penup': length_penup,
            'length_pendown': length_pendown,
            'length': length_penup + length_pendown}

    keys = ['n_paths', 'n_segments', 'length_penup', 'length_pendown', 'length']
    all_results = apply_per_layer(drawing, layer_stats, ensure_layer=True)

    result = {k: sum([x[k] for x in all_results.values()])
              for k in keys}
    return result


def vis_drawing(drawing, *args, **kwargs):
    if type(drawing) is not dict:
        for line in drawing:
            plt.plot(line[:, 0], line[:, 1], *args, **kwargs)
    else:
        options = kwargs.get('layer_options', {})
        for layer_name, layer in drawing.items():
            layer_options = options.get(layer_name, ([], {}))
            layer_args, layer_kwargs = layer_options
            for line in layer:
                plt.plot(line[:, 0], line[:, 1], *layer_args, **layer_kwargs)

    plt.axis('equal')
    if not plt.gca().yaxis_inverted():
        plt.gca().invert_yaxis()


def save_drawing(path, drawing):
    with open(path, 'wb') as fout:
        pickle.dump(drawing, fout, protocol=3)


def load_drawing(path):
    with open(path, 'rb') as fin:
        drawing = pickle.load(fin)
    return drawing


def rotation_matrix(radians):
    s, c = np.sin(radians), np.cos(radians)
    R = np.array([[c, -s],
                  [s, c]])
    return R


if __name__ == '__main__':
    lines = grid((0, 0), (60, 30), (1, 1))
    vis_drawing(lines)
    plt.axis('equal')
    plt.gca().invert_yaxis()
    plt.show()
