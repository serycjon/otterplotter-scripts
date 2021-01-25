# -*- coding: utf-8 -*-
import sys
import argparse

import scipy.spatial
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
from scipy.cluster.vq import vq, kmeans
import numpy as np
import cv2
import matplotlib.pyplot as plt

from img_tools import sample_points_naive, floyd_steinberg

import tqdm

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


def parse_arguments():
    parser = argparse.ArgumentParser(description='',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('img', help='image path')
    return parser.parse_args()


def point_tsp(points, return_ids=False,
              limit_seconds=30, solution_limit=None):
    ''' adapted from https://developers.google.com/optimization/routing/tsp '''
    assert points.shape[1] == 2
    if not solution_limit:
        solution_limit = np.iinfo(np.int64).max

    distances = scipy.spatial.distance.cdist(points, points, 'euclidean')
    int_distances = np.round(distances * 1e2).astype(np.int64)

    data = {'distance_matrix': int_distances,
            'num_vehicles': 1,
            'depot': 0}
    manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']),
                                           data['num_vehicles'], data['depot'])
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        """Returns the distance between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['distance_matrix'][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)

    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
    search_parameters.time_limit.seconds = limit_seconds
    search_parameters.solution_limit = solution_limit

    search_parameters.log_search = False

    solution = routing.SolveWithParameters(search_parameters)

    if not solution:
        raise RuntimeError("Solution not found")
    path = []
    ids = []
    index = routing.Start(0)
    while not routing.IsEnd(index):
        index = solution.Value(routing.NextVar(index))
        node = manager.IndexToNode(index)
        path.append(points[node, :])
        ids.append(node)

    finished_path = np.array(path + [path[0]])
    if return_ids:
        return finished_path, ids
    return finished_path


def run(args):
    img = cv2.imread(args.img)
    H, W = 600, 800
    img = cv2.resize(img, (W, H))
    # gray = bgr2cmyk(img)[:, :, 3]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dithered = floyd_steinberg(np.float64(gray) / 255)
    cv2.imshow("cv: dithered", dithered)
    c = cv2.waitKey(0)
    if c == ord('q'):
        import sys
        sys.exit(1)

    # src = img_resize_to_px_count(gray, 4 * 800 * 800)
    src = gray
    N_points = 300
    logging.info(f'Sampling {N_points} points')
    pts = sample_points_naive(src, N=N_points)
    float_pts = pts.astype(np.float64)

    sub_route_N = 60
    K = pts.shape[0] // sub_route_N
    logging.info(f'K-means clustering with K={K}')
    centroids, _ = kmeans(float_pts, K, check_finite=False, thresh=100)
    labels, _ = vq(float_pts, centroids)

    logging.info(f'Main-route TSP ({K} nodes)')
    main_route, main_route_ids = point_tsp(centroids, return_ids=True, limit_seconds=4)

    plt.plot(main_route[:, 1], main_route[:, 0], color='b', lw=0.3)
    # plt.scatter(pts[:, 1], pts[:, 0], color='r', s=0.01)

    logging.info(f'Sub-route TSPs (each about {sub_route_N} nodes)')
    sub_routes = []
    for i in tqdm.tqdm(range(K)):
        cluster_pts = pts[labels == i, :]
        sub_route = point_tsp(cluster_pts, limit_seconds=2, solution_limit=35)
        sub_routes.append(sub_route)
        plt.plot(sub_route[:, 1], sub_route[:, 0], lw=0.2)
        plt.scatter(cluster_pts[:, 1], cluster_pts[:, 0], c=plt.gca().lines[-1].get_color(), s=0.01)


    for i in main_route_ids:
        pass
        
    # plt.axis('equal')
    # plt.gca().invert_yaxis()
    # plt.imshow(src, cmap='gray')
    plt.show()
    return 0


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
