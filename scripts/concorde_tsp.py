import numpy as np
from concorde.tsp import TSPSolver


def concorde_tsp(points, time_bound=3.0, return_ids=False):
    """ Calculate TSP route using pyconcorde.

    args:
        points: (N, 2) numpy array
        return_ids: bool - also return route node ids

    returns:
        tour_points: (N, 2) numpy array
        tour_ids: ordering of points """
    norm = "EUC_2D"
    solver = TSPSolver.from_data(points[:, 0], points[:, 1], norm)
    solution = solver.solve(time_bound=time_bound, verbose=False)
    tour_ids = solution.tour

    tour_points = []
    for idx in tour_ids:
        tour_points.append(points[idx, :])

    tour_points = np.array(tour_points)

    if return_ids:
        return tour_points, tour_ids
    else:
        return tour_points
