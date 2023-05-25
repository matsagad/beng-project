import heapq
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from nptyping import NDArray
from typing import Callable, List


class BruteNearestNeighbors:
    mp_instance = None

    def __init__(self, n_neighbors: int, metric: Callable, n_jobs=1):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.n_jobs = n_jobs
        self.equal_query = False
        self.cache_results = False
        self.d = None
        self.use_np = True

    def fit(self, points: List[any] | NDArray):
        self.points = points

    def kneighbors(self, queries: List[any] | NDArray, return_distance: bool = False):
        m, n = len(queries), len(self.points)

        if self.n_jobs == 1:
            if self.use_np:
                d = np.empty((m, n))
                if self.equal_query:
                    for query_index, (neighbors, query) in enumerate(zip(d, queries)):
                        neighbors[query_index] = 0
                        for offset, point in enumerate(self.points[query_index + 1 :]):
                            point_index = query_index + offset + 1
                            distance = self.metric(query, point)
                            neighbors[point_index] = distance
                            d[point_index, query_index] = distance
                else:
                    for neighbors, query in zip(d, queries):
                        for point_index, point in enumerate(self.points):
                            distance = self.metric(query, point)
                            neighbors[point_index] = distance

                k_neighbors = np.argsort(d)[:, : self.n_neighbors]
                distances = d[np.arange(m)[:, None], k_neighbors]
            else:
                d = [[] for _ in range(m)]
                if self.equal_query:
                    for query_index, neighbors in enumerate(d):
                        neighbors.append((0, query_index))
                    for query_index, (neighbors, query) in enumerate(zip(d, queries)):
                        for offset, point in enumerate(self.points[query_index + 1 :]):
                            point_index = query_index + offset + 1
                            distance = self.metric(query, point)
                            heapq.heappush(neighbors, (distance, point_index))
                            heapq.heappush(d[point_index], (distance, query_index))
                else:
                    for neighbors, query in zip(d, queries):
                        for point_index, point in enumerate(self.points):
                            distance = self.metric(query, point)
                            heapq.heappush(neighbors, (distance, point_index))

                k_neighbors, distances = map(
                    list,
                    zip(
                        *(
                            map(
                                list,
                                zip(
                                    *self.get_top_from_heap(neighbors, self.n_neighbors)
                                ),
                            )
                            for neighbors in d
                        )
                    ),
                )
            if self.cache_results:
                self.d = d

            if not return_distance:
                return k_neighbors
            return distances, k_neighbors

        BruteNearestNeighbors.mp_instance = self

        points_a, points_b = (
            np.triu_indices(m)
            if self.equal_query
            else (grid.flatten() for grid in np.indices((m, n)))
        )

        indices = np.array_split(np.arange(points_a.shape[0]), self.n_jobs)

        d = np.empty((m, m))

        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            futures = []
            for pair in indices:
                if pair.shape[0] > 0:
                    futures.append(
                        executor.submit(
                            BruteNearestNeighbors.find_distances,
                            points_a[pair],
                            points_b[pair],
                        )
                    )
            d[np.diag_indices(m)] = 0

            for future in as_completed(futures):
                point_a, point_b, distances = future.result()
                d[point_a, point_b] = distances
                if self.equal_query:
                    d[point_b, point_a] = distances

        k_neighbors = np.argsort(d)[:, : self.n_neighbors]
        distances = d[np.arange(m)[:, None], k_neighbors]

        if self.cache_results:
            self.d = d

        if not return_distance:
            return k_neighbors
        return distances, k_neighbors

    def find_distances(point_a: List[int], point_b: List[int]):
        mp = BruteNearestNeighbors.mp_instance
        distances = []
        for a, b in zip(point_a, point_b):
            point = mp.points[a]
            other = mp.points[b]
            distances.append(mp.metric(point, other))
        return point_a, point_b, distances

    def get_top_from_heap(self, heap: List[any], n: int):
        assert n <= len(heap)
        return (heapq.heappop(heap) for _ in range(n))
