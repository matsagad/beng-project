from models.model import PromoterModel
from models.rates.function import RateFunction
from typing import List, Tuple
import numpy as np
import random


class ModelGenerator:
    RATE_FNS = RateFunction.Function.__subclasses__()
    TF_COUNT = 5

    def get_random_rate_fn(low: float = -2, high: float = 2) -> RateFunction:
        rf_cls = np.random.choice(ModelGenerator.RATE_FNS)
        n_rates, n_tfs = rf_cls.num_params()
        return rf_cls(
            rates=(
                10
                ** np.random.uniform(
                    low,
                    high,
                    n_rates,
                )
            ).tolist(),
            tfs=np.random.choice(ModelGenerator.TF_COUNT, n_tfs).tolist(),
        )

    def _find_uniform_spanning_tree(states: int) -> List[Tuple[int, int]]:
        # Find a uniform spanning tree through Wilson's loop-erased random walk
        root = np.random.choice(states)
        remaining = set(i for i in range(states))
        remaining.discard(root)

        tree = {root}
        edges = []

        while remaining:
            curr = random.sample(remaining, 1)[0]
            candidates = set(i for i in range(states) if i != curr)
            path = [curr]

            while candidates:
                # Find random node not yet visited but could be in tree
                node = random.sample(candidates, 1)[0]
                path.append(node)

                # Stop if next node is already in the tree
                if node in tree:
                    edges.extend(edge for edge in zip(path, path[1:]))
                    tree.update(path)
                    break

                candidates.discard(node)

            remaining.difference_update(path)

        return edges

    def get_random_model(states: int, p_edge: float = 0.5) -> PromoterModel:
        # Find a random unfirom spanning tree
        ust = ModelGenerator._find_uniform_spanning_tree(states)
        rate_fn_matrix = [[None] * states for _ in range(states)]
        for start, end in ust:
            rate_fn_matrix[start][end] = ModelGenerator.get_random_rate_fn()

        # Connect any other edges with some probability
        unconnected = np.array(
            [
                (i, j)
                for i in range(states)
                for j in range(states)
                if i != j and rate_fn_matrix[i][j] is None
            ],
            dtype=object,
        )
        for start, end in unconnected[
            np.random.binomial(1, p_edge, len(unconnected)) == 1
        ]:
            rate_fn_matrix[start][end] = ModelGenerator.get_random_rate_fn()
        return PromoterModel(rate_fn_matrix).with_active_states(
            np.random.binomial(1, 0.5, states).astype(bool)
        )
