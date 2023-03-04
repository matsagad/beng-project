from models.model import PromoterModel
from models.rates.function import RateFunction
from typing import List, Tuple
import numpy as np
import random


class ModelGenerator:
    # (Temporarily) restrict search to constant and linear rate functions
    RATE_FNS = [
        RateFunction.Constant,
        RateFunction.Linear,
    ]  # RateFunction.Function.__subclasses__()
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

    def get_random_model(
        states: int, p_edge: float = 0.5, reversible: bool = True, one_active_state: bool = True
    ) -> PromoterModel:
        # Find a random unfirom spanning tree
        ust = ModelGenerator._find_uniform_spanning_tree(states)
        rate_fn_matrix = [[None] * states for _ in range(states)]
        for start, end in ust:
            rate_fn_matrix[start][end] = ModelGenerator.get_random_rate_fn()
            if reversible:
                rate_fn_matrix[end][start] = ModelGenerator.get_random_rate_fn()

        # Connect any other edges with some probability
        unconnected = np.array(
            [
                (i, j)
                for i in range(states)
                for j in range(states)
                if rate_fn_matrix[i][j] is None
                and ((not reversible and i != j) or (reversible and i > j))
            ],
            dtype=object,
        )
        for start, end in unconnected[
            np.random.binomial(1, p_edge, len(unconnected)) == 1
        ]:
            rate_fn_matrix[start][end] = ModelGenerator.get_random_rate_fn()
            if reversible:
                rate_fn_matrix[end][start] = ModelGenerator.get_random_rate_fn()

        # Must only have one active state
        if one_active_state:
            active_states = np.zeros(states).astype(bool)
            active_states[0] = True
        else:
            active_states = np.random.binomial(1, 0.5, states)

        return PromoterModel(rate_fn_matrix).with_active_states(active_states)

    def is_valid(model: PromoterModel, verbose: bool = False) -> bool:
        # Check if all components are connected
        adj_list = [
            {adj for adj, rate_fn in enumerate(row) if rate_fn is not None}
            for row in model.rate_fn_matrix
        ]
        ## There must be at least one connection to first state
        if not adj_list or not adj_list[0]:
            if verbose:
                print(f"Model {model.hash()} is not connected.")
            return False
        stack = [next(iter(adj_list[0]))]
        seen = set()

        while stack:
            state = stack.pop()
            stack.extend(adj for adj in adj_list[state] if adj not in seen)
            seen.add(state)

        num_states = len(model.rate_fn_matrix)
        if len(seen) != num_states:
            if verbose:
                print(f"Model {model.hash()} is not connected.")
            return False

        # Check all reactions are reversible
        for state in range(len(adj_list)):
            if any(state not in adj_list[adj] for adj in adj_list[state]):
                if verbose:
                    print(f"Model {model.hash()} has an irreversible edge.")
                return False

        # Check there is exactly one active state
        num_active_states = sum(model.active_states)
        if num_active_states != 1:
            if verbose:
                print(
                    f"Model {model.hash()} has {num_active_states} active states (expected one)."
                )
            return False

        return True
