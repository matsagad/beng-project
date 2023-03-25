from typing import List, Set, Tuple
from nptyping import Int, NDArray, Shape
from models.model import PromoterModel
import copy
import numpy as np
from collections import deque


class CrossoverOperator:
    def one_point_row_swap(
        model: PromoterModel,
        other: PromoterModel,
        model_is_elite: bool = False,
        other_is_elite: bool = False,
    ) -> List[PromoterModel]:
        """
        Performs a one-point crossover on the rows of the adjacency matrices of two
        promoter models. Note: this does not preserve the reversibility of reactions.
        """
        num_states = len(model.rate_fn_matrix)
        # Randomly choose splitting point
        split = int(1 if num_states == 2 else 1 + np.random.choice(num_states - 2, 1))

        # If model is not elite then it is no longer used in the next generation
        # and thus its rate functions can be reused. If it is an elite, then its
        # rate functions should be copied to avoid side-effects from mutations.
        model_copy = copy.deepcopy if model_is_elite else lambda x: x
        other_copy = copy.deepcopy if other_is_elite else lambda x: x
        _models = [(model, model_copy), (other, other_copy)]

        # Generate offspring
        return [
            PromoterModel(
                fst_copy(fst.rate_fn_matrix[:split])
                + snd_copy(snd.rate_fn_matrix[split:])
            ).with_activity_weights(
                np.concatenate(
                    (
                        fst_copy(fst.activity_weights[:split]),
                        snd_copy(snd.activity_weights[split:]),
                    )
                )
            )
            for (fst, fst_copy), (snd, snd_copy) in zip(_models, _models[::-1])
        ]

    def one_point_triangular_row_swap(
        model: PromoterModel,
        other: PromoterModel,
        model_is_elite: bool = False,
        other_is_elite: bool = False,
    ) -> List[PromoterModel]:
        """
        Performs a one-point crossover on the rows of the upper-triangular portion of the
        adjacency matrices of two promoter models. By swapping an edge (i, j), we also
        include the edge (j, i) in the lower-triangular half. This preserves reversibility.
        """
        num_states = len(model.rate_fn_matrix)
        # Randomly choose splitting point
        split = int(
            1
            if num_states == 2
            else 1 + np.random.choice(((num_states - 1) * (num_states - 2)) // 2, 1)
        )

        # If model is not elite then it is no longer used in the next generation
        # and thus its rate functions can be reused. If it is an elite, then its
        # rate functions should be copied to avoid side-effects from mutations.
        model_copy = copy.deepcopy if model_is_elite else lambda x: x
        other_copy = copy.deepcopy if other_is_elite else lambda x: x
        _models = [(model, model_copy), (other, other_copy)]

        # Generate offspring
        child1 = np.array(model_copy(model.rate_fn_matrix))
        child2 = np.array(other_copy(other.rate_fn_matrix))

        top_split = tuple(arr[:split] for arr in np.triu_indices(num_states, 1))
        bot_split = top_split[::-1]

        for tri_split in (top_split, bot_split):
            child1[tri_split], child2[tri_split] = (
                child2[tri_split],
                child1[tri_split],
            )

        return [
            PromoterModel(child).with_activity_weights(
                np.concatenate(
                    (
                        fst_copy(fst.activity_weights[:split]),
                        snd_copy(snd.activity_weights[split:]),
                    )
                )
            )
            for ((fst, fst_copy), (snd, snd_copy), child) in zip(
                _models, _models[::-1], (child1, child2)
            )
        ]

    def _shortest_path_between(
        start: int, end: int, adj_matrix: NDArray[Shape["Any, Any"], Int]
    ) -> List[Tuple[int, int]]:
        queue = deque()
        queue.append(([], start))
        seen = set()

        while queue:
            path, node = queue.popleft()
            if node == end:
                return path
            seen.add(node)

            for adj, is_connected in enumerate(adj_matrix[node]):
                if is_connected and adj not in seen:
                    queue.append((path + [(node, adj)], adj))

        return []

    def _get_disjoint_vertex_sets(
        adj_matrix: NDArray[Shape["Any, Any"], Int]
    ) -> List[Set[int]]:
        vertex_sets = []
        seen = set()

        for node in range(len(adj_matrix)):
            if node in seen:
                continue
            seen_in_component = set()

            # Only gets run N times where N is the number of disjoint vertex sets
            stack = [node]
            while stack:
                curr = stack.pop()
                seen_in_component.add(curr)

                for adj, is_connected in enumerate(adj_matrix[curr]):
                    if is_connected and adj not in seen_in_component:
                        stack.append(adj)

            seen.update(seen_in_component)
            vertex_sets.append(seen_in_component)

        return vertex_sets

    def subgraph_swap(
        model: PromoterModel,
        other: PromoterModel,
        model_is_elite: bool = False,
        other_is_elite: bool = False,
    ) -> List[PromoterModel]:
        """
        Performs a crossover similar to Globus, Lawton, and Wipke's JavaGenes. It is adapted
        to ensure there is exactly one active state in each offspring. The main idea is to split
        each model graph into two subgraphs - one with the active state and one without - and
        cross-merge these subgraphs so one with an active state is paired with one without.
        """
        # DIVISION: split each model into two subgraphs
        cut_set = [[], []]
        model_masks = [[], []]

        for child, p_model in enumerate((model, other)):
            ## Choose a random (undirected) edge
            adj_matrix = np.array(
                [
                    [int(rate_fn is not None) for rate_fn in row]
                    for row in p_model.rate_fn_matrix
                ]
            )
            edges = np.argwhere(adj_matrix)
            chosen_edge = tuple(edges[np.random.choice(len(edges))])

            ## (TODO?) Possible optimisation by finding all paths from u -> v at once
            ## (a list of sets of edges) and find a minimum cover for these sets.
            while True:
                ## Break the edge
                adj_matrix[chosen_edge] = 0
                adj_matrix[chosen_edge[::-1]] = 0
                cut_set[child].append(chosen_edge)

                path = CrossoverOperator._shortest_path_between(
                    *chosen_edge, adj_matrix
                )
                if not path:
                    break
                ## Choose a random edge in the path if a path still exists
                chosen_edge = path[np.random.choice(len(path))]

            model_masks[child] = adj_matrix

        ## Include reversible counterpart of each directed edge
        for broken_edges in cut_set:
            ## Note to future self: can't use extend here as a recursive definition will hang!
            broken_edges += [(b, a) for (a, b) in broken_edges]
            np.random.shuffle(broken_edges)

        ## Find vertex sets and arrange them so active state is in first component
        _ACTIVE_STATE = 0
        vertex_sets = [
            vertex_set if _ACTIVE_STATE in vertex_set[0] else vertex_set[::-1]
            for vertex_set in (
                CrossoverOperator._get_disjoint_vertex_sets(model_mask)
                for model_mask in model_masks
            )
        ]
        cut_sets_by_activity = [
            [
                [
                    (a, b)
                    for (a, b) in cuts
                    if a in vertex_sets[i][activity]
                    and b not in vertex_sets[i][activity]
                ]
                for i, cuts in enumerate(cut_set)
            ]
            for activity in range(2)
        ]
        same_activity_cuts = [
            [
                [
                    (a, b)
                    for (a, b) in cuts
                    if a in vertex_sets[i][activity] and b in vertex_sets[i][activity]
                ]
                for activity in range(2)
            ]
            for i, cuts in enumerate(cut_set)
        ]

        ## If model is not elite then it is no longer used in the next generation
        #  and thus its rate functions can be reused. If it is an elite, then its
        #  rate functions should be copied to avoid side-effects from mutations.
        copies = tuple(
            (
                np.array(
                    copy.deepcopy(p_model.rate_fn_matrix)
                    if p_model_is_elite
                    else p_model.rate_fn_matrix
                )
                for p_model, p_model_is_elite in zip(
                    (model, other), (model_is_elite, other_is_elite)
                )
            )
        )

        ## Map old states to new states as indices of states may change
        state_maps = [{}, {}]

        for child in (0, 1):
            keys = [(0, ind) for ind in vertex_sets[child][0]] + [
                (1, ind) for ind in vertex_sets[1 - child][1]
            ]
            state_maps[child] = {keys: i for i, keys in enumerate(keys)}

        ## Populate children with states and edges from parents that are carried over.
        ## (First child will inherit the active state from first model)
        _num_states = [
            len(vertex_sets[i][0]) + len(vertex_sets[1 - i][1]) for i in range(2)
        ]
        children = [np.full((states, states), None) for states in _num_states]

        for child in (0, 1):
            for activity in (0, 1):
                model_to_choose = (child + activity) % 2
                model_activity = np.array(list(vertex_sets[model_to_choose][activity]))

                activity_rates = model_masks[model_to_choose][
                    np.ix_(model_activity, model_activity)
                ]

                activity_indices = tuple(
                    zip(*(model_activity[inds] for inds in np.argwhere(activity_rates)))
                )

                if activity_indices:
                    children[child][
                        tuple(
                            [state_maps[child][(activity, j)] for j in dims]
                            for dims in activity_indices
                        )
                    ] = copies[model_to_choose][activity_indices]

        # RECOMBINATION: combine the two sets of subgraphs together
        for child, cut_sets in enumerate(
            zip(cut_sets_by_activity[0], cut_sets_by_activity[1][::-1])
        ):
            model_cut_set, other_cut_set = cut_sets
            while model_cut_set and other_cut_set:
                model_edge, other_edge = model_cut_set.pop(), other_cut_set.pop()

                index = (
                    state_maps[child][(0, model_edge[0])],
                    state_maps[child][(1, other_edge[0])],
                )
                children[child][index] = copies[child][model_edge]
                children[child][index[::-1]] = copies[1 - child][other_edge]

            for activity, broken_edges in enumerate(cut_sets):
                for edge_pair in broken_edges:
                    start = state_maps[child][(activity, edge_pair[0])]
                    end = np.random.choice(_num_states[child] - 1)
                    index = (start, end if end < start else end + 1)

                    ##  Either accept change or use already existing edge
                    for scale in (1, -1):
                        if children[child][
                            index[::scale]
                        ] is None or np.random.binomial(1, 0.5):
                            children[child][index[::scale]] = copies[
                                (activity + child) % 2
                            ][edge_pair[::scale]]

        for child in (0, 1):
            active_set = same_activity_cuts[child][0]
            inactive_set = same_activity_cuts[1 - child][1]
            for activity, broken_edges in enumerate((active_set, inactive_set)):
                for edge_pair in broken_edges:
                    start = state_maps[child][(activity, edge_pair[0])]
                    end = np.random.choice(_num_states[child] - 1)
                    index = (start, end if end < start else end + 1)

                    for scale in (1, -1):
                        if children[child][
                            index[::scale]
                        ] is None or np.random.binomial(1, 0.5):
                            children[child][index[::scale]] = copies[
                                (child + activity) % 2
                            ][edge_pair[::scale]]

        parents = (model, other)
        return [
            PromoterModel(child_matrix).with_activity_weights(
                np.concatenate(
                    (
                        parents[i].activity_weights[sorted(vertex_sets[i][0])],
                        parents[1 - i].activity_weights[sorted(vertex_sets[1 - i][1])],
                    )
                )
            )
            for (i, child_matrix) in enumerate(children)
        ]
