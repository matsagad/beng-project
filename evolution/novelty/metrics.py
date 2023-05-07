from models.model import PromoterModel
from nptyping import NDArray
from sklearn.metrics import pairwise_distances
from typing import Tuple, List
import hashlib
import json
import numpy as np
import ot


class TrajectoryMetric:
    def kl_divergence(p_dist: NDArray, q_dist: NDArray) -> float:
        """
        Kullback-Leibler Divergence
        """
        non_zero = np.logical_and(p_dist > 0, q_dist > 0)
        _p_dist = p_dist[non_zero]
        _q_dist = q_dist[non_zero]
        return np.sum(_p_dist * np.log2(_p_dist / _q_dist))

    def js_divergence(p_dist: NDArray, q_dist: NDArray) -> float:
        """
        Jensen-Shannon Divergence
        """
        m_dist = (p_dist + q_dist) / 2
        return (
            TrajectoryMetric.kl_divergence(p_dist, m_dist)
            + TrajectoryMetric.kl_divergence(q_dist, m_dist)
        ) / 2

    def rms_js_metric_for_trajectories(p_traj: NDArray, q_traj: NDArray) -> float:
        """
        Root mean square Jensen-Shannon distance metric

        Trajectory values represent activity within the promoter region. This is a weighted
        sum of the probabilities of being in each state. An interpretation of this value
        is the probability of being active at that point in time.

        Treating this as a time series of probability distributions, we can compare
        point-wise the probability distributions using the JS divergence to get an
        estimate of how distinct two trajectories are (i.e. how reliably one can
        distinguish between the two, in units of bits).

        Given two vectors of trajectory values x and y, this function outputs
        the root mean square of d_JS(x_i, y_i) where d_JS is the Jensen-Shannon
        distance metric (square root of JS divergence).
        """
        return np.sqrt(
            min(
                TrajectoryMetric.js_divergence(p_traj, q_traj)
                + TrajectoryMetric.js_divergence(1 - p_traj, 1 - q_traj),
                TrajectoryMetric.js_divergence(p_traj, 1 - q_traj)
                + TrajectoryMetric.js_divergence(1 - p_traj, q_traj),
            )
            / p_traj.size
        )

    def v_kl_divergence(p_dist: NDArray, q_dist: NDArray) -> NDArray:
        """
        Vectorised Kullback-Leibler Divergence
        """
        non_zero = np.logical_and(p_dist > 0, q_dist > 0)
        _p_dist = p_dist[non_zero]
        _q_dist = q_dist[non_zero]

        res = np.zeros(p_dist.shape)
        res[non_zero] = _p_dist * np.log2(_p_dist / _q_dist)

        return res

    def v_js_divergence(p_dist: NDArray, q_dist: NDArray) -> float:
        """
        Vectorised Jensen-Shannon Divergence
        """
        m_dist = (p_dist + q_dist) / 2
        return (
            TrajectoryMetric.v_kl_divergence(p_dist, m_dist)
            + TrajectoryMetric.v_kl_divergence(q_dist, m_dist)
        ) / 2

    def mean_js_metric_for_trajectories(p_traj: NDArray, q_traj: NDArray) -> float:
        return np.log2(
            1
            + (
                np.sum(
                    (
                        np.sqrt(
                            (
                                TrajectoryMetric.js_divergence(p_traj, q_traj)
                                + TrajectoryMetric.js_divergence(1 - p_traj, 1 - q_traj)
                            )
                        )
                    )
                )
            )
            / p_traj.size
        )


class TopologyMetric:
    PAD_ENTRY = -1

    def _to_line_digraph(model: PromoterModel) -> Tuple[List[int], NDArray]:
        """
        Convert model's architecture to a line digraph based on the TF dependent
        on each edge.
        """
        adj_matrix = model.rate_fn_matrix
        indicator = np.array(
            [[int(rate_fn is not None) for rate_fn in row] for row in adj_matrix]
        )

        num_nodes_per_row = np.sum(indicator, axis=1)

        node_indices = np.cumsum(num_nodes_per_row)
        adj_lists = [
            [i for i in range(start, end)]
            for start, end in zip([0] + list(node_indices), node_indices)
        ]

        num_dims = indicator.shape[0]
        row_connections = indicator * np.tile(np.arange(num_dims), (num_dims, 1)) - (
            1 - indicator
        )

        num_nodes = sum(num_nodes_per_row)
        line_matrix = np.zeros((num_nodes, num_nodes))

        for new_node_index, row_connected in enumerate(
            row_connections[row_connections != -1]
        ):
            line_matrix[new_node_index, adj_lists[row_connected]] = 1

        line_labels = [
            rate_fn.tfs[0] if rate_fn.tfs else -1
            for row in adj_matrix
            for rate_fn in row
            if rate_fn is not None
        ]
        line_weights = [
            rate_fn.rates[0]
            for row in adj_matrix
            for rate_fn in row
            if rate_fn is not None
        ]
        return line_labels, np.log10(line_weights), line_matrix

    def _hash(object: any) -> int:
        """
        Note: getting only the last 16 digits of hashlib.sha256 does not guarantee
        a perfect hash as is technically required. However, it is rather unlikely
        to have collisions and is computationally efficient.
        """
        return (
            int(hashlib.sha256(json.dumps(object).encode("utf-8")).hexdigest(), 16)
            % 10**16
        )

    def weisfeiler_lehman_iterate(
        labels: List[int], weights: List[float], graph: NDArray, h_iterations: int = 3
    ) -> List[List[int]]:
        """
        Weisfeiler-Lehman iteration scheme

        """
        hash_labels = [[TopologyMetric._hash(int(label)) for label in labels]]
        weight_attributes = [weights]
        neighborhood = [[i for i, is_adj in enumerate(row) if is_adj] for row in graph]

        for _ in range(h_iterations):
            curr_labels = hash_labels[-1]
            new_labels = [
                TopologyMetric._hash(
                    (
                        int(v_label),
                        *(int(curr_labels[neighbor]) for neighbor in neighborhood[i]),
                    )
                )
                for i, v_label in enumerate(curr_labels)
            ]
            hash_labels.append(new_labels)

            curr_weights = weight_attributes[-1]
            new_weights = np.array(
                [
                    (v_weight + np.mean(curr_weights[neighborhood[i]])) / 2
                    for i, v_weight in enumerate(curr_weights)
                ]
            )
            weight_attributes.append(new_weights)

        # Transpose so trailing dimension is the number of iterations
        return np.hstack((np.array(hash_labels), np.array(weight_attributes))).T

    def get_feature_vector(model: PromoterModel, h_iterations: int = 3) -> NDArray:
        return TopologyMetric.weisfeiler_lehman_iterate(
            *TopologyMetric._to_line_digraph(model), h_iterations=h_iterations
        )

    def wwl_metric_for_models(model: PromoterModel, other: PromoterModel) -> float:
        """
        Wasserstein Weisfeiler-Lehman (on line digraph) distance metric

        The promoter architecture is largely dictated by which TFs each rate
        function depends on when going between two states. With this in mind,
        a line digraph can be constructed from the Markov chain where each node
        is labelled by the TF it depends on (-1 for constant rates).

        With a labelled directed graph, the Wasserstein Weisfeiler-Lehman graph
        distance can be utilised as described by Togninalli et. al (2019). In this
        case, only one categorical value is attached to nodes (originally edges)
        - its label. The Weisfeiler-Lehman iterative procedure is done three times
        as is in line with recommended in Chen et. al (2022). Hamming distance for
        pairwise distances and a network simplex method (optimal transport) for
        calculating the transport matrix are both followed as done by Togninalli et al.
        """
        f_model, f_other = (
            TopologyMetric.get_feature_vector(m) for m in (model, other)
        )

        return TopologyMetric.wwl_metric_for_wl_feature_vectors(f_model, f_other)

    def wwl_metric_for_wl_feature_vectors(f_p: NDArray, f_q: NDArray) -> float:
        """
        A metric to act on pre-computed Weisfeiler-Lehman feature vectors.
        """
        f_p_cat, f_p_cont = np.split(f_p, 2)
        f_q_cat, f_q_cont = np.split(f_q, 2)

        D_cat = pairwise_distances(f_p_cat, f_q_cat, metric="hamming")
        D_cont = pairwise_distances(f_p_cont, f_q_cont, metric="euclidean")

        D = (9 * D_cat + D_cont) / 10

        P_min = ot.emd([], [], D)
        distance = np.sum(P_min * D)

        return distance

    def wwl_metric_for_serialised_wl_feature_vectors(
        f_p: NDArray, f_q: NDArray
    ) -> float:
        """
        This acts as an adapter to the sklearn NearestNeighbors API as it
        only works for numerical arrays as points in the feature space, all
        arranged in a single 2D matrix.
        """
        return TopologyMetric.wwl_metric_for_wl_feature_vectors(
            TopologyMetric.deserialise(f_p), TopologyMetric.deserialise(f_q)
        )

    def serialise(feature_vectors: List[NDArray]) -> NDArray:
        f_lens = [len(f) for f in feature_vectors]
        dim = max(f_lens)
        # Use -1 as hash implementation is stricly positive
        nn_arrays = np.zeros((len(feature_vectors), dim)) + TopologyMetric.PAD_ENTRY
        nn_arrays[np.arange(dim) < np.array(f_lens)[:, None]] = np.concatenate(
            feature_vectors
        )
        return nn_arrays

    def deserialise(feature_vector: NDArray, h_iterations: int = 3) -> NDArray:
        return feature_vector[
            np.where(feature_vector != TopologyMetric.PAD_ENTRY)
        ].reshape((-1, h_iterations + 1))
