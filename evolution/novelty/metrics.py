from nptyping import NDArray
import numpy as np


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
            (
                TrajectoryMetric.js_divergence(p_traj, q_traj)
                + TrajectoryMetric.js_divergence(1 - p_traj, 1 - q_traj)
            )
            / p_traj.size
        )


class ArchitectureMetric:
    pass
