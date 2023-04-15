from nptyping import NDArray
import numpy as np


class TrajectoryMetric:
    def kl_divergence(p_dist: NDArray, q_dist: NDArray) -> float:
        """
        Kullback-Leibler Divergence
        """
        _p_dist, _q_dist = p_dist[p_dist > 0], q_dist[p_dist > 0]
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

    def js_divergence_for_trajectories(p_traj: NDArray, q_traj: NDArray) -> float:
        """
        Trajectory values represent activity within the promoter region. This is a weighted
        sum of the probabilities of being in each state. An interpretation of this value
        is the probability of being active at that point in time.

        Treating this as a time series of probability distributions, we can compare
        point-wise the probability distributions using the JS divergence to get an
        estimate of how distinct two trajectories are (i.e. how reliably one can
        distinguish between the two, in units of bits).
        """
        return (
            TrajectoryMetric.js_divergence(p_traj, q_traj)
            + TrajectoryMetric.js_divergence(1 - p_traj, 1 - q_traj)
        ) / np.prod(p_traj.shape)


class ArchitectureMetric:
    pass
