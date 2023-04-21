from models.generator import ModelGenerator
from pipeline.one_step_decoding import OneStepDecodingPipeline
from tests.test import TestWithData
import numpy as np


class DistanceMetricTests(TestWithData):
    def __init__(self):
        super().__init__()

    def test_trajectory_measure_is_a_metric(self):
        from evolution.novelty.metrics import TrajectoryMetric
        from itertools import permutations

        n_models = 50

        pip = OneStepDecodingPipeline(self.data, realised=False)
        pip.set_parallel()
        get_classes = lambda model: pip.estimator._split_classes(
            model, pip.simulator.simulate(model)
        )
        models = [
            ModelGenerator.get_random_model(states)
            for states in np.random.choice(4, n_models) + 2
        ]

        multi_trajectories = []
        avg_trajectories = []

        print("\tFinding trajectories...")
        for model in models:
            multi_trajectories.append(get_classes(model))
            avg_trajectories.append(np.average(multi_trajectories[-1], axis=1))

        d = TrajectoryMetric.rms_js_metric_for_trajectories

        print("\tChecking...")
        for trajectories in (multi_trajectories, avg_trajectories):
            for i, j, k in permutations(trajectories, 3):
                if d(i, k) > d(i, j) + d(j, k):
                    assert (
                        False
                    ), f"Triangle inequality condition failed: d(i, j) + d(j, k) < d(i, k), {d(i, j) + d(j, k):.3f} < {d(i, k):.3f}"

    def test_topology_measure_is_a_metric(self):
        from evolution.novelty.metrics import TopologyMetric
        from itertools import permutations

        n_models = 50

        models = [
            ModelGenerator.get_random_model(states)
            for states in np.random.choice(4, n_models) + 2
        ]

        feature_vectors = [TopologyMetric.get_feature_vector(model) for model in models]

        d = TopologyMetric.wwl_metric_for_wl_feature_vectors

        print("\tChecking...")
        for i, j, k in permutations(feature_vectors, 3):
            # Round calculations to avoid float imprecision leading to
            # incorrect comparison. This happens as the topology metric
            # often calculates the same distance, e.g. 0.87566666666 and 0.7566666667
            if round(d(i, k), 8) > round(d(i, j) + d(j, k), 8):
                assert (
                    False
                ), f"Triangle inequality condition failed: d(i, j) + d(j, k) <= d(i, k), {d(i, j) + d(j, k):.8f} < {d(i, k):.8f}"
