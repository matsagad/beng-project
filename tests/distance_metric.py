from models.generator import ModelGenerator
from pipeline.one_step_decoding import OneStepDecodingPipeline
from tests.test import TestWithData
import numpy as np


class DistanceMetricTests(TestWithData):
    def __init__(self):
        super().__init__()

    def trajectory_measure_is_a_metric(self):
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

        print("Finding trajectories...")
        for model in models:
            multi_trajectories.append(get_classes(model))
            avg_trajectories.append(np.average(multi_trajectories[-1], axis=1))

        d = TrajectoryMetric.rms_js_metric_for_trajectories

        print("Checking...")
        for trajectories in (multi_trajectories, avg_trajectories):
            for i, j, k in permutations(trajectories, 3):
                if d(i, k) > d(i, j) + d(j, k):
                    assert (
                        False
                    ), f"Triangle inequality condition failed: d(i, j) + d(j, k) < d(i, k), {d(i, j) + d(j, k):.3f} < {d(i, k):.3f}"
