from mi_estimation.decoding import DecodingEstimator
from models.generator import ModelGenerator
from models.model import PromoterModel
from pipeline.one_step_decoding import OneStepDecodingPipeline
from tests.test import TestWithData
import numpy as np


class MIEstimationTests(TestWithData):
    def __init__(self):
        super().__init__()

    def test_hypothetical_perfect_model(self):
        """
        Tests if a model that theoretically outputs a perfectly distinguishable trajectory
        in all environments should attain the maximum MI of 2.0.
        """
        est = DecodingEstimator(**self.default_est_args)
        est.parallel = True

        num_envs, _, num_cells, num_times = self.data.shape
        trajectory = np.zeros((num_times, num_envs, num_cells * est.replicates, 2))

        # Rich state left as is. Note: first state is active state
        origin = self.origin
        trajectory[origin : origin + origin // 4, 0, :, 0] = 1
        trajectory[origin : origin + 2 * origin // 4, 1, :, 0] = 1
        trajectory[origin : origin + 3 * origin // 4, 2, :, 0] = 1

        model = PromoterModel.dummy()
        mi_score = est.estimate(model, trajectory)

        assert mi_score == 2.0, "Projected MI != 2.0"

    def test_random_model_variance(self):
        """
        Tests if low MI models are handled correctly, e.g. there are
        no sklearn exceptions raised due to zero variance.

        Due to the curse of dimensionality, randomly generated
        large models will tend to perform poorly and are a good test sets.
        """
        threshold = 0.1
        num_models = 10
        num_trials = 10

        pip = OneStepDecodingPipeline(**self.default_pip_args)
        pip.set_parallel()

        for _ in range(num_models):
            model = ModelGenerator.get_random_model(10)
            trajectories = pip.simulator.simulate(model)
            mi_score = pip.estimator.estimate(model, trajectories)

            # If close to boundary, then repeat to ensure no errors occur
            if mi_score < threshold:
                for _ in range(num_trials):
                    mi_score = pip.estimator.estimate(model, trajectories)
