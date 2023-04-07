from models.model import PromoterModel
from models.preset import Preset
from mi_estimation.decoding import DecodingEstimator
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
        classifier = "naive_bayes"
        replicates = 10

        est = DecodingEstimator(
            self.origin, self.interval, classifier, replicates=replicates
        )
        est.parallel = True

        num_envs, _, num_cells, num_times = self.data.shape
        trajectory = np.zeros((num_times, num_envs, num_cells * replicates, 2))

        # Rich state left as is. Note: first state is active state
        origin = self.origin
        trajectory[origin : origin + origin // 4, 0, :, 0] = 1
        trajectory[origin : origin + 2 * origin // 4, 1, :, 0] = 1
        trajectory[origin : origin + 3 * origin // 4, 2, :, 0] = 1

        model = PromoterModel.dummy()
        mi_score = est.estimate(model, trajectory)

        assert mi_score == 2.0, "Projected MI != 2.0"
