from nptyping import NDArray
from models.model import PromoterModel
from models.rates.function import RateFunction as RF
from pipeline.one_step_decoding import OneStepDecodingPipeline
from pyswarms.single.global_best import GlobalBestPSO
from pyswarms.utils.search import RandomSearch
import numpy as np


class ParticleSwarm:
    def evaluate_model(X: NDArray, pip):
        print(f"evaluating...{X}")
        return -np.array(
            [
                pip.evaluate(
                    PromoterModel(
                        rate_fn_matrix=[
                            [None, None, RF.Constant(particle[0])],
                            [None, None, RF.Constant(particle[1])],
                            [
                                RF.Linear(particle[2], 1),
                                RF.Linear(particle[3], 2),
                                None,
                            ],
                        ]
                    ).with_active_states([0, 1])
                )
                for particle in 10**X
            ]
        )

    def optimise_simple(self, data: NDArray):
        replicates = 5
        classifier = "naive_bayes"

        pip = OneStepDecodingPipeline(
            data,
            realised=True,
            replicates=replicates,
            classifier_name=classifier,
        )

        dims = 4
        low, high = -4, 4  # log-space
        bounds = (low * np.ones(dims), high * np.ones(dims))
        options = {
            "c1": 0.5,
            "c2": 0.3,
            "w": 0.9,
        }

        optimiser = GlobalBestPSO(
            n_particles=10, dimensions=dims, options=options, bounds=bounds
        )

        cost, pos = optimiser.optimize(
            ParticleSwarm.evaluate_model, iters=20, pip=pip, n_processes=10
        )

        print(cost, pos)
