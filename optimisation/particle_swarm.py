from nptyping import NDArray
from models.model import PromoterModel
from models.rates.function import RateFunction as RF
from pipeline.one_step_decoding import OneStepDecodingPipeline
from pyswarms.single.global_best import GlobalBestPSO
from pyswarms.utils.search import RandomSearch
import numpy as np


class ParticleSwarm:
    _mp_instance = None

    def evaluate_model(X: NDArray, pip):
        models = []
        for particle in 10**X:
            rate_fn_matrix = []
            params_used = 0
            for params, fn_row, fn_tfs in zip(*ParticleSwarm._mp_instance._params):
                params_needed = sum(params)
                param_rates = np.split(
                    particle[params_used : params_used + params_needed],
                    np.cumsum(params),
                )
                rate_fn_matrix.append(
                    [
                        fn(rates, tfs) if fn else None
                        for rates, fn, tfs in zip(param_rates, fn_row, fn_tfs)
                    ]
                )
                params_used += params_needed
            models.append(PromoterModel(rate_fn_matrix))
        return -np.array([pip.evaluate(model) for model in models])

    def optimise(
        self,
        data: NDArray,
        reference_model: PromoterModel,
        n_particles: int = 10,
        n_processes=10,
        iters: int = 10,
        start_at_pos: bool = False,
    ):
        replicates = 10
        classifier = "naive_bayes"

        pip = OneStepDecodingPipeline(
            data,
            realised=True,
            replicates=replicates,
            classifier_name=classifier,
        )

        num_states = reference_model.num_states

        reference_params = []
        params_per_rate_fun = np.zeros((num_states, num_states)).astype(int)
        for i, row in enumerate(reference_model.rate_fn_matrix):
            for j, rate_fun in enumerate(row):
                if not rate_fun:
                    continue
                # Keep TF fixed, only change constants
                params_per_rate_fun[i, j] = len(rate_fun.rates)
                reference_params.extend(rate_fun.rates)
        total_params = np.sum(params_per_rate_fun)

        dims = total_params
        low, high = -2, 2  # log-space
        bounds = (low * np.ones(dims), high * np.ones(dims))
        options = {
            "c1": 0.5,
            "c2": 0.3,
            "w": 0.9,
        }

        rate_fns = [
            [rf.__class__ if rf else None for rf in row]
            for row in reference_model.rate_fn_matrix
        ]
        rate_fn_tfs = [
            [rf.tfs if rf else None for rf in row]
            for row in reference_model.rate_fn_matrix
        ]

        self._params = (params_per_rate_fun, rate_fns, rate_fn_tfs)
        ParticleSwarm._mp_instance = self

        init_pos = np.random.uniform(low, high, size=(n_particles, dims))
        if start_at_pos:
            init_pos = 0.05 * init_pos + np.log10(reference_params)

        optimiser = GlobalBestPSO(
            n_particles=n_particles,
            dimensions=dims,
            options=options,
            bounds=bounds,
            init_pos=init_pos,
        )
        cost, pos = optimiser.optimize(
            ParticleSwarm.evaluate_model, iters=iters, pip=pip, n_processes=n_processes
        )
        return cost, pos

    def _evaluate_model_simple(
        X: NDArray, pip: OneStepDecodingPipeline, tf: int
    ) -> NDArray:
        print(f"evaluating...{X}")
        return -np.array(
            [
                pip.evaluate(
                    PromoterModel(
                        rate_fn_matrix=[
                            [None, RF.Linear([particle[0]], [tf])],
                            [RF.Constant([particle[1]]), None],
                        ]
                    )
                )
                for particle in 10**X
            ]
        )

    def _optimise_simple(self, data: NDArray, tf: int):
        replicates = 10
        classifier = "naive_bayes"

        pip = OneStepDecodingPipeline(
            data,
            realised=True,
            replicates=replicates,
            classifier_name=classifier,
        )

        dims = 2
        low, high = -2, 2  # log-space
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
            ParticleSwarm._evaluate_model_simple,
            iters=20,
            pip=pip,
            tf=tf,
            n_processes=10,
        )

        return cost, pos
