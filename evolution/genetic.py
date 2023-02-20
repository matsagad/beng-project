from models.model import PromoterModel
from models.rates.function import RateFunction as RF
import numpy as np


class GeneticSetup:
    ABS_RATE_BOUND = 2
    RATE_SCALE = 0.05
    TF_COUNT = 5

    class Mutation:
        def add_noise(model: PromoterModel, p: float = 1.0) -> None:
            # Unpack rate functions
            rate_fns = np.array(
                [
                    rate_fn
                    for row in model.rate_fn_matrix
                    for rate_fn in row
                    if rate_fn is not None
                ],
                dtype=object,
            )

            # Batch rates through a numpy array padded with nans
            rate_fns = rate_fns[np.random.binomial(1, p, len(rate_fns)) == 1]
            max_num_rates = max(len(rate_fn.rates) for rate_fn in rate_fns)

            rates = np.empty((len(rate_fns), max_num_rates))
            rates[:] = np.nan
            for i, rate_fn in enumerate(rate_fns):
                rates[i, : len(rate_fn.rates)] = rate_fn.rates

            # Apply scaled Gaussian noise
            _rates = rates[~np.isnan(rates)]
            modified_rates = np.log10(
                _rates
            ) + GeneticSetup.RATE_SCALE * np.random.normal(0, 1, len(_rates))
            out_of_bounds = np.absolute(modified_rates) > GeneticSetup.ABS_RATE_BOUND
            modified_rates[out_of_bounds] = np.sign(modified_rates[out_of_bounds]) * (
                GeneticSetup.ABS_RATE_BOUND
                - (
                    np.absolute(modified_rates[out_of_bounds])
                    / (2 * GeneticSetup.ABS_RATE_BOUND)
                )
                % 1.0
            )
            rates[~np.isnan(rates)] = 10**modified_rates

            # Update rates
            for new_rates, rate_fn in zip(rates, rate_fns):
                rate_fn.rates = list(new_rates[: len(rate_fn.rates)])

        def flip_tf(model: PromoterModel, p: float = 1.0) -> None:
            pass

        def flip_activity(model: PromoterModel, p: float = 1.0) -> None:
            pass

        def change_rate(model: PromoterModel) -> None:
            pass

    class Crossover:
        def swap_inputs(model: PromoterModel, other: PromoterModel) -> PromoterModel:
            pass
