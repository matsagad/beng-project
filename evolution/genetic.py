from typing import List
from models.model import PromoterModel
from models.rates.function import RateFunction
import numpy as np


class GeneticSetup:
    ABS_RATE_BOUND = 2
    RATE_SCALE = 0.05
    TF_COUNT = 5

    class Mutation:
        def add_noise(model: PromoterModel, p: float = 0.8) -> None:
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

        def flip_tf(model: PromoterModel, p: float = 0.1) -> None:
            rate_fns = np.array(
                [
                    rate_fn
                    for row in model.rate_fn_matrix
                    for rate_fn in row
                    if rate_fn is not None and rate_fn.tfs
                ],
                dtype=object,
            )
            # For some rate functions, randomly select TFs to be used as input.
            for rate_fn in rate_fns[np.random.binomial(1, p, len(rate_fns)) == 1]:
                rate_fn.tfs = list(
                    np.random.choice(GeneticSetup.TF_COUNT, len(rate_fn.tfs))
                )

        def flip_activity(model: PromoterModel, p: float = 0.1) -> None:
            num_states = len(model.rate_fn_matrix)
            # Randomly flip activity of node by some probability
            # Find XOR of Bernoulli sample and current active states
            model.active_states = model.active_states != np.random.binomial(
                1, p, num_states
            ).astype(bool)

        def _get_random_rate_fn() -> RateFunction:
            pass

        def _modify_edge(
            model: PromoterModel, p: float = 0.1, existing: bool = False
        ) -> None:
            none_indices = np.array(
                [
                    (i, j)
                    for (i, row) in enumerate(model.rate_fn_matrix)
                    for (j, rate_fn) in enumerate(row)
                    if (not existing and rate_fn is None and i != j)
                    or (existing and rate_fn is not None)
                ],
                dtype=object,
            )
            none_indices = none_indices[
                np.random.binomial(1, p, len(none_indices)) == 1
            ]

            for i, j in none_indices:
                model.rate_fn_matrix[i][j] = GeneticSetup.Mutation._get_random_rate_fn()

        def add_edge(model: PromoterModel, p: float = 0.1) -> None:
            GeneticSetup.Mutation._modify_edge(model, p, False)

        def flip_edge(model: PromoterModel, p: float = 0.01) -> None:
            GeneticSetup.Mutation._modify_edge(model, p, True)

    class Crossover:
        def swap_rows(
            model: PromoterModel, other: PromoterModel
        ) -> List[PromoterModel]:
            num_states = len(model.rate_fn_matrix)
            # Treat rows as chromosomes and perform one-point crossover
            # Randomly choose splitting point
            split = 1 + np.random.choice(num_states - 2, 1)

            offspring = []
            # Generate offspring
            for fst, snd in [(model, other), (other, model)]:
                offspring.append(
                    PromoterModel(
                        fst.rate_fn_matrix[:split] + snd.rate_fn_matrix[split:]
                    ).with_active_states(
                        np.concatenate(
                            fst.active_states[:split], snd.active_states[split:]
                        )
                    )
                )

            return offspring
