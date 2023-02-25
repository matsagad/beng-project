from typing import List
from models.generator import ModelGenerator
from models.model import PromoterModel
import copy
import numpy as np


class GeneticOperator:
    ABS_RATE_BOUND = 2
    RATE_SCALE = 0.1
    TF_COUNT = 5

    class Mutation:
        def add_noise(model: PromoterModel, p: float = 0.8) -> PromoterModel:
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
            if len(rate_fns) == 0:
                return model
            max_num_rates = max(len(rate_fn.rates) for rate_fn in rate_fns)

            rates = np.empty((len(rate_fns), max_num_rates))
            rates[:] = np.nan
            for i, rate_fn in enumerate(rate_fns):
                rates[i, : len(rate_fn.rates)] = rate_fn.rates

            # Apply scaled Gaussian noise
            _rates = rates[~np.isnan(rates)]
            modified_rates = np.log10(
                _rates
            ) + GeneticOperator.RATE_SCALE * np.random.normal(0, 1, len(_rates))
            out_of_bounds = np.absolute(modified_rates) > GeneticOperator.ABS_RATE_BOUND
            modified_rates[out_of_bounds] = np.sign(modified_rates[out_of_bounds]) * (
                GeneticOperator.ABS_RATE_BOUND
                - (
                    np.absolute(modified_rates[out_of_bounds])
                    / (2 * GeneticOperator.ABS_RATE_BOUND)
                )
                % 1.0
            )
            rates[~np.isnan(rates)] = 10**modified_rates

            # Update rates
            for new_rates, rate_fn in zip(rates, rate_fns):
                rate_fn.rates = list(new_rates[: len(rate_fn.rates)])

            return model

        def flip_tf(model: PromoterModel, p: float = 0.4) -> PromoterModel:
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
                    np.random.choice(GeneticOperator.TF_COUNT, len(rate_fn.tfs))
                )

            return model

        def flip_activity(model: PromoterModel, p: float = 0.2) -> PromoterModel:
            num_states = len(model.rate_fn_matrix)
            # Randomly flip activity of node by some probability
            # Find XOR of Bernoulli sample and current active states
            model.active_states = model.active_states != np.random.binomial(
                1, p, num_states
            ).astype(bool)

            return model

        def _modify_edge(
            model: PromoterModel,
            p: float = 0.1,
            existing: bool = False,
            reversible: bool = True,
        ) -> None:
            none_indices = np.array(
                [
                    (i, j)
                    for (i, row) in enumerate(model.rate_fn_matrix)
                    for (j, rate_fn) in enumerate(row)
                    if (existing and rate_fn is not None)
                    or (
                        not existing
                        and rate_fn is None
                        and ((reversible and i > j) or (not reversible and i != j))
                    )
                ],
                dtype=object,
            )
            none_indices = none_indices[
                np.random.binomial(1, p, len(none_indices)) == 1
            ]

            for i, j in none_indices:
                model.rate_fn_matrix[i][j] = ModelGenerator.get_random_rate_fn()
                # Create edge in both directions to make reaction reversible
                if reversible:
                    model.rate_fn_matrix[j][i] = ModelGenerator.get_random_rate_fn()

        def add_edge(
            model: PromoterModel, p: float = 0.2, reversible: bool = True
        ) -> PromoterModel:
            GeneticOperator.Mutation._modify_edge(model, p, False, reversible)
            return model

        def edit_edge(model: PromoterModel, p: float = 0.4) -> PromoterModel:
            GeneticOperator.Mutation._modify_edge(model, p, True, False)
            return model

    class Crossover:
        def swap_rows(
            model: PromoterModel,
            other: PromoterModel,
            model_is_elite: bool = False,
            other_is_elite: bool = False,
        ) -> List[PromoterModel]:
            num_states = len(model.rate_fn_matrix)
            # Treat rows as chromosomes and perform one-point crossover
            # Randomly choose splitting point
            split = int(
                1 if num_states == 2 else 1 + np.random.choice(num_states - 2, 1)
            )

            offspring = []
            # If model is not elite then it is no longer used in the next generation
            # and thus its rate functions can be reused. If it is an elite, then its
            # rate functions should be copied to avoid side-effects from mutations.
            model_copy = copy.deepcopy if model_is_elite else lambda x: x
            other_copy = copy.deepcopy if other_is_elite else lambda x: x
            _models = [(model, model_copy), (other, other_copy)]

            # Generate offspring
            for (fst, fst_copy), (snd, snd_copy) in zip(_models, _models[::-1]):
                offspring.append(
                    PromoterModel(
                        fst_copy(fst.rate_fn_matrix[:split])
                        + snd_copy(snd.rate_fn_matrix[split:])
                    ).with_active_states(
                        np.concatenate(
                            (
                                fst_copy(fst.active_states[:split]),
                                snd_copy(snd.active_states[split:]),
                            )
                        )
                    )
                )

            return offspring
