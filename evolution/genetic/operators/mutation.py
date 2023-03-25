from models.generator import ModelGenerator
from models.model import PromoterModel
import numpy as np


class MutationOperator:
    ABS_RATE_BOUND = 2
    RATE_SCALE = 0.1
    TF_COUNT = 5

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
        ) + MutationOperator.RATE_SCALE * np.random.normal(0, 1, len(_rates))
        out_of_bounds = np.absolute(modified_rates) > MutationOperator.ABS_RATE_BOUND
        modified_rates[out_of_bounds] = np.sign(modified_rates[out_of_bounds]) * (
            MutationOperator.ABS_RATE_BOUND
            - (
                np.absolute(modified_rates[out_of_bounds])
                / (2 * MutationOperator.ABS_RATE_BOUND)
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
                np.random.choice(MutationOperator.TF_COUNT, len(rate_fn.tfs))
            )

        return model

    def add_activity_noise(model: PromoterModel, p: float = 0.8) -> PromoterModel:
        # Add Gaussian noise.
        noise = np.random.binomial(1, p, model.num_states) * np.random.normal(
            0, 1, model.num_states
        )
        model.activity_weights += noise

        # If out of bounds, then "bounce" values against the boundaries.
        out_of_bounds = (model.activity_weights < 0) & (model.activity_weights > 1)
        dec_part, int_part = np.modf(model.activity_weights[out_of_bounds])
        model.activity_weights[out_of_bounds] = (int_part % 2 == 0) * np.abs(
            dec_part
        ) + (int_part % 2 == 1) * (1 - np.abs(dec_part))

        return model

    def flip_activity(model: PromoterModel, p: float = 0.2) -> PromoterModel:
        # Randomly flip activity of node by some probability (except the first state)
        # If flipped from inactive to active, then initialise with random uniform weight.
        to_flip = np.random.binomial(1, p, model.num_states - 1).astype(bool)
        active = model.activity_weights[1:] > 0

        model.activity_weights[1:][to_flip & active] = 0
        model.activity_weights[1:][to_flip & ~active] = np.random.uniform(
            0, 1, np.sum(to_flip & ~active)
        )

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
        none_indices = none_indices[np.random.binomial(1, p, len(none_indices)) == 1]

        for i, j in none_indices:
            model.rate_fn_matrix[i][j] = ModelGenerator.get_random_rate_fn()
            # Create edge in both directions to make reaction reversible
            if reversible:
                model.rate_fn_matrix[j][i] = ModelGenerator.get_random_rate_fn()

    def add_edge(
        model: PromoterModel, p: float = 0.2, reversible: bool = True
    ) -> PromoterModel:
        MutationOperator._modify_edge(model, p, False, reversible)
        return model

    def edit_edge(model: PromoterModel, p: float = 0.4) -> PromoterModel:
        MutationOperator._modify_edge(model, p, True, False)
        return model
