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
        out_of_bounds = (model.activity_weights < 0) | (model.activity_weights > 1)
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

    def add_vertex(
        model: PromoterModel, p: float = 0.1, p_edge: float = 0.25
    ) -> PromoterModel:
        if not bool(np.random.binomial(1, p)):
            return model
        rate_fn_matrix = model.rate_fn_matrix
        num_states = len(rate_fn_matrix)

        connect_to_state = np.zeros(num_states, dtype=bool)

        # Randomly choose one state to connect to
        connect_to_state[np.random.choice(num_states)] = True
        # Randomly choose other states to connect to
        connect_to_state[np.random.binomial(1, p_edge, num_states)] = True

        for row, connect in zip(rate_fn_matrix, connect_to_state):
            row.append(
                ModelGenerator.get_random_rate_fn(num_tfs=MutationOperator.TF_COUNT)
                if connect
                else None
            )

        rate_fn_matrix.append(
            [
                ModelGenerator.get_random_rate_fn(num_tfs=MutationOperator.TF_COUNT)
                if connect
                else None
                for connect in connect_to_state
            ]
            + [None]
        )
        model.activity_weights = np.append(model.activity_weights, np.random.uniform())

        # Update model stats
        model.num_states += 1
        model.num_edges += 2 * np.sum(connect_to_state)
        model.init_state = np.ones(len(rate_fn_matrix), dtype=int) / len(rate_fn_matrix)

        return model

    def remove_vertex(model: PromoterModel, p: float = 0.1) -> PromoterModel:
        if model.num_states == 2 or not bool(np.random.binomial(1, p)):
            return model

        rate_fn_matrix = model.rate_fn_matrix
        connected_states = set(
            i for i, rate_fn in enumerate(rate_fn_matrix[-1]) if rate_fn is not None
        )

        # Remove last state
        for i, row in enumerate(rate_fn_matrix):
            rate_fn_matrix[i] = row[:-1]
        rate_fn_matrix.pop(-1)

        # Union-find for which neighbors are connected
        groups = {}

        count = 0
        for state in connected_states:
            if state not in groups:
                groups[state] = count
                for i, rate_fn in enumerate(rate_fn_matrix[state]):
                    if i in connected_states and rate_fn is not None:
                        groups[i] = count
                count += 1

        state_groups = [[] for _ in range(count)]
        for state, group in groups.items():
            state_groups[group].append(state)
        np.random.shuffle(state_groups)

        for group, next_group in zip(state_groups, state_groups[1:]):
            state1, state2 = group[0], next_group[0]
            rate_fn_matrix[state1][state2] = ModelGenerator.get_random_rate_fn(
                num_tfs=MutationOperator.TF_COUNT
            )
            rate_fn_matrix[state2][state1] = ModelGenerator.get_random_rate_fn(
                num_tfs=MutationOperator.TF_COUNT
            )

        model.activity_weights = model.activity_weights[:-1]

        # Update model stats
        model.num_states -= 1
        model.num_edges = sum(sum(map(bool, row)) for row in rate_fn_matrix)
        model.init_state = np.ones(len(rate_fn_matrix), dtype=int) / len(rate_fn_matrix)

        return model
