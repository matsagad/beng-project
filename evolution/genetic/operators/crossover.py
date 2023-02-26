from typing import List
from models.model import PromoterModel
import copy
import numpy as np


class CrossoverOperator:
    def one_point_row_swap(
        model: PromoterModel,
        other: PromoterModel,
        model_is_elite: bool = False,
        other_is_elite: bool = False,
    ) -> List[PromoterModel]:
        """
        Performs a one-point crossover on the rows of the adjacency matrices of two
        promoter models. Note: this does not preserve the reversibility of reactions.
        """
        num_states = len(model.rate_fn_matrix)
        # Randomly choose splitting point
        split = int(1 if num_states == 2 else 1 + np.random.choice(num_states - 2, 1))

        # If model is not elite then it is no longer used in the next generation
        # and thus its rate functions can be reused. If it is an elite, then its
        # rate functions should be copied to avoid side-effects from mutations.
        model_copy = copy.deepcopy if model_is_elite else lambda x: x
        other_copy = copy.deepcopy if other_is_elite else lambda x: x
        _models = [(model, model_copy), (other, other_copy)]

        # Generate offspring
        return [
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
            for (fst, fst_copy), (snd, snd_copy) in zip(_models, _models[::-1])
        ]

    def one_point_triangular_row_swap(
        model: PromoterModel,
        other: PromoterModel,
        model_is_elite: bool = False,
        other_is_elite: bool = False,
    ) -> List[PromoterModel]:
        """
        Performs a one-point crossover on the rows of the upper-triangular portion of the
        adjacency matrices of two promoter models. By swapping an edge (i, j), we also
        include the edge (j, i) in the lower-triangular half. This preserves reversibility.
        """
        num_states = len(model.rate_fn_matrix)
        # Randomly choose splitting point
        split = int(
            1
            if num_states == 2
            else 1 + np.random.choice(((num_states - 1) * (num_states - 2)) // 2, 1)
        )

        # If model is not elite then it is no longer used in the next generation
        # and thus its rate functions can be reused. If it is an elite, then its
        # rate functions should be copied to avoid side-effects from mutations.
        model_copy = copy.deepcopy if model_is_elite else lambda x: x
        other_copy = copy.deepcopy if other_is_elite else lambda x: x
        _models = [(model, model_copy), (other, other_copy)]

        # Generate offspring
        child1 = np.array(model_copy(model.rate_fn_matrix))
        child2 = np.array(other_copy(other.rate_fn_matrix))

        top_split = tuple(arr[:split] for arr in np.triu_indices(num_states, 1))
        bot_split = top_split[::-1]

        for tri_split in (top_split, bot_split):
            child1[tri_split], child2[tri_split] = (
                child2[tri_split],
                child1[tri_split],
            )

        return [
            PromoterModel(child).with_active_states(
                np.concatenate(
                    (
                        fst_copy(fst.active_states[:split]),
                        snd_copy(snd.active_states[split:]),
                    )
                )
            )
            for ((fst, fst_copy), (snd, snd_copy), child) in zip(
                _models, _models[::-1], (child1, child2)
            )
        ]
