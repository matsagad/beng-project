from typing import Callable
from models.model import PromoterModel
import math


class ModelPenalty:
    MAX_MI = 2
    MIN_STATES = 2
    MIN_EDGES = 2

    def _old_state_penalty(k: float = 6.0) -> Callable:
        return lambda model, mi: max(
            0, mi - ModelPenalty.MAX_MI / (1 + k * math.exp(k - model.num_states))
        )

    def state_penalty(m: float = 8.0, n: float = 6.0) -> Callable:
        return lambda model, mi: max(
            0,
            mi
            - ModelPenalty.MAX_MI
            * (
                1 - math.exp(-(((model.num_states - ModelPenalty.MIN_STATES) / m) ** n))
            ),
        )

    def balanced_state_penalty(
        target_state: int = 5, m: float = 3.0, n: float = 2.0
    ) -> Callable:
        return lambda model, mi: max(
            0,
            mi
            - ModelPenalty.MAX_MI
            * (1 - math.exp(-((((model.num_states - target_state) ** 2) / m) ** n))),
        )

    def edge_penalty(m: float = 16.0, n: float = 2.0) -> Callable:
        return lambda model, mi: max(
            0,
            mi
            - ModelPenalty.MAX_MI
            * (1 - math.exp(-(((model.num_edges - ModelPenalty.MIN_EDGES) / m) ** n))),
        )

    def erdos_penalty(beta: float) -> Callable:
        norm_term = max(-math.log(beta), -math.log(1 - beta))

        def _penalty(model: PromoterModel, mi: float) -> float:
            vertices = model.num_states

            edges = model.num_edges
            complete_edges = math.comb(vertices, 2)

            penalty = (
                ModelPenalty.MAX_MI
                * (
                    -edges * math.log(beta)
                    - (complete_edges - edges) * math.log(1 - beta)
                )
                / (complete_edges * norm_term)
            )
            return max(0, mi - penalty)

        return _penalty
