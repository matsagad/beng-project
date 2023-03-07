from typing import Callable
from models.model import PromoterModel
import math


class ModelPenalty:
    MAX_MI = 2

    def state_penalty(k: float) -> Callable:
        return lambda model, mi: max(
            0, mi - ModelPenalty.MAX_MI / (1 + k * math.exp(k - model.num_states))
        )

    def edge_penalty(k: float) -> Callable:
        return lambda model, mi: max(
            0, mi - ModelPenalty.MAX_MI / (1 + k * math.exp(k - model.num_edges))
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
