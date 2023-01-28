from typing import Callable
import numpy as np


class RateFunction:
    @staticmethod
    def constant(a: float, index: int) -> Callable[[np.ndarray, np.ndarray], float]:
        return lambda state, _: a * state[index]

    @staticmethod
    def simple(
        a: float, index: int, exo_index: int
    ) -> Callable[[np.ndarray, np.ndarray], float]:
        return lambda state, exo_state: a * exo_state[exo_index] * state[index]

    @staticmethod
    def hill(
        a: float, b: float, index: int, exo_index: int
    ) -> Callable[[np.ndarray, np.ndarray], float]:
        return (
            lambda state, exo_state: state[index]
            * (a * exo_state[exo_index] / b)
            / (1 + exo_state[exo_index] / b)
        )
