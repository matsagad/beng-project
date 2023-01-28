from typing import Callable
import numpy as np


class RateFunction:
    @staticmethod
    def constant(a: float) -> Callable[[np.ndarray, np.ndarray], float]:
        return lambda exo_states: a + np.zeros(exo_states.shape[1:])

    @staticmethod
    def linear(a: float, exo_index: int) -> Callable[[np.ndarray, np.ndarray], float]:
        return lambda exo_state: a * exo_state[exo_index]

    @staticmethod
    def hill(
        a: float, b: float, exo_index: int
    ) -> Callable[[np.ndarray, np.ndarray], float]:
        return lambda exo_state: (
            (a * exo_state[exo_index] / b) / (1 + exo_state[exo_index] / b)
        )
