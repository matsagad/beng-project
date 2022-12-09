from typing import Callable
import numpy as np


class RateFunction:
    @staticmethod
    def constant(a: float) -> Callable[[np.ndarray], float]:
        return lambda _: a

    @staticmethod
    def simple(a: float, index: int) -> Callable[[np.ndarray], float]:
        return lambda x: a * x[index]

    @staticmethod
    def hill(a: float, b: float, index: int) -> Callable[[np.ndarray], float]:
        return lambda x: (a * x[index] / b) / (1 + x[index] / b)
