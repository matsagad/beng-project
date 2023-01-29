from typing import Callable
from abc import ABC, abstractmethod
from nptyping import NDArray
import numpy as np


class RateFunction:
    class Function(ABC):
        @abstractmethod
        def evaluate(self, exo_states: NDArray) -> float:
            pass

        @abstractmethod
        def str(self) -> str:
            pass

    class Constant(Function):
        def __init__(self, a: float):
            self.a = a

        def evaluate(self, exo_states: NDArray) -> float:
            return self.a + np.zeros(exo_states.shape[1:])

        def str(self) -> str:
            return f"{self.a}"

    class Linear(Function):
        def __init__(self, a: float, exo_index: int):
            self.a = a
            self.exo_index = exo_index

        def evaluate(self, exo_states: NDArray) -> float:
            return self.a * exo_states[self.exo_index]

        def str(self) -> str:
            return f"{self.a} * TF[{self.exo_index}]"

    class Hill(Function):
        def __init__(self, a: float, b: float, exo_index: int):
            self.a = a
            self.b = b
            self.exo_index = exo_index

        def evaluate(self, exo_states: NDArray) -> float:
            return (self.a * exo_states[self.exo_index]) / (
                self.b + exo_states[self.exo_index]
            )

        def str(self) -> str:
            return f"({self.a} * TF[{self.exo_index}]) / ({self.b} + TF[{self.exo_index}])"
