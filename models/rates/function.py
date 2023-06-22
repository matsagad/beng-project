from typing import List, Tuple
from abc import ABC, abstractmethod, abstractproperty
from nptyping import NDArray
import numpy as np
import copy

class RateFunction:
    class Function(ABC):
        def __init__(self, rates: List[float], tfs: List[int] = []):
            self.rates = rates
            self.tfs = tfs

        @abstractmethod
        def evaluate(self, exo_states: NDArray) -> float:
            pass

        @abstractmethod
        def str(self, with_rates: bool = True) -> str:
            pass

        @staticmethod
        @abstractmethod
        def num_params() -> Tuple[int, int]:
            pass

        def _str(self) -> str:
            return f"{self.__class__.__name__}({self.rates}, {self.tfs})"

        def __hash__(self) -> int:
            return hash(
                (*map(ord, self.__class__.__name__), tuple(self.rates), tuple(self.tfs))
            )
        
        def copy(self) -> "RateFunction.Function":
            return self.__class__(copy.copy(self.rates), copy.copy(self.tfs))
        
        def __eq__(self, __value: object) -> bool:
            if not isinstance(__value, self.__class__):
                return False
            return self.rates == __value.rates and self.tfs == __value.tfs

    class Constant(Function):
        def evaluate(self, exo_states: NDArray) -> float:
            return self.rates[0] + np.zeros(exo_states.shape[1:])

        def str(self, with_rates: bool = True) -> str:
            return f"${self.rates[0]:.3f}$" if with_rates else "$const$"

        def num_params() -> Tuple[int, int]:
            return 1, 0

    class Linear(Function):
        def evaluate(self, exo_states: NDArray) -> float:
            return sum(rate * exo_states[tf] for rate, tf in zip(self.rates, self.tfs))

        def str(self, with_rates: bool = True) -> str:
            return (
                "$"
                + "+".join(
                    f"{rate:.3f} \cdot \mathrm{{TF}}_{{{tf}}}" if with_rates else f"TF_{{{tf}}}"
                    for rate, tf in zip(self.rates, self.tfs)
                )
                + "$"
            )

        def num_params() -> Tuple[int, int]:
            return 1, 1

    class Hill(Function):
        def evaluate(self, exo_states: NDArray) -> float:
            return (self.rates[0] * exo_states[self.tfs[0]]) / (
                self.rates[1] + exo_states[self.tfs[0]]
            )

        def str(self, with_rates: bool = True) -> str:
            return (
                (
                    f"$\\frac{{{self.rates[0]:.3f} \cdot \mathrm{{TF}}_{{{self.tfs[0]}}}}}"
                    + f"{{{self.rates[1]:.3f} + \mathrm{{TF}}_{{{self.tfs[0]}}}}}$"
                )
                if with_rates
                else f"$Hill(TF_{{{self.tfs[0]}}})$"
            )

        def num_params() -> Tuple[int, int]:
            return 2, 1
