from typing import Callable, Tuple
from models.model import PromoterModel
from systems.biochemical import BioChemicalSystem
from systems.rates.equation import RatesEquation
from systems.rates.function import RateFunction
import numpy as np


class SimpleModel(PromoterModel):
    def __init__(self, rate_fns: Tuple[Callable[[float, np.ndarray], float]] = None):
        self.rate_fns = rate_fns
        self.system = BioChemicalSystem(
            [
                RatesEquation.parse_str(eq, rate_fn)
                for (eq, rate_fn) in zip(
                    (
                        "I -> A",
                        "A -> I",
                        "A -> A + M",
                        "M -> P",
                    ),
                    self.rate_fns,
                )
            ]
        )

    @staticmethod
    def with_rates(rates: Tuple[float, float, float, float]) -> "SimpleModel":
        """Constructs a simple model given constant rate coefficients.

        Args:
          rates: tuple of reaction rates (i.e. (k_on, k_off, k_syn, k_dec))
        """
        return SimpleModel(rate_fns=(RateFunction.constant(rate) for rate in rates))
