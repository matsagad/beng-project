from typing import Callable, Tuple
from models.model import PromoterModel
from systems.biochemical import BioChemicalSystem
from systems.rates.equation import RatesEquation
from systems.rates.function import RateFunction
import numpy as np


class SimpleModel(PromoterModel):
    _SUBSTANCE_MAP = {sub: index for (index, sub) in enumerate("AIMP")}
    _RATE_EQUATIONS = (
        "A -> I",
        "I -> A",
        "A -> A + M",
        "M -> P",
    )
    # Reactants from the rate equations
    _RATE_DEPENDENTS = list(map(_SUBSTANCE_MAP.get, "AIAM"))

    def __init__(self, rate_fns: Tuple[Callable[[float, np.ndarray], float]] = None):
        self.rate_fns = rate_fns
        self.system = BioChemicalSystem(
            [
                RatesEquation.parse_str(eq, rate_fn)
                for (eq, rate_fn) in zip(
                    self._RATE_EQUATIONS,
                    self.rate_fns,
                )
            ]
        )

    @staticmethod
    def with_rates(rates: Tuple[float, float, float, float]) -> "SimpleModel":
        """Constructs a simple model given constant rate coefficients.

        Args:
          rates: tuple of reaction rates (i.e. (k_off, k_on, k_syn, k_dec))
        """
        return SimpleModel(
            rate_fns=(
                RateFunction.constant(rate, index)
                for rate, index in zip(rates, SimpleModel._RATE_DEPENDENTS)
            )
        )

    @staticmethod
    def simple_with_rates(rates: Tuple[float, float, float, float]) -> "SimpleModel":
        """Constructs a simple model where k_on is proportional to TF concentration.

        Args:
          rates: tuple of reaction rates (i.e. (k_off, k_on, k_syn, k_dec))
        """
        # TODO: make builder patterns DRY
        rate_dependents = SimpleModel._RATE_DEPENDENTS
        return SimpleModel(
            rate_fns=(
                RateFunction.constant(rates[0], index=rate_dependents[0]),
                RateFunction.simple(rates[1], index=rate_dependents[1], exo_index=0),
                RateFunction.constant(rates[2], index=rate_dependents[2]),
                RateFunction.constant(rates[3], index=rate_dependents[3]),
            )
        )
