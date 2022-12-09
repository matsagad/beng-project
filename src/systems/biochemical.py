from typing import List
from systems.rates.equation import RatesEquation
import numpy as np


class BioChemicalSystem:
    def __init__(self, equations: List[RatesEquation]):
        self.equations = equations
        self.elements = sorted(set().union(*(eq.elements for eq in equations)))

        self.composition_matrix = self._composition_matrix()
        # Only concerned with magnitude of negative entries (i.e. for degeneracy)
        self.propensity_matrix = -self.composition_matrix * (
            self.composition_matrix < 0
        )

    def _composition_matrix(self) -> np.ndarray:
        """Finds the composition matrix of the system.

        Returns:
          A numpy array of the composition matrix.
        """
        count_map = {el: [] for el in self.elements}

        for eq in self.equations:
            for el in self.elements:
                count_map[el].append(eq.products.get(el, 0) - eq.reactants.get(el, 0))

        return np.transpose(list(count_map.values()))

    def propensity(self, state: np.ndarray, exogenous_state: np.ndarray) -> np.ndarray:
        """Produces the propensities of the system.

        Args:
            state           : a vector of the system's state, i.e. [[A], [I], [M], [P]]
            exogenous_state : a vector of an exogenous system's state, i.e. [[TF1], [TF2] [TF3]]

        Returns:
            A unit vector of propensities.
        """
        return np.array([eq.rate_fn(state, exogenous_state) for eq in self.equations])

    def __str__(self) -> str:
        return "\n".join(str(eq) for eq in self.equations)
