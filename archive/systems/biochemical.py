from typing import List
from systems.equation import RatesEquation
import numpy as np


class BioChemicalSystem:
    def __init__(self, equations: List[RatesEquation]):
        self.equations = equations
        self.elements = sorted(set().union(*(eq.elements for eq in equations)))

    def composition_matrix(self) -> np.ndarray:
        """Finds the composition matrix of the system.

        Returns:
          A numpy array of the composition matrix.
        """
        count_map = {el: [] for el in self.elements}

        for eq in self.equations:
            for el in self.elements:
                count_map[el].append(eq.products.get(el, 0) - eq.reactants.get(el, 0))

        return np.transpose(list(count_map.values()))
