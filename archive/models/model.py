from abc import ABC, abstractmethod
import numpy as np


class PromoterModel(ABC):
    def __init__(self):
        self.system = None

    def composition_matrix(self) -> np.ndarray:
        """Finds the composition matrix of the system.

        Columns are indexed by lexicographical ordering of substance symbols.

        Returns:
          A numpy array of the composition matrix.
        """
        return self.system.composition_matrix if self.system else None

    def propensity(self, state: np.ndarray) -> np.ndarray:
        """Produces the propensities of the system.

        The propensity of a reaction is defined as the product of
        its specific probability rate and its reaction degeneracy.
        (Ostrenko et al., 2017)

        Args:
            state : a vector of the system's state, i.e. ([A], [I], [M], [P])

        Returns:
            A unit vector of propensities.
        """
        return self.system.propensity(state)
