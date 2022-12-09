from abc import ABC, abstractmethod
import numpy as np


class PromoterModel(ABC):
    _TRANSCRIPTION_FACTORS = {
        sub: index for (index, sub) in enumerate(["dot6", "mig1", "sfp1"])
    }

    def __init__(self):
        self.system = None

    def composition_matrix(self) -> np.ndarray:
        """Finds the composition matrix of the system.

        Columns are indexed by lexicographical ordering of substance symbols.

        Returns:
          A numpy array of the composition matrix.
        """
        return self.system.composition_matrix if self.system else None

    def propensity(self, state: np.ndarray, exogenous_state: np.ndarray) -> np.ndarray:
        """Produces the propensities of the system.

        The propensity of a reaction is defined as the product of
        its specific probability rate and its reaction degeneracy.
        (Ostrenko et al., 2017)

        Args:
            state           : a vector of the system's state, i.e. [[A], [I], [M], [P]]
            exogenous_state : a vector of an exogenous system's state, i.e. [[TF1], [TF2] [TF3]]

        Returns:
            A unit vector of propensities.
        """
        return self.system.propensity(state, exogenous_state)
