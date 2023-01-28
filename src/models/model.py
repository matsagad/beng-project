from typing import Any, List
from models.rates.function import RateFunction

from nptyping import NDArray, Shape, Float
import numpy as np
from scipy.linalg import expm


class PromoterModel:
    def __init__(self, rate_fn_matrix: List[List[RateFunction]]):
        """
        Args:
            rate_fn_matrix  : a 2D matrix of rate functions; diagonals are
                              expected to be None and adjusted later to get
                              a row sum of zero.
        """
        self.rate_fn_matrix = rate_fn_matrix

    def get_generator(
        self, exogenous_data: NDArray[Shape["Any, Any"], Float]
    ) -> NDArray[Shape["Any, Any, Any"], Float]:
        length = exogenous_data.shape[1]

        generator = np.stack(
            [
                np.stack(
                    [
                        np.zeros(length) if not rate_fn else rate_fn(exogenous_data)
                        for rate_fn in row
                    ],
                    axis=1,
                )
                for row in self.rate_fn_matrix
            ],
            axis=1,
        )
        for i in range(length):
            np.fill_diagonal(generator[i, :, :], -np.sum(generator[i], axis=1))
        return generator

    def get_transition(
        self, exogenous_data: NDArray[Shape["Any, Any"], Float], tau: float
    ) -> NDArray[Shape["Any, Any, Any"], Float]:
        return expm(tau * self.get_generator(exogenous_data))
