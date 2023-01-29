from typing import List
from models.rates.function import RateFunction
from nptyping import NDArray, Shape, Float
from scipy.linalg import expm
import numpy as np


class PromoterModel:
    def __init__(self, rate_fn_matrix: List[List[RateFunction]]):
        """
        Args:
            rate_fn_matrix  : a 2D matrix of rate functions; diagonals are
                              expected to be None and adjusted later to get
                              a row sum of zero.
        """
        self.rate_fn_matrix = rate_fn_matrix
        self.init_state = [1] + [0] * (len(rate_fn_matrix) - 1)
        self.active_state = -1

    def get_generator(
        self, exogenous_data: NDArray[Shape["Any, Any, Any"], Float]
    ) -> NDArray[Shape["Any, Any, Any"], Float]:
        batch_size = exogenous_data.shape[1]
        rate_fn_len = exogenous_data.shape[1:]
        length = exogenous_data.shape[-1]

        generator = np.stack(
            [
                np.stack(
                    [
                        np.zeros(rate_fn_len)
                        if not rate_fn
                        else rate_fn(exogenous_data)
                        for rate_fn in row
                    ],
                    axis=2,
                )
                for row in self.rate_fn_matrix
            ],
            axis=2,
        )

        for j in range(batch_size):
            for i in range(length):
                np.fill_diagonal(generator[j, i], -np.sum(generator[j, i], axis=1))
        return generator

    def get_matrix_exp(
        self, exogenous_data: NDArray[Shape["Any, Any, Any"], Float], tau: float
    ) -> NDArray[Shape["Any, Any, Any"], Float]:
        return expm(tau * self.get_generator(exogenous_data))
