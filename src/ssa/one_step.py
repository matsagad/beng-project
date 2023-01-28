from models.model import PromoterModel
from ssa.simulator import StochasticSimulator
from nptyping import NDArray, Shape, Float
import numpy as np


class OneStepSimulator(StochasticSimulator):
    """
    Simulates the promoter trajectory through the one-step Master equation
    after reducing the case to a continuous-time Markov chain.
    """

    def __init__(self, exogenous_data: NDArray[Shape["Any, Any"], Float], tau: float):
        self.exogenous_data = exogenous_data
        self.batch_size = self.exogenous_data.shape[1]
        self.tau = tau

    def simulate(self, model: PromoterModel) -> NDArray[Shape["Any"], Float]:
        # Initialise state
        state = np.tile(model.init_state, (self.batch_size, 1))

        # Pre-calculate matrix exponentials for all time points and batches
        # (shift axes to allow ease in enumeration)
        matrix_exps = np.moveaxis(
            model.get_matrix_exp(self.exogenous_data, self.tau), 0, -1
        )

        # Sample random numbers in batches
        rand_nums = np.random.uniform(size=(len(matrix_exps), self.batch_size))

        num_states = state.shape[-1]
        states = [state]

        for (matrix_exp, rand_num) in zip(matrix_exps, rand_nums):
            # Multiply state: P(0) and matrix_exp: e^At
            # (einsum used to multiply tensors and allow batching)
            prob_dist = np.einsum("ij,jki->ik", state, matrix_exp)

            # Choose state based on random number and probability distribution
            chosen = list(
                arr.searchsorted(num)
                for arr, num in zip(np.cumsum(prob_dist, axis=1), rand_num)
            )

            # Update the state
            state = np.zeros((self.batch_size, num_states))
            state[np.arange(self.batch_size), chosen] = 1
            states.append(state)

        return np.array(states)
