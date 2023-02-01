from models.model import PromoterModel
from ssa.simulator import StochasticSimulator
from nptyping import NDArray, Shape, Float
import numpy as np


class OneStepSimulator(StochasticSimulator):
    """
    Simulates the promoter trajectory through the one-step Master equation
    after reducing the case to a continuous-time Markov chain.
    """

    def __init__(
        self,
        exogenous_data: NDArray[Shape["Any, Any, Any, Any"], Float],
        tau: float,
        realised: bool = False,
    ):
        self.exogenous_data = exogenous_data
        self.num_classes, _, self.batch_size, _ = self.exogenous_data.shape
        self.tau = tau
        self.realised = realised

    def simulate(
        self, model: PromoterModel
    ) -> NDArray[Shape["Any, Any, Any, Any, Any"], Float]:

        # Initialise state
        state = np.tile(
            model.realised_init_state if self.realised else model.init_state,
            (self.num_classes, self.batch_size, 1),
        )
        ## dimensions are: # of classes, batch size, # of states

        # Pre-calculate matrix exponentials for all time points and batches
        # (shift axes to allow ease in enumeration)
        TIME_AXIS = 2
        matrix_exps = np.moveaxis(
            model.get_matrix_exp(self.exogenous_data, self.tau), TIME_AXIS, 0
        )
        ## dimensions are: # of times, # of classes, batch size, # of states, # of states

        states = [state]

        # Find probability distribution trajectory
        if not self.realised:
            for matrix_exp in matrix_exps:
                # Multiply state: P(0) and matrix_exp: e^At
                # (use einsum to multiply tensors and allow batching)
                prob_dist = np.einsum("ijk,ijkl->ijl", state, matrix_exp)
                state = prob_dist
                states.append(state)

            return np.array(states)

        # Find realised trajectory
        ## Sample random numbers in batches
        rand_mats = np.random.uniform(
            size=(len(matrix_exps), self.num_classes, self.batch_size)
        )

        STATE_AXIS = 2
        for (matrix_exp, rand_mat) in zip(matrix_exps, rand_mats):
            prob_dist = np.einsum("ijk,ijkl->ijl", state, matrix_exp)

            chosen = []
            for env_id, env_pair in enumerate(
                zip(np.cumsum(prob_dist, axis=STATE_AXIS), rand_mat)
            ):
                env, rand_vec = env_pair
                for batch_id, batch_pair in enumerate(zip(env, rand_vec)):
                    batch, rand_num = batch_pair
                    chosen.append((env_id, batch_id, batch.searchsorted(rand_num)))
            
            # Update the state
            state = np.zeros(state.shape)

            state[tuple(np.array(chosen).T)] = 1

            states.append(state)

        return np.array(states)
