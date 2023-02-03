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
        replicates: int = 1,
    ):
        self.exogenous_data = exogenous_data
        self.num_classes, _, self.batch_size, self.num_times = self.exogenous_data.shape
        self.tau = tau
        self.realised = realised
        self.replicates = replicates

    def simulate(
        self, model: PromoterModel
    ) -> NDArray[Shape["Any, Any, Any, Any, Any"], Float]:

        # Pre-calculate matrix exponentials for all time points and batches
        # (shift axes to allow ease in enumeration)
        TIME_AXIS = 2
        matrix_exps = np.moveaxis(
            model.get_matrix_exp(self.exogenous_data, self.tau), TIME_AXIS, 0
        )
        ## dimensions are: # of times, # of classes, batch size, # of states, # of states

        # Find probability distribution trajectory
        if not self.realised:
            # Initialise state
            state = np.tile(
                model.init_state,
                (self.num_classes, self.batch_size, 1),
            )
            ## dimensions are: # of classes, batch size, # of states
            states = [state]

            for matrix_exp in matrix_exps:
                # Multiply state: P(0) and matrix_exp: e^At
                # (use einsum to multiply tensors and allow batching)
                prob_dist = np.einsum("ijk,ijkl->ijl", state, matrix_exp)
                state = prob_dist
                states.append(state)

            return np.array(states)

        # Find realised trajectory
        num_model_states = len(model.init_state)

        ## Randomly initialise state
        init_cdf = np.cumsum(model.init_state)
        init_chosen = init_cdf.searchsorted(
            np.random.uniform(size=(self.num_classes, self.batch_size, self.replicates))
        )
        state = np.zeros(
            (self.num_classes, self.batch_size, self.replicates, num_model_states)
        )
        state[
            tuple(
                np.array(
                    [
                        (*indices, chosen)
                        for indices, chosen in np.ndenumerate(init_chosen)
                    ]
                ).T
            )
        ] = 1
        ## dimensions are: # of classes, batch size, # of replicates, # of states

        states = [state]

        ## Sample random numbers in batches
        rand_mats = np.random.uniform(
            size=(self.num_times, self.num_classes, self.batch_size)
        )

        STATE_AXIS = 3
        for (matrix_exp, rand_mat) in zip(matrix_exps, rand_mats):
            prob_dist = np.einsum("ijkl,ijlm->ijkm", state, matrix_exp)
            chosen = []
            for env_id, env_pair in enumerate(
                zip(np.cumsum(prob_dist, axis=STATE_AXIS), rand_mat)
            ):
                env, rand_vec = env_pair
                for batch_id, batch_pair in enumerate(zip(env, rand_vec)):
                    batch, rand_num = batch_pair
                    chosen.extend(
                        (
                            env_id,
                            batch_id,
                            replicate_id,
                            replicate.searchsorted(rand_num),
                        )
                        for replicate_id, replicate in enumerate(batch)
                    )

            # Update the state
            state = np.zeros(state.shape)
            state[tuple(np.array(chosen).T)] = 1
            states.append(state)

        states = np.array(states)
        return np.array(states).reshape((*(states.shape[:-3]), -1, states.shape[-1]))
