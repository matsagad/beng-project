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
        realised: bool = True,
        replicates: int = 1,
    ):
        self.exogenous_data = exogenous_data
        self.num_classes, _, self.batch_size, self.num_times = self.exogenous_data.shape
        self.tau = tau
        self.realised = realised
        self.replicates = replicates
        self.seed = None
        self.binary_search = False
        self.states_threshold = 15

    def simulate(
        self, model: PromoterModel
    ) -> NDArray[Shape["Any, Any, Any, Any, Any"], Float]:
        if model.num_states > self.states_threshold:
            # If number of states is incredibly large, then lazily evaluate the
            # matrix exponentials when they are needed to conserve memory.
            matrix_exps = model.get_matrix_exp_iterable(
                self.exogenous_data[:, :, :, : self.num_times], self.tau
            )
        else:
            # If number of states is tractable, pre-calculate the matrix exponentials
            # in a vectorised manner to speed up the process.
            matrix_exps = model.get_matrix_exp(
                self.exogenous_data[:, :, :, : self.num_times], self.tau
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
        num_model_states = model.num_states

        ## Randomly initialise state
        np_rs = np.random.RandomState(seed=self.seed)
        init_cdf = np.cumsum(model.init_state)
        _init_chosen = init_cdf.searchsorted(
            np_rs.uniform(size=(self.num_classes, self.batch_size, self.replicates))
        )

        init_chosen = np.indices((*_init_chosen.shape, 1))
        init_chosen[-1] = np.expand_dims(_init_chosen, -1)
        ## dimensions are: # of classes, batch size, # of replicates, # of states

        # Additional time for t=0
        states = np.full(
            (
                self.num_times + 1,
                self.num_classes,
                self.batch_size,
                self.replicates,
                num_model_states,
            ),
            False,
        )
        states[0][tuple(init_chosen)] = True

        STATE_AXIS = 3
        for i, matrix_exp in enumerate(matrix_exps):
            prob_dist = np.einsum("ijkl,ijlm->ijkm", states[i], matrix_exp)

            ## Sample random numbers in batches
            rand_mats = np_rs.uniform(
                size=(self.num_classes, self.batch_size, self.replicates)
            )

            if not self.binary_search:
                # A fully vectorised O(n) approach
                _chosen = np.argmax(
                    np.cumsum(prob_dist, axis=STATE_AXIS)
                    > np.expand_dims(rand_mats, -1),
                    axis=-1,
                )
                chosen = np.indices((*_chosen.shape, 1))
                chosen[-1] = np.expand_dims(_chosen, -1)
            else:
                # A non-vectorised O(log(n)) approach
                chosen = []
                for env_id, (env_prob_cdf, rand_mat) in enumerate(
                    zip(np.cumsum(prob_dist, axis=STATE_AXIS), rand_mats)
                ):
                    for batch_id, (batch_prob_cdf, rand_vec) in enumerate(
                        zip(env_prob_cdf, rand_mat)
                    ):
                        chosen.extend(
                            (
                                env_id,
                                batch_id,
                                replicate_id,
                                replicate.searchsorted(rand_num),
                            )
                            for replicate_id, (replicate, rand_num) in enumerate(
                                zip(batch_prob_cdf, rand_vec)
                            )
                        )
                chosen = np.array(chosen).T

            # Update the state
            states[i + 1][tuple(chosen)] = True

        return states.reshape((*(states.shape[:2]), -1, states.shape[-1]))
