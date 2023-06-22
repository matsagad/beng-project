from models.model import PromoterModel
from ssa.simulator import StochasticSimulator
from nptyping import NDArray, Shape, Float
import numpy as np


class GillespieSimulator(StochasticSimulator):
    """
    Simulates the promoter trajectory through the Gillespie algorithm (Next reaction method).
    """

    def __init__(
        self,
        exogenous_data: NDArray[Shape["Any, Any, Any, Any"], Float],
        tau: float,
        replicates: int = 1,
    ):
        self.exogenous_data = exogenous_data
        (
            self.num_classes,
            self.num_tfs,
            self.batch_size,
            self.num_times,
        ) = self.exogenous_data.shape
        self.tau = tau
        self.replicates = replicates

    def simulate(self, model: PromoterModel) -> NDArray[Shape["Any, Any"], Float]:
        max_t = self.num_times * self.tau
        states = np.zeros(
            (self.num_classes, self.batch_size, self.replicates, self.num_times),
            dtype=int,
        )
        generators = model.get_generator(self.exogenous_data)

        for env_num, env_states in enumerate(states):
            for batch_num, batched_states in enumerate(env_states):
                generator = generators[:, env_num, batch_num]
                for state in batched_states:
                    t = 0
                    curr_time_interval = 0
                    state[0] = np.random.choice(model.num_states)
                    while t < max_t:
                        curr_state = state[curr_time_interval]

                        rates = generator[curr_time_interval, curr_state]
                        holding_rate = -rates[curr_state]

                        if holding_rate == 0:
                            if curr_time_interval + 1 >= self.num_times:
                                break
                            state[curr_time_interval + 1] = curr_state
                            t = (curr_time_interval + 1) * self.tau
                            curr_time_interval += 1
                            continue

                        tau = np.random.exponential(1 / holding_rate)

                        if t + tau > (curr_time_interval + 1) * self.tau:
                            if curr_time_interval + 1 >= self.num_times:
                                break
                            state[curr_time_interval + 1] = curr_state

                            t = (curr_time_interval + 1) * self.tau
                            curr_time_interval += 1
                            continue

                        u = np.random.uniform(0, 1)
                        _rate_sum = (
                            np.cumsum(np.delete(rates, curr_state)) / holding_rate
                        )
                        next_state = np.searchsorted(_rate_sum, u, side="right")
                        if next_state >= curr_state:
                            next_state += 1

                        state[curr_time_interval] = next_state
                        t += tau

        return states
