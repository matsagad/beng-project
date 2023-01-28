from systems.biochemical import BioChemicalSystem
from models.model import PromoterModel
import numpy as np


class Extrande:
    @staticmethod
    def simulate(
        model: PromoterModel,
        max_time: float,
        initial_state: np.ndarray,
        exogenous_times: np.ndarray,
        exogenous_states: np.ndarray,
        scaled: bool = True,
    ) -> np.ndarray:
        """Simulates the biochemical system involved in the given model
        whilst taking into account exogenous states and their time of observation.

        Args:
          model           : a promoter model
          max_time        : maximum time for the simulation to run
          initial_state   : initial state of the system, i.e. [[A], [I], [M], [P]]
          exogenous_times : time stamps of observations for an exogenous experiment
          exogenous_states: states matching each time stamp for an exogenous experiment

        Returns:
          A time series data of the states matching time stamps of the given exogenous experiment.
        """
        # Initial conditions
        time, state = 0.0, initial_state
        times, states = [], []

        # Constants
        num_time_stamps = len(exogenous_times)
        sim_time_factor = num_time_stamps / max_time
        reactions_per_time_stamp = 2

        composition_matrix = model.composition_matrix()
        find_total_propensities_vec = np.vectorize(
            lambda ex_index, state: sum(
                model.propensity(state, exogenous_states[:, ex_index])
            ),
            excluded=["state"],
        )

        while time < max_time:
            # Record data
            times.append(time)
            states.append(list(state))

            # Find current time stamp relative to exogenous time series
            sim_time_stamp = int(time * sim_time_factor)

            # Find propensity bounds
            L = min(3 * reactions_per_time_stamp / sim_time_factor, max_time - time)
            next_sim_time_stamp = int((time + L) * sim_time_factor)
            B = np.max(
                find_total_propensities_vec(
                    ex_index=np.arange(sim_time_stamp, next_sim_time_stamp), state=state
                )
            )

            # Generate putative reaction time
            tau = np.random.exponential(1 / B)

            # "Reject": keep state unchanged
            if tau > L:
                time += L
                continue

            time += tau

            # Generate uniformly distributed random number
            u = np.random.uniform(0, 1)
            # Find the exogenous state relative to simulation time
            exogenous_state = exogenous_states[:, sim_time_stamp]
            propensity = model.propensity(state, exogenous_state)
            total_propensity = sum(propensity)

            # "Thin": keep state unchanged
            if total_propensity < B * u:
                continue

            # Smallest index for which sum is >= B * u.
            j = min(np.cumsum(propensity).searchsorted(B * u), len(propensity))
            state += composition_matrix[j, :]

        # Record last time and state right after the simulation ends
        times.append(time)
        states.append(list(state))

        time_arr, state_arr = np.array(times), np.array(states)
        if not scaled:
            return time_arr, state_arr

        # Scale time series data to match time stamps of realised experiment
        scaled_states = state_arr[time_arr.searchsorted(exogenous_times)]
        return exogenous_times, scaled_states
