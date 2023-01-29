from mi_estimation.estimator import MIEstimator
from nptyping import NDArray, Shape, Float
from models.model import PromoterModel
from utils.mi_decoding import estimateMIS
import numpy as np


class DecodingEstimator(MIEstimator):
    def __init__(self, origin: int, interval: int):
        self.origin = origin
        self.interval = interval

    def estimate(
        self,
        model: PromoterModel,
        trajectory: NDArray[Shape["Any, Any, Any, Any"], Float],
    ) -> float:
        active_states = model.active_states
        rich_state = None
        states = []

        for env_class in trajectory:
            # Sum probabilities of the active states
            rich_trajectory = np.sum(
                env_class[self.origin - self.interval : self.origin, :, active_states],
                axis=2,
            ).T
            if rich_state is None:
                rich_state = rich_trajectory
            else:
                rich_state += rich_trajectory

            stress_trajectory = np.sum(
                env_class[self.origin : self.origin + self.interval, :, active_states],
                axis=2,
            ).T
            states.append(stress_trajectory)

        avg_rich_state = rich_state / len(trajectory)

        # Evaluate
        miscore = estimateMIS([[avg_rich_state] + states])[0]

        return miscore
