from abc import ABC, abstractmethod
from nptyping import NDArray, Shape, Float


class MIEstimator(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def estimate(self, trajectory: NDArray[Shape["Any, Any, Any"], Float]) -> float:
        """Estimates the MI of the trajectory with respect to the
           exogenous input data.

        Args:
            trajectory  : a time series of the model states

        Returns:
            A unit vector of MI values. Its dimension is equal to the number
            of samples from the exogenous input.
        """
        pass
