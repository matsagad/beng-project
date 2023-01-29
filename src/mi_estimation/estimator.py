from abc import ABC, abstractmethod
from models.model import PromoterModel
from nptyping import NDArray, Shape, Float


class MIEstimator(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def estimate(
        self,
        model: PromoterModel,
        trajectory: NDArray[Shape["Any, Any, Any, Any"], Float],
    ) -> float:
        """Estimates the MI of the trajectory with respect to the
           exogenous input data.

        Args:
            model       : promoter model responsible for the trajectory
            trajectory  : time series of the model states; dimensions are:
                          # of classes, # of time stamps, batch size, # of model states

        Returns:
            A float value corresponding to the MI.
        """
        pass
