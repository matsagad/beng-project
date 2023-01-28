from abc import ABC, abstractmethod
from models.model import PromoterModel
from nptyping import NDArray, Shape, Float


class StochasticSimulator(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def simulate(self, model: PromoterModel) -> NDArray[Shape["Any"], Float]:
        """Simulates the trajectory of the model.

        Args:
            model : a promoter model

        Returns:
            A time series of the model's state. Its z-dimension is
            equal to the number of samples from the exogenous input.
        """
        pass
