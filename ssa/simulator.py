from abc import ABC, abstractmethod
from models.model import PromoterModel
from nptyping import NDArray, Shape, Float


class StochasticSimulator(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def simulate(
        self, model: PromoterModel
    ) -> NDArray[Shape["Any, Any, Any, Any"], Float]:
        """Simulates the trajectory of the model.

        Args:
            model : a promoter model

        Returns:
            A time series of the model's state. Its dimensions are given
            by: # of classes, # of time stamps, batch size, # of model states.
        """
        pass

    @staticmethod
    def visualise_trajectory(
        trajectory: NDArray[Shape["Any, Any, Any, Any"], Float],
        average: bool = True,
        batch_num: int = 0,
    ) -> None:
        import matplotlib.pyplot as plt
        import numpy as np

        num_classes, num_times, batch_size, num_states = trajectory.shape
        x = np.arange(num_times)

        fig, axes = plt.subplots(1, num_classes, sharey=True)

        if num_classes == 1:
            axes = [axes]

        for (env_class, ax) in enumerate(axes):
            for state in range(num_states):
                ax.plot(
                    x,
                    np.average(trajectory[env_class, :, :, state], axis=1)
                    if average
                    else trajectory[env_class, :, batch_num, state],
                )

        plt.show()
