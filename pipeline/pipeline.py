from abc import ABC, abstractmethod
from mi_estimation.estimator import MIEstimator
from models.model import PromoterModel
from ssa.simulator import StochasticSimulator
import time


class Pipeline(ABC):
    def __init__(
        self,
        simulator: StochasticSimulator,
        estimator: MIEstimator,
        verbose: bool = False,
    ):
        self.simulator = simulator
        self.estimator = estimator
        self.verbose = verbose

    @abstractmethod
    def evaluate(self, model: PromoterModel) -> float:
        """Evaluates the model by some metric.

        Args:
            model : a promoter model

        Returns:
            A raw fitness value for the model.
        """
        pass

    def estimateMI(self, model: PromoterModel) -> float:
        """Estimates the MI supplied by the model.

        Args:
            model : a promoter model

        Returns:
            A float value corresponding to the MI.
        """
        if self.verbose:
            print("Simulating model...")
            start = time.time()

        trajectory = self.simulator.simulate(model)

        if self.verbose:
            print(f"Trajectory obtained! ({time.time() - start}s)")
            print("Estimating MI...")
            start = time.time()

        mi = self.estimator.estimate(model, trajectory)

        if self.verbose:
            print(f"MI found: {mi}. ({time.time() - start}s)")

        return mi
