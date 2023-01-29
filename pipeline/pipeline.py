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
    ):
        self.simulator = simulator
        self.estimator = estimator

    @abstractmethod
    def evaluate(self, model: PromoterModel, verbose: bool = False) -> float:
        """Evaluates the model by some metric.

        Args:
            model   : a promoter model
            verbose : flag for printing progress onto the console

        Returns:
            A raw fitness value for the model.
        """
        pass

    def estimateMI(self, model: PromoterModel, verbose: bool = False) -> float:
        """Estimates the MI supplied by the model.

        Args:
            model   : a promoter model
            verbose : flag for printing progress onto the console

        Returns:
            A float value corresponding to the MI.
        """
        if verbose:
            print("Simulating model...")
            start = time.time()

        trajectory = self.simulator.simulate(model)

        if verbose:
            print(f"({'%.3f' % (time.time() - start)}s) Trajectory obtained!")
            print("Estimating MI...")
            start = time.time()

        miscore = self.estimator.estimate(model, trajectory)

        if verbose:
            print(f"({'%.3f' % (time.time() - start)}s) MI found: {miscore}")

        return miscore
