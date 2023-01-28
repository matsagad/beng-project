from abc import ABC, abstractmethod
from mi_estimation.estimator import MIEstimator
from models.model import PromoterModel
from ssa.simulator import StochasticSimulator
from numpy import ndarray

class Pipeline(ABC):
    def __init__(self, simulator: StochasticSimulator, estimator: MIEstimator):
        self.simulator = simulator
        self.estimator = estimator

    @abstractmethod
    def evaluate(self, model: PromoterModel) -> float:
      """Evaluates the model by some metric.

      Args:
          model : a promoter model

      Returns:
          A raw fitness value for the model.
      """
      pass
    
    def estimateMI(self, model: PromoterModel) -> ndarray:
      """Estimates the MI supplied by the model.

      Args:
          model : a promoter model
          
      Returns:
          A unit vector of MI values. Its dimension is equal to the number
          of samples from the exogenous input used by the estimator.
      """
      trajectory = self.simulator.simulate(model)
      return self.estimator.estimate(trajectory)