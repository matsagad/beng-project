from mi_estimation.estimator import MIEstimator
from nptyping import NDArray, Shape, Float

class Decoding(MIEstimator):
  def __init__(self):
    pass

  def estimate(self, trajectory: NDArray[Shape["Any, Any, Any"], Float]) -> float:
    return super().estimate(trajectory)