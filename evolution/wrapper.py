from typing import Tuple
from models.model import PromoterModel
from nptyping import NDArray


class ModelWrapper:
    def __init__(
        self,
        model: PromoterModel,
        fitness: float = 0,
        mi: float = 0,
        runs_left: int = 3,
        classes: NDArray = None,
    ):
        self.model = model
        self.fitness = fitness
        self.mi = mi
        self.runs_left = runs_left
        self.classes = classes

    def __lt__(self, other: "ModelWrapper") -> bool:
        return -self.fitness < -other.fitness

    def as_tuple(self) -> Tuple[float, float, int, PromoterModel]:
        return self.fitness, self.mi, self.runs_left, self.model