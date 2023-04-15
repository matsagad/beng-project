from typing import Tuple
from models.model import PromoterModel
from nptyping import NDArray
import numpy as np


class ModelWrapper:
    def __init__(
        self,
        model: PromoterModel,
        fitness: float = 0,
        mi: float = 0,
        runs_left: int = 3,
        novelty: float = 0,
        classes: NDArray = None,
    ):
        self.model = model
        self.fitness = fitness
        self.mi = mi
        self.runs_left = runs_left
        self.novelty = novelty
        self.classes = classes

    def __lt__(self, other: "ModelWrapper") -> bool:
        """
        Used for heapq sort during genetic algorithm.
        """
        return -self.fitness < -other.fitness

    def as_tuple(self) -> Tuple[float, float, int, PromoterModel]:
        return self.fitness, self.mi, self.runs_left, self.model

    def as_nn_array(self) -> NDArray:
        return self.classes.flatten()

    def dominates(self, other: "ModelWrapper") -> bool:
        not_worse = self.novelty >= other.novelty and self.local_fitness >= other.local_fitness
        is_better = self.novelty > other.novelty or self.local_fitness > other.local_fitness
        return not_worse and is_better
