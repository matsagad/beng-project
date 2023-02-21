from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import reduce
from models.generator import ModelGenerator
from models.model import PromoterModel
from nptyping import NDArray
from pipeline.one_step_decoding import OneStepDecodingPipeline
from pipeline.pipeline import Pipeline
from typing import Callable, List, Tuple
import heapq
import numpy as np


class GeneticRunner:
    def _compose(self, arg, fs):
        return reduce(lambda _arg, _f: _f(_arg), fs, arg)

    def __init__(self, data: NDArray, mutations: List[Callable], crossover: Callable):
        self.pip = OneStepDecodingPipeline(
            data, realised=True, replicates=10, classifier_name="naive_bayes"
        )
        self.mutate = lambda x: self._compose(x, mutations)
        self.crossover = crossover

    def _evaluate(pip: Pipeline, model: PromoterModel, index: int) -> Tuple[int, float]:
        return index, pip.evaluate(model, verbose=False)

    def run(
        self,
        states: int,
        population: int = 8,
        elite_ratio: float = 0.2,
        iterations: int = 1,
        n_processors: int = 8,
        verbose: bool = True,
    ) -> None:
        # Randomly initialise random models with specified number of states
        models = [ModelGenerator.get_random_model(states) for _ in range(population)]

        num_elites = int(population * elite_ratio)
        num_children = population - num_elites
        best_mi = 0
        best_model = None

        for iter in range(iterations):
            top_models = []

            # Find fitness of each model
            with ProcessPoolExecutor(
                max_workers=min(population, n_processors)
            ) as executor:
                futures = []
                for i, model in enumerate(models):
                    futures.append(
                        executor.submit(GeneticRunner._evaluate, self.pip, model, i)
                    )

                for future in as_completed(futures):
                    i, mi = future.result()
                    heapq.heappush(top_models, (-mi, i))

            if -top_models[0][0] > best_mi:
                best_mi = -top_models[0][0]
                best_model = top_models[0][1]

            if verbose:
                print(f"({iter + 1}/{iterations}):\t Best found: {best_mi}")

            # Keep elites in next generation
            elite = [model for _, model in top_models[:num_elites]]

            # Crossover random parents and mutate their offspring
            parents = np.random.permutation(population)
            children = []
            for parent1, parent2 in zip(
                parents[: 2 * num_children : 2],
                parents[1 : 1 + 2 * num_children : 2],
            ):
                children.extend(
                    self.mutate(child)
                    for child in self.crossover(models[parent1], models[parent2])
                )

            models = children
            models.extend(elite)

        return models[best_model]
