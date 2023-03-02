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

    def __init__(
        self,
        data: NDArray,
        mutations: List[Callable],
        crossover: Callable,
        select: Callable,
    ):
        self.pip = OneStepDecodingPipeline(
            data, realised=True, replicates=10, classifier_name="naive_bayes"
        )
        self.mutate = lambda x: self._compose(x, mutations)
        self.crossover = crossover
        self.select = select

    def _evaluate(pip: Pipeline, model: PromoterModel, index: int) -> Tuple[int, float]:
        return index, pip.evaluate(model, verbose=False)

    def run(
        self,
        states: int,
        population: int = 10,
        elite_ratio: float = 0.2,
        iterations: int = 10,
        n_processors: int = 10,
        verbose: bool = True,
        debug: bool = False,
    ) -> None:
        # Randomly initialise random models with specified number of states
        models = [ModelGenerator.get_random_model(states) for _ in range(population)]

        num_elites = int(population * elite_ratio)
        num_children = population - num_elites
        best_mi = 0
        best_model_hash = ""

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

            curr_best_mi = -top_models[0][0]
            curr_best_model_hash = models[top_models[0][1]].hash()[2:8]

            if curr_best_mi > best_mi:
                best_mi = curr_best_mi
                best_model_hash = curr_best_model_hash

            if verbose:
                print(
                    f"({(iter + 1):02}/{iterations}):\t Best: {best_mi:.3f} ({best_model_hash})"
                    + f"\t Current: {curr_best_mi:.3f} ({curr_best_model_hash})"
                )

            # Keep elites in next generation
            _sorted_pairs = heapq.nsmallest(len(top_models), top_models)
            sorted_indices = [i for _, i in _sorted_pairs]

            _elite_pairs = _sorted_pairs[:num_elites]
            elite_indices = sorted_indices[:num_elites]
            elite = [models[i] for i in elite_indices]

            if debug:
                print(
                    "Elites: "
                    + ", ".join(
                        [
                            f"({-mi:.3f}, {models[i].hash()[2:8]})"
                            for mi, i in _elite_pairs
                        ]
                    )
                )
            if iter == iterations - 1:
                return [
                    models[i] for _, i in heapq.nsmallest(len(top_models), top_models)
                ]

            # Use selection operator to choose parents for next generation
            _mi_scores = [-mi for mi, _ in _sorted_pairs]
            parents = self.select(
                np.array(_mi_scores)[np.argsort(sorted_indices)], n=num_children
            )
            np.random.shuffle(parents)

            # Crossover parents and mutate their offspring
            children = []
            for parent1, parent2 in zip(
                parents[::2],
                parents[1::2],
            ):
                children.extend(
                    self.mutate(child)
                    for child in self.crossover(
                        models[parent1],
                        models[parent2],
                        parent1 in elite_indices,
                        parent2 in elite_indices,
                    )
                )

            models = elite + children
            if debug:
                print(f"Mean Population MI: {np.average(_mi_scores):.3f}")
