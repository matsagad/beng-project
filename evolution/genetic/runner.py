from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import reduce
from models.generator import ModelGenerator
from models.model import PromoterModel
from nptyping import NDArray
from pipeline.one_step_decoding import OneStepDecodingPipeline
from typing import Callable, Dict, List, Tuple
import heapq
import numpy as np
import time


class GeneticRunner:
    mp_instance = None

    def _compose(self, arg, fs):
        return reduce(lambda _arg, _f: _f(_arg), fs, arg)

    def __init__(
        self,
        data: NDArray,
        mutations: List[Callable],
        crossover: Callable,
        select: Callable,
        scale_fitness: Callable = lambda _, mi: mi,
    ):
        self.pip = OneStepDecodingPipeline(
            data, realised=True, replicates=10, classifier_name="naive_bayes"
        )
        self.mutate = lambda x: self._compose(x, mutations)
        self.crossover = crossover
        self.select = select
        self.scale_fitness = scale_fitness

    def evaluate_wrapper(model: PromoterModel, index: int) -> Tuple[int, float, float]:
        return (index, *GeneticRunner.mp_instance._evaluate(model))

    def _evaluate(self, model: PromoterModel) -> Tuple[float, float]:
        mi = self.pip.evaluate(model, verbose=False)
        return self.scale_fitness(model, mi), mi

    def run(
        self,
        states: int,
        population: int = 10,
        elite_ratio: float = 0.2,
        iterations: int = 10,
        n_processors: int = 10,
        runs_per_model: int = 3,
        model_generator_params: Dict[str, any] = dict(),
        verbose: bool = True,
        debug: bool = False,
    ) -> None:
        # Randomly initialise random models with specified number of states
        models = [
            (
                0,
                0,
                runs_per_model,
                ModelGenerator.get_random_model(states, **model_generator_params),
            )
            for _ in range(population)
        ]

        # Model tuple access
        _FITNESS, _MI, _RUNS, _MODEL = 0, 1, 2, 3
        _INDEX = _MODEL

        num_elites = int(population * elite_ratio)
        num_children = population - num_elites
        best_stats = {"fitness": 0, "mi": 0, "num_states": 0, "hash": ""}

        num_digits = len(str(iterations))

        # Use copy-on-write to avoid pickling entire exogenous data per process.
        # (Can do this as long as child processes don't modify the instance)
        GeneticRunner.mp_instance = self

        for iter in range(iterations):
            top_models = []

            start = time.time()

            # Find fitness of each model
            with ProcessPoolExecutor(
                max_workers=min(population, n_processors)
            ) as executor:
                futures = []
                for i, (fitness, mi, runs_left, model) in enumerate(models):
                    if runs_left > 0:
                        futures.append(
                            executor.submit(GeneticRunner.evaluate_wrapper, model, i)
                        )
                    else:
                        heapq.heappush(top_models, (-fitness, mi, 0, i))

                for future in as_completed(futures):
                    i, curr_fitness, curr_mi = future.result()
                    avg_fitness, avg_mi, runs_left, _ = models[i]

                    # Find running average for fitness and MI of each model
                    fitness = (
                        avg_fitness * (runs_per_model - runs_left) + curr_fitness
                    ) / (runs_per_model - runs_left + 1)
                    mi = (avg_mi * (runs_per_model - runs_left) + curr_mi) / (
                        runs_per_model - runs_left + 1
                    )
                    heapq.heappush(top_models, (-fitness, mi, runs_left - 1, i))

            curr_best_model = models[top_models[0][_INDEX]][_MODEL]
            curr_stats = {
                "fitness": -top_models[0][_FITNESS],
                "mi": top_models[0][_MI],
                "num_states": curr_best_model.num_states,
                "hash": curr_best_model.hash()[2:8],
            }

            if curr_stats["fitness"] > best_stats["fitness"]:
                best_stats = curr_stats

            if verbose:
                print(
                    f"({str(iter + 1).zfill(num_digits)}/{iterations}):\t\033[1m "
                    + "\t ".join(
                        f"{label}: {stats['fitness']:.3f} ({stats['mi']:.3f}, {stats['num_states']}, {stats['hash']})"
                        for label, stats in zip(
                            ("Best", "Current"), (best_stats, curr_stats)
                        )
                    )
                    + "\033[0m"
                )

            # Keep elites in next generation
            sorted_tuples = heapq.nsmallest(len(top_models), top_models)
            sorted_indices = [tup[_INDEX] for tup in sorted_tuples]

            elite_tuples = sorted_tuples[:num_elites]
            elite_indices = set(sorted_indices[:num_elites])
            elite = [
                (-tup[_FITNESS], *tup[_MI:_INDEX], models[tup[_INDEX]][_MODEL])
                for tup in elite_tuples
            ]

            if debug:
                print(
                    "\tTop Elites: "
                    + ", ".join(
                        [
                            f"({-fitness:.3f}, {mi:.3f}, {models[i][_MODEL].num_states}, {models[i][_MODEL].hash()[2:8]})"
                            for fitness, mi, _, i in elite_tuples
                        ]
                    )
                )
            if iter == iterations - 1:
                return [models[i][_MODEL] for i in sorted_indices]

            _fitness_scores = [-tup[_FITNESS] for tup in sorted_tuples]
            _mi_scores = [tup[_MI] for tup in sorted_tuples]

            if debug:
                print(f"\tMean Population Fitness: {np.average(_fitness_scores):.3f}")
                print(f"\tMean Population MI: {np.average(_mi_scores):.3f}")
                print(
                    f"\tAvg Number of States: {np.average([model[_MODEL].num_states for model in models]):.3f}"
                )
                print(f"\tIteration Duration: {(time.time() - start):.3f}s")

            # Use selection operator to choose parents for next generation
            parents = self.select(
                np.array(_fitness_scores)[np.argsort(sorted_indices)], n=num_children
            )
            np.random.shuffle(parents)

            # Crossover parents and mutate their offspring
            children = []
            for parent1, parent2 in zip(
                parents[::2],
                parents[1::2],
            ):
                children.extend(
                    (0, 0, runs_per_model, self.mutate(child))
                    for child in self.crossover(
                        models[parent1][_MODEL],
                        models[parent2][_MODEL],
                        parent1 in elite_indices,
                        parent2 in elite_indices,
                    )
                )

            models = elite + children
