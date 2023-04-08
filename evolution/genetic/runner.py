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

        self.sorted_models = []
        self.runner_stats = {"avg_time_duration": []}

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
        initial_population: List[Tuple[float, float, int, PromoterModel]] = [],
        verbose: bool = True,
        debug: bool = False,
    ) -> Tuple[List[PromoterModel], Dict[str, any]]:
        # Continue from a previous iteration if available
        models = [
            (self.scale_fitness(model, mi), mi, runs_left, model)
            for _, mi, runs_left, model in initial_population[:population]
        ]
        models_remaining = population - len(models)

        # Randomly initialise models with specified number of states
        models.extend(
            (
                0,
                0,
                runs_per_model,
                ModelGenerator.get_random_model(states, **model_generator_params),
            )
            for _ in range(models_remaining)
        )

        # Access model tuples through the following indices
        _FITNESS, _MI, _RUNS, _MODEL = 0, 1, 2, 3
        _INDEX = _MODEL

        # Get number of elites to keep and children to produce
        num_elites = int(population * elite_ratio)
        num_children = population - num_elites

        # Statistics for running the genetic algorithm
        labels = ("elite", "non_elite", "population")
        self.runner_stats = runner_stats = {"avg_time_duration": []}
        for label in labels:
            runner_stats[label] = {
                "avg_fitness": [],
                "std_fitness": [],
                "avg_mi": [],
                "std_mi": [],
                "avg_num_states": [],
            }
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

            # Get sorted list of models
            sorted_tuples = heapq.nsmallest(len(top_models), top_models)
            self.sorted_models = sorted_models = [
                (-tup[_FITNESS], *tup[_MI:_INDEX], models[tup[_INDEX]][_MODEL])
                for tup in sorted_tuples
            ]

            # Keep elites in next generation
            elite = sorted_models[:num_elites]

            if debug:
                print(
                    "\tTop Elites: "
                    + ", ".join(
                        [
                            f"({fitness:.3f}, {mi:.3f}, {model.num_states}, {model.hash()[2:8]})"
                            for fitness, mi, _, model in elite
                        ]
                    )
                )

            # Update runner statistics
            for label, tuples in zip(
                labels,
                (
                    sorted_models[:num_elites],
                    sorted_models[num_elites:],
                    sorted_models,
                ),
            ):
                runner_stats[label]["avg_fitness"].append(
                    np.average([tup[_FITNESS] for tup in tuples])
                )
                runner_stats[label]["std_fitness"].append(
                    np.std([tup[_FITNESS] for tup in tuples])
                )
                runner_stats[label]["avg_mi"].append(
                    np.average([tup[_MI] for tup in tuples])
                )
                runner_stats[label]["std_mi"].append(
                    np.std([tup[_MI] for tup in tuples])
                )
                runner_stats[label]["avg_num_states"].append(
                    np.average([tup[_MODEL].num_states for tup in tuples])
                )
            runner_stats["avg_time_duration"].append(time.time() - start)

            if debug:
                avg_fitness, avg_mi, std_mi, avg_num_states = [
                    runner_stats["population"][stat][-1]
                    for stat in ("avg_fitness", "avg_mi", "std_mi", "avg_num_states")
                ]
                time_duration = runner_stats["avg_time_duration"][-1]

                print(f"\tMean Population Fitness: {avg_fitness:.3f}")
                print(f"\tMean Population MI: {avg_mi:.3f}")
                print(f"\tStandard Deviation MI: {std_mi:.3f}")
                print(f"\tAvg Number of States: {avg_num_states:.3f}")
                print(f"\tIteration Duration: {time_duration:.3f}s")

            if iter == iterations - 1:
                break

            # Use selection operator to choose parents for next generation
            sorted_indices = [tup[_INDEX] for tup in sorted_tuples]

            parents = self.select(
                np.array([tup[_FITNESS] for tup in sorted_models])[
                    np.argsort(sorted_indices)
                ],
                n=num_children,
            )
            np.random.shuffle(parents)

            # Crossover parents and mutate their offspring
            elite_indices = set(sorted_indices[:num_elites])

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

        return self.sorted_models, self.runner_stats
