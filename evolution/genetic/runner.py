from concurrent.futures import ProcessPoolExecutor, as_completed
from evolution.wrapper import ModelWrapper
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

    def evaluate_wrapper(model: PromoterModel, index: int) -> Tuple[int, float, float, float]:
        return (index, *GeneticRunner.mp_instance._evaluate(model))

    def _evaluate(self, model: PromoterModel) -> Tuple[float, float, float]:
        trajectory = self.pip.simulator.simulate(model)
        mi, std_mi = self.pip.estimator.estimate(model, trajectory, return_std=True)
        return self.scale_fitness(model, mi), mi, std_mi

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
    ) -> Tuple[List[Tuple[float, float, int, PromoterModel]], Dict[str, any]]:
        # Continue from a previous iteration if available
        models = [
            ModelWrapper(
                model=model,
                fitness=self.scale_fitness(model, mi),
                mi=mi,
                runs_left=runs_left,
            )
            for _, mi, runs_left, model in initial_population[:population]
        ]
        models_remaining = population - len(models)

        # Randomly initialise models with specified number of states
        models.extend(
            ModelWrapper(
                model=ModelGenerator.get_random_model(states, **model_generator_params),
                runs_left=runs_per_model,
            )
            for _ in range(models_remaining)
        )

        # Access model tuples through the following indices
        WRAPPER, INDEX = 0, 1

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
                "avg_std_mi": [],
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
                for i, wrapper in enumerate(models):
                    if wrapper.runs_left > 0:
                        futures.append(
                            executor.submit(
                                GeneticRunner.evaluate_wrapper, wrapper.model, i
                            )
                        )
                    else:
                        heapq.heappush(top_models, (wrapper, i))

                for future in as_completed(futures):
                    i, curr_fitness, curr_mi, curr_std_mi = future.result()

                    wrapper = models[i]
                    avg_fitness, avg_mi = wrapper.fitness, wrapper.mi
                    runs_left = wrapper.runs_left

                    # Find running average for fitness and MI of each model
                    fitness = (
                        avg_fitness * (runs_per_model - runs_left) + curr_fitness
                    ) / (runs_per_model - runs_left + 1)
                    mi = (avg_mi * (runs_per_model - runs_left) + curr_mi) / (
                        runs_per_model - runs_left + 1
                    )
                    wrapper.fitness, wrapper.mi = fitness, mi
                    wrapper.std_mi = max(wrapper.std_mi, curr_std_mi)

                    heapq.heappush(top_models, (wrapper, i))

            curr_best_model_wrapper = top_models[0][WRAPPER]
            curr_stats = {
                "fitness": curr_best_model_wrapper.fitness,
                "mi": curr_best_model_wrapper.mi,
                "std_mi": curr_best_model_wrapper.std_mi,
                "num_states": curr_best_model_wrapper.model.num_states,
                "hash": curr_best_model_wrapper.model.hash(short=True),
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
            sorted_tuples = [heapq.heappop(top_models) for _ in range(len(top_models))]
            self.sorted_models = sorted_models = [
                wrapper for wrapper, _ in sorted_tuples
            ]

            # Keep elites in next generation
            elite = sorted_models[:num_elites]

            if debug:
                print(
                    "\tTop Elites: "
                    + ", ".join(
                        [
                            f"({wrapper.fitness:.3f}, {wrapper.mi:.3f}, {wrapper.model.num_states}, {wrapper.model.hash(short=True)})"
                            for wrapper in elite
                        ]
                    )
                )

            # Update runner statistics
            for label, wrappers in zip(
                labels,
                (
                    sorted_models[:num_elites],
                    sorted_models[num_elites:],
                    sorted_models,
                ),
            ):
                fitnesses = [wrapper.fitness for wrapper in wrappers]
                mis = [wrapper.mi for wrapper in wrappers]
                std_mis = [
                    wrapper.std_mi for wrapper in wrappers if wrapper.std_mi >= 0
                ]

                runner_stats[label]["avg_fitness"].append(np.average(fitnesses))
                runner_stats[label]["std_fitness"].append(np.std(fitness))
                runner_stats[label]["avg_mi"].append(np.average(mis))
                runner_stats[label]["std_mi"].append(np.std(mis))
                runner_stats[label]["avg_std_mi"].append(np.average(std_mis))
                runner_stats[label]["avg_num_states"].append(
                    np.average([wrapper.model.num_states for wrapper in wrappers])
                )
            runner_stats["avg_time_duration"].append(time.time() - start)

            if debug:
                avg_fitness, avg_mi, std_mi, avg_std_mi, avg_num_states = [
                    runner_stats["population"][stat][-1]
                    for stat in (
                        "avg_fitness",
                        "avg_mi",
                        "std_mi",
                        "avg_std_mi",
                        "avg_num_states",
                    )
                ]
                time_duration = runner_stats["avg_time_duration"][-1]

                print(f"\tMean Population Fitness: {avg_fitness:.3f}")
                print(f"\tMean Population MI: {avg_mi:.3f}")
                print(f"\tPopulation Standard Deviation of MI: {std_mi:.3f}")
                print(f"\tMean MI Standard Deviation: {avg_std_mi:.3f}")
                print(f"\tAvg Number of States: {avg_num_states:.3f}")
                print(f"\tIteration Duration: {time_duration:.3f}s")

            if iter == iterations - 1:
                break

            # Use selection operator to choose parents for next generation
            sorted_indices = [tup[INDEX] for tup in sorted_tuples]

            parents = self.select(
                np.array([wrapper.fitness for wrapper in sorted_models])[
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
                    ModelWrapper(self.mutate(child), runs_left=runs_per_model)
                    for child in self.crossover(
                        models[parent1].model,
                        models[parent2].model,
                        parent1 in elite_indices,
                        parent2 in elite_indices,
                    )
                )

            models = elite + children

        return [wrapper.as_tuple() for wrapper in self.sorted_models], self.runner_stats
