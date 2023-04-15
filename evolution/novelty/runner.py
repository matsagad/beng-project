from concurrent.futures import ProcessPoolExecutor, as_completed
from evolution.novelty.metrics import TrajectoryMetric
from evolution.wrapper import ModelWrapper
from functools import reduce
from models.generator import ModelGenerator
from models.model import PromoterModel
from nptyping import NDArray
from pipeline.one_step_decoding import OneStepDecodingPipeline
from sklearn.neighbors import NearestNeighbors
from typing import Callable, Dict, List, Tuple
import heapq
import itertools
import numpy as np
import time


class NoveltySearchRunner:
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
        self.prob_pip = OneStepDecodingPipeline(data, realised=False)

        self.mutate = lambda x: self._compose(x, mutations)
        self.crossover = crossover
        self.select = select
        self.scale_fitness = scale_fitness

        self.sorted_models = []
        self.runner_stats = {"avg_time_duration": []}

    def evaluate_wrapper(
        model: PromoterModel, index: int, find_classes: bool = False
    ) -> Tuple[int, float, float]:
        return (index, *NoveltySearchRunner.mp_instance._evaluate(model, find_classes))

    def _evaluate(
        self, model: PromoterModel, find_classes: bool
    ) -> Tuple[float, float, NDArray | None]:
        mi = self.pip.evaluate(model, verbose=False)
        classes = None
        if find_classes:
            dist_traj = self.prob_pip.simulator.simulate(model)
            classes = self.prob_pip.estimator._split_classes(model, dist_traj)
        return self.scale_fitness(model, mi), mi, classes

    def run(
        self,
        states: int,
        population: int = 100,
        elite_ratio: float = 0.2,
        iterations: int = 100,
        linear_metric: bool = True,
        novelty_threshold: float = 0.05,
        archival_rate_threshold: int = 4,
        archival_stagnation_threshold: int = 5,
        n_neighbors: int = 15,
        n_processors: int = 10,
        runs_per_model: int = 3,
        model_generator_params: Dict[str, any] = dict(),
        initial_population: List[Tuple[float, float, int, PromoterModel]] = [],
        verbose: bool = True,
        debug: bool = False,
    ) -> Tuple[
        List[Tuple[float, float, int, PromoterModel]],
        List[Tuple[float, float, int, PromoterModel]],
        Dict[str, any],
    ]:
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

        # Get number of elites to keep and children to produce
        num_elites = int(population * elite_ratio)
        num_children = population - num_elites

        # Statistics for running the genetic algorithm
        labels = ("elite", "non_elite", "population")
        self.runner_stats = runner_stats = {"avg_time_duration": [], "archive_size": []}
        for label in labels:
            runner_stats[label] = {
                "avg_novelty": [],
                "std_novelty": [],
                "avg_fitness": [],
                "std_fitness": [],
                "avg_mi": [],
                "std_mi": [],
                "avg_num_states": [],
            }
        best_stats = {"novelty": 0, "fitness": 0, "mi": 0, "num_states": 0, "hash": ""}

        num_digits = len(str(iterations))

        if linear_metric:
            nn = NearestNeighbors(
                n_neighbors=n_neighbors + 1,  # excluding self
                metric=TrajectoryMetric.js_divergence_for_trajectories,
                n_jobs=-1,
            )

        novelty_archive = []
        num_iterations_without_archive = 0

        # Use copy-on-write to avoid pickling entire exogenous data per process.
        # (Can do this as long as child processes don't modify the instance)
        NoveltySearchRunner.mp_instance = self
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
                        find_classes = linear_metric and wrapper.classes is None
                        futures.append(
                            executor.submit(
                                NoveltySearchRunner.evaluate_wrapper,
                                wrapper.model,
                                i,
                                find_classes,
                            )
                        )
                    else:
                        heapq.heappush(top_models, wrapper)

                for future in as_completed(futures):
                    i, curr_fitness, curr_mi, curr_classes = future.result()
                    wrapper = models[i]
                    avg_fitness, avg_mi = wrapper.fitness, wrapper.mi
                    runs_left = wrapper.runs_left

                    if wrapper.classes is None:
                        wrapper.classes = curr_classes

                    # Find running average for fitness and MI of each model
                    fitness = (
                        avg_fitness * (runs_per_model - runs_left) + curr_fitness
                    ) / (runs_per_model - runs_left + 1)
                    mi = (avg_mi * (runs_per_model - runs_left) + curr_mi) / (
                        runs_per_model - runs_left + 1
                    )
                    wrapper.fitness, wrapper.mi = fitness, mi
                    heapq.heappush(top_models, wrapper)

            # Find novelties and local competition between models
            if linear_metric:
                nn_arrays = np.array(
                    [wrapper.as_nn_array() for wrapper in models + novelty_archive]
                )

                nn.fit(nn_arrays)
                distances, neighbors = (
                    arr[:, 1:] for arr in nn.kneighbors(nn_arrays, return_distance=True)
                )
            novelties = np.average(distances, axis=1)
            fitnesses = np.array(
                [wrapper.fitness for wrapper in models + novelty_archive]
            )

            # Archive models and adjust thresholds if necessary
            num_models_archived = 0
            for wrapper, k_neighbors, novelty in zip(models, neighbors, novelties):
                # Find local competition score
                wrapper.local_fitness = np.sum(wrapper.fitness > fitnesses[k_neighbors])
                wrapper.novelty = novelty

                if novelty > novelty_threshold:
                    novelty_archive.append(wrapper)
                    num_models_archived += 1

            if num_models_archived > archival_rate_threshold:
                novelty_threshold = min(1, 1.05 * novelty_threshold)

            if num_models_archived == 0:
                num_iterations_without_archive += 1
            else:
                num_iterations_without_archive = 0

            if num_iterations_without_archive > archival_stagnation_threshold:
                novelty_threshold *= 0.95

            # Keep Pareto optimal front and those succeeding as elites
            fronts = self.get_pareto_fronts(models)
            elite = []

            i = 0
            while len(elite) + len(fronts[i]) <= num_elites:
                elite.extend(fronts[i])
                i += 1

            remaining = num_elites - len(elite)
            curr_front = sorted(fronts[i], key=lambda x: -x.novelty)

            elite.extend(curr_front[:remaining])
            non_elite = curr_front[remaining:]
            non_elite.extend(itertools.chain.from_iterable(fronts[i + 1 :]))

            # Log current iteration's progress
            curr_best_model_wrapper = top_models[0]
            curr_stats = {
                "novelty": curr_best_model_wrapper.novelty,
                "fitness": curr_best_model_wrapper.fitness,
                "mi": curr_best_model_wrapper.mi,
                "num_states": curr_best_model_wrapper.model.num_states,
                "hash": curr_best_model_wrapper.model.hash(short=True),
            }

            if curr_stats["fitness"] > best_stats["fitness"]:
                best_stats = curr_stats

            if verbose:
                print(
                    f"({str(iter + 1).zfill(num_digits)}/{iterations}):\t\033[1m "
                    + "\t ".join(
                        f"{label}: {stats['novelty']:.3f} {stats['fitness']:.3f} ({stats['mi']:.3f}, {stats['num_states']}, {stats['hash']})"
                        for label, stats in zip(
                            ("Best", "Current"), (best_stats, curr_stats)
                        )
                    )
                    + "\033[0m"
                )

            if debug:
                print(
                    "\t(Pareto) Elites: "
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
                    elite,
                    non_elite,
                    models,
                ),
            ):
                novelties = [wrapper.novelty for wrapper in wrappers]
                fitnesses = [wrapper.fitness for wrapper in wrappers]
                mis = [wrapper.mi for wrapper in wrappers]

                runner_stats[label]["avg_novelty"].append(np.average(novelties))
                runner_stats[label]["std_novelty"].append(np.average(novelties))

                runner_stats[label]["avg_fitness"].append(np.average(fitnesses))
                runner_stats[label]["std_fitness"].append(np.std(fitness))

                runner_stats[label]["avg_mi"].append(np.average(mis))
                runner_stats[label]["std_mi"].append(np.std(mis))

                runner_stats[label]["avg_num_states"].append(
                    np.average([wrapper.model.num_states for wrapper in wrappers])
                )
            runner_stats["avg_time_duration"].append(time.time() - start)

            if debug:
                avg_novelty, avg_fitness, avg_mi, std_mi, avg_num_states = [
                    runner_stats["population"][stat][-1]
                    for stat in (
                        "avg_novelty",
                        "avg_fitness",
                        "avg_mi",
                        "std_mi",
                        "avg_num_states",
                    )
                ]
                time_duration = runner_stats["avg_time_duration"][-1]
                print(f"\tNovelty Archive Size: {len(novelty_archive)}")
                print(f"\tArchival Threshold: {novelty_threshold:.3f}")
                print(f"\tMean Population Novelty: {avg_novelty:.3f}")
                print(f"\tMean Population Fitness: {avg_fitness:.3f}")
                print(f"\tMean Population MI: {avg_mi:.3f}")
                print(f"\tStandard Deviation MI: {std_mi:.3f}")
                print(f"\tAvg Number of States: {avg_num_states:.3f}")
                print(f"\tIteration Duration: {time_duration:.3f}s")

            if iter == iterations - 1:
                break

            # Use selection operator to choose parents for next generation
            parents = self.select(
                np.array([wrapper.rank for wrapper in models]),
                n=num_children,
            )
            np.random.shuffle(parents)

            # Crossover parents and mutate their offspring
            elite_indices = set(range(num_elites))

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

        return (
            [wrapper.as_tuple() for wrapper in novelty_archive],
            [wrapper.as_tuple() for wrapper in self.sorted_models],
            self.runner_stats,
        )

    def get_pareto_fronts(
        self, wrappers: List[ModelWrapper]
    ) -> List[List[ModelWrapper]]:
        """
        Fast non-dominated sort as in NSGA II by Deb et al.
        """
        fronts = [[]]

        for wrapper in wrappers:
            dominated = []
            num_dominated_by = 0

            for other in wrappers:
                if wrapper.dominates(other):
                    dominated.append(other)
                elif other.dominates(wrapper):
                    num_dominated_by += 1

            if num_dominated_by == 0:
                wrapper.rank = 1
                fronts[0].append(wrapper)

            wrapper.dominated = dominated
            wrapper.num_dominated_by = num_dominated_by

        curr_front = 0
        while fronts[curr_front]:
            next_front = []
            for wrapper in fronts[curr_front]:
                for other in wrapper.dominated:
                    other.num_dominated_by -= 1
                    if other.num_dominated_by == 0:
                        other.rank = curr_front + 1
                        next_front.append(other)
            curr_front += 1
            fronts.append(next_front)

        # Remove last (empty) front
        return fronts[:-1]
