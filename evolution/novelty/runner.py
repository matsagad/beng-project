from concurrent.futures import ProcessPoolExecutor, as_completed
from evolution.novelty.metrics import TrajectoryMetric, TopologyMetric
from evolution.wrapper import ModelWrapper
from functools import reduce
from joblib import parallel_backend
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

        self.models = []
        self.novelty_archive = []
        self.runner_stats = {"avg_time_duration": []}

    def evaluate_wrapper(
        model: PromoterModel,
        index: int,
        find_classes: bool = False,
        find_feature_vector: bool = False,
    ) -> Tuple[int, float, float]:
        return (
            index,
            *NoveltySearchRunner.mp_instance._evaluate(
                model, find_classes, find_feature_vector
            ),
        )

    def _evaluate(
        self, model: PromoterModel, find_classes: bool, find_feature_vector: bool
    ) -> Tuple[float, float, NDArray | None]:
        mi = self.pip.evaluate(model, verbose=False)
        classes, feature_vector = None, None
        if find_classes:
            dist_traj = self.prob_pip.simulator.simulate(model)
            classes = np.mean(
                self.prob_pip.estimator._split_classes(model, dist_traj), axis=1
            )
        if find_feature_vector:
            feature_vector = TopologyMetric.get_feature_vector(model)
        return self.scale_fitness(model, mi), mi, classes, feature_vector

    def run(
        self,
        states: int,
        population: int = 100,
        elite_ratio: float = 0.2,
        iterations: int = 100,
        linear_metric: bool = True,
        novelty_threshold: float = -1,
        archival_rate_threshold: int = 4,
        archival_stagnation_threshold: int = 3,
        max_archival_rate: int = -1,
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
            # Use Jensen-Shannon divergence on trajectories
            metric = TrajectoryMetric.rms_js_metric_for_trajectories
            epsilon = 0.01
        else:
            # Use Wasserstein Weisfeiler-Lehman distance on line digraph
            metric = TopologyMetric.wwl_metric_for_serialised_wl_feature_vectors
            epsilon = 0.05

        nn = NearestNeighbors(
            n_neighbors=n_neighbors + 1,  # excluding self
            algorithm="ball_tree",
            metric=metric,
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
                        find_feature_vector = (
                            not linear_metric and wrapper.feature_vector is None
                        )
                        futures.append(
                            executor.submit(
                                NoveltySearchRunner.evaluate_wrapper,
                                wrapper.model,
                                i,
                                find_classes,
                                find_feature_vector,
                            )
                        )
                    else:
                        heapq.heappush(top_models, wrapper)

                for future in as_completed(futures):
                    (
                        i,
                        curr_fitness,
                        curr_mi,
                        curr_classes,
                        curr_feature_vector,
                    ) = future.result()
                    wrapper = models[i]
                    avg_fitness, avg_mi = wrapper.fitness, wrapper.mi
                    runs_left = wrapper.runs_left

                    if wrapper.classes is None:
                        wrapper.classes = curr_classes
                    if wrapper.feature_vector is None:
                        wrapper.feature_vector = curr_feature_vector

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
            collective = models + novelty_archive
            if linear_metric:
                nn_arrays = np.array(
                    [
                        wrapper.as_nn_array(linear=linear_metric)
                        for wrapper in collective
                    ]
                )
            else:
                feature_vectors = [
                    wrapper.as_nn_array(linear=linear_metric) for wrapper in collective
                ]
                nn_arrays = TopologyMetric.serialise(feature_vectors)

            nn.fit(nn_arrays)
            with parallel_backend("loky"):
                collective_distances, collective_neighbors = (
                    arr[:, 1:] for arr in nn.kneighbors(nn_arrays, return_distance=True)
                )

            novelties = [
                # Scale by two as max(MI)=2, max(Novelty)=1
                self.scale_fitness(wrapper.model, 2 * d) / 2
                for d, wrapper in zip(
                    np.average(collective_distances, axis=1), collective
                )
            ]
            fitnesses = np.array([wrapper.fitness for wrapper in collective])

            # Find two nearest neighbors in the archive to possibly replace them
            if novelty_archive:
                nn.fit(nn_arrays[len(models) :])
                with parallel_backend("loky"):
                    archive_distances, archive_neighbors = nn.kneighbors(
                        nn_arrays[: len(models)],
                        n_neighbors=min(2, len(novelty_archive)),
                        return_distance=True,
                    )

            else:
                archive_distances = archive_neighbors = [-1, -1]

            # If novelty threshold is not specified then set it to the top Nth model's novelty
            # where N is the max_archival_rate or (arbitrarily chosen) minimum between 10 and 10%
            if iter == 0 and novelty_threshold == -1:
                max_in_first_iteration = (
                    max_archival_rate
                    if max_archival_rate > 0
                    else min(10, population // 10)
                )
                novelty_threshold = (
                    sorted(novelties, reverse=True)[:max_in_first_iteration][-1] - 0.001
                )

            # Archive models and adjust thresholds if necessary
            models_to_archive = []
            swaps = {}
            for wrapper, k_neighbors, novelty in zip(
                collective, collective_neighbors, novelties
            ):
                wrapper.local_fitness = np.sum(wrapper.fitness > fitnesses[k_neighbors])
                wrapper.novelty = novelty

            if novelty_archive:
                for i, (wrapper, n_archive_neighbors, n_archive_distances) in enumerate(
                    zip(models, archive_neighbors, archive_distances)
                ):
                    closest_archived_neighbor = novelty_archive[n_archive_neighbors[0]]

                    if np.all(n_archive_distances > novelty_threshold):
                        # Add model to archive if the distance to its two nearest neighbors in the archive
                        # are above the novelty threshold
                        models_to_archive.append(wrapper)
                    elif self.exclusively_epsilon_dominated(
                        wrapper, closest_archived_neighbor, epsilon=epsilon
                    ):
                        # Swap model for its closest archived neighbor if epsilon dominated
                        swap_ref = (wrapper.fitness, i)
                        if (
                            closest_archived_neighbor.archive_position not in swaps
                            or swap_ref[0]
                            > swaps[closest_archived_neighbor.archive_position][0]
                        ):
                            swaps[closest_archived_neighbor.archive_position] = swap_ref
            else:
                models_to_archive.extend(
                    wrapper for wrapper in models if wrapper.novelty > novelty_threshold
                )

            for archive_position, (_, index) in swaps.items():
                wrapper = models[index]
                archived_to_swap = novelty_archive[archive_position]

                models[index] = archived_to_swap
                novelty_archive[archive_position] = wrapper

                wrapper.archive_position = archive_position
                archived_to_swap.archive_position = -1

            ## Archive models if constraints permit
            if max_archival_rate > 0:
                models_to_archive = models_to_archive[:max_archival_rate]

            curr_archive_size = len(novelty_archive)
            for i, wrapper in enumerate(models_to_archive):
                wrapper.archive_position = curr_archive_size + i

            novelty_archive.extend(models_to_archive)
            self.novelty_archive = novelty_archive

            ## Adjust threshold
            num_models_archived = len(models_to_archive)

            if num_models_archived > archival_rate_threshold:
                novelty_threshold = min(1, 1.05 * novelty_threshold)

            if num_models_archived == 0 and len(swaps) == 0:
                num_iterations_without_archive += 1
            else:
                num_iterations_without_archive = 0

            if num_iterations_without_archive >= archival_stagnation_threshold:
                novelty_threshold *= 0.95

            # Keep Pareto optimal front and those succeeding as elites
            ## Exclude archived models from being delegated to a front but
            ## still assign them with a rank (i.e. front number).
            fronts = self.get_pareto_fronts(models, exclude_archived=True)
            elite = []

            i = 0
            while i < len(fronts) and len(elite) + len(fronts[i]) <= num_elites:
                elite.extend(fronts[i])
                i += 1

            remaining = num_elites - len(elite)
            curr_front = (
                [] if i >= len(fronts) else sorted(fronts[i], key=lambda x: -x.novelty)
            )

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
                    "\tPareto-based Elites: "
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
                runner_stats[label]["std_fitness"].append(np.std(fitnesses))

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
            max_rank = max(wrapper.rank for wrapper in models)
            parents = self.select(
                np.array([max_rank - wrapper.rank for wrapper in models]),
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
            self.models = models

        return (
            [wrapper.as_tuple() for wrapper in self.novelty_archive],
            [wrapper.as_tuple() for wrapper in self.models],
            self.runner_stats,
        )

    def get_pareto_fronts(
        self, wrappers: List[ModelWrapper], exclude_archived: bool = True
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
                    if other.num_dominated_by > 0:
                        continue
                    other.rank = curr_front + 1
                    next_front.append(other)
            curr_front += 1
            fronts.append(next_front)

        # Remove last (empty) front
        return [
            [
                wrapper
                for wrapper in front
                if not exclude_archived or wrapper.archive_position == -1
            ]
            for front in fronts[:-1]
        ]

    def exclusively_epsilon_dominated(
        self, wrapper: ModelWrapper, other: ModelWrapper, epsilon: float = 0.01
    ) -> bool:
        return (
            (wrapper.novelty >= (1 - epsilon) * other.novelty)
            and (wrapper.fitness >= (1 - epsilon) * other.fitness)
            and (
                (wrapper.novelty - other.novelty) * other.fitness
                > -(wrapper.fitness - other.fitness) * other.novelty
            )
        )
