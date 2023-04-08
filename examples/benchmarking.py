from evolution.genetic.operators.mutation import MutationOperator
from evolution.genetic.operators.crossover import CrossoverOperator
from evolution.genetic.operators.selection import SelectionOperator
from evolution.genetic.runner import GeneticRunner
from models.generator import ModelGenerator
from models.model import PromoterModel
from models.preset import Preset
from mi_estimation.decoding import DecodingEstimator
from pipeline.one_step_decoding import OneStepDecodingPipeline
from ssa.one_step import OneStepSimulator
from typing import Tuple
from utils.data import ClassWithData
import numpy as np
import time


class BenchmarkingExamples(ClassWithData):
    def __init__(self):
        super().__init__()
        self.model = Preset.simple(2, 1, 1)

    def matrix_exponentials(self):
        """
        Plot the execution time of matrix exponential calculation
        as the size of models is increased.
        """
        repeats = 100

        times = []

        for num_states in range(2, 10):
            total = 0
            for _ in range(repeats):
                model = ModelGenerator.get_random_model(num_states)
                start = time.time()
                model.get_matrix_exp(self.data, self.time_delta)
                total += time.time() - start
            avg_time = total / repeats
            print(f"{num_states} states: {avg_time:.3f}s")
            times.append(avg_time)

        print(total)

    def trajectory_simulation(self):
        """
        Plot the execution time of trajectory simulation as
        the size of models is increased. Two schemes are compared:

          1. Vectorised O(n) approach with np.argwhere
          2. Non-vetorised O(log(n)) approach with np.searchsorted

        See https://github.com/numpy/numpy/issues/4224 for why
        np.searchsorted has yet to work for multidimensional arrays.
        """
        repeats = 10
        states = np.arange(2, 20)

        sim = OneStepSimulator(**self.default_sim_args)
        sim.seed = 27
        res = [[], []]

        # Simulate
        for num_states in states:
            totals = [0, 0]
            for _ in range(repeats):
                model = ModelGenerator.get_random_model(num_states, p_edge=0.5)
                trajectories = []

                for i in (0, 1):
                    start = time.time()
                    sim.binary_search = bool(i)
                    trajectories.append(sim.simulate(model))
                    totals[i] += time.time() - start

                if not np.array_equal(trajectories[0], trajectories[1]):
                    print(
                        "Trajectories are not equal for the same model and random seed!"
                    )

            for i in (0, 1):
                res[i].append(totals[i] / repeats)
            print(
                f"{num_states} states: {', '.join(f'{res[i][-1]:.3f}s' for i in (0, 1))}"
            )

        import matplotlib.pyplot as plt

        plt.plot(states, res[0], label="Vectorised $O(n)$")
        plt.plot(states, res[1], label="Non-vectorised $O(log(n))$")

        plt.xticks(states)

        plt.ylabel("Time (s)")
        plt.xlabel("Number of States")
        plt.legend()

        plt.savefig(f"{self.SAVE_FOLDER}/trajectory_simulation.png")

    def mi_estimation(self):
        """
        See trend of MI estimation execution time as number of
        replicates is increased.
        """
        repeats = 10
        replicate_counts = [1, 5, 10, 50, 100]

        for replicates in replicate_counts:
            pip = OneStepDecodingPipeline(
                **{**self.default_pip_args, "replicates": replicates}
            )

            total = 0
            for _ in range(repeats):
                start = time.time()
                pip.evaluate(self.model)
                total += time.time() - start
            print(f"{replicates} replicates: {total / repeats:.3f}s")

    def mi_estimation_table(self):
        """
        Construct table of MI estimation execution times as
        the number of cell replicates is increased and the classifiers
        used are varied. A timeout is set in case execution takes too long.
        """
        model = self.model

        repeats = 10
        replicate_counts = [1, 2, 5, 10, 20, 50]
        classifiers = ["svm", "random_forest", "decision_tree", "naive_bayes"]
        dims = (len(classifiers), len(replicate_counts))
        TIMEOUT = 300

        res_times = np.zeros(dims) + float("inf")
        res_mi_mean = np.zeros(dims) - 1
        res_mi_std = np.zeros(dims) - 1

        def _benchmark(cls_index: int, rep_index: int) -> None:
            classifier, replicates = classifiers[cls_index], replicate_counts[rep_index]
            print(f"{classifier} - {replicates} reps")

            sim = OneStepSimulator(
                **{**self.default_sim_args, "replicates": replicates}
            )
            trajectory = sim.simulate(model)
            est = DecodingEstimator(
                **{**self.default_est_args, "replicates": replicates}
            )

            total_time = 0
            mi_scores = []
            for _ in range(repeats):
                start = time.time()
                mi_score = est.estimate(model, trajectory)
                total_time += time.time() - start
                mi_scores.append(mi_score)

            res_times[cls_index, rep_index] = total_time / repeats
            res_mi_mean[cls_index, rep_index] = np.average(mi_scores)
            res_mi_std[cls_index, rep_index] = np.std(mi_scores)

        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=1) as executor:
            for i in range(len(classifiers)):
                for j in range(len(replicate_counts)):
                    try:
                        executor.submit(_benchmark, i, j).result(TIMEOUT)
                    except Exception as e:
                        print(e)

        print("Avg execution times:")
        print(res_times)

        print("MI mean:")
        print(res_mi_mean)

        print("MI standard deviation:")
        print(res_mi_std)

    def _sklearn_nested_parallelism__pip_evaluate(
        pip: OneStepDecodingPipeline, model: PromoterModel, index: int
    ) -> Tuple[float, int]:
        mi = pip.evaluate(model, verbose=False)
        return index, mi

    def sklearn_nested_parallelism(self):
        """
        Demonstrate the ability to perform nested parallelism
        (i.e. multiprocessing on top of sklearn's joblib backend)
        by using the dask backend on top.

        Currently, using ProcessPoolExecutor on futures requiring
        n_jobs != 1 in an sklearn function (e.g. GridSearch) causes
        the program to hang - some type of deadlock.
        """
        n_processors = 5
        trials = 2

        pip = OneStepDecodingPipeline(**self.default_pip_args)

        # Parallelised n_job=1 tasks
        from concurrent.futures import ProcessPoolExecutor, as_completed

        start = time.time()
        with ProcessPoolExecutor(
            max_workers=min(n_processors, trials),
        ) as executor:
            futures = []
            for i in range(trials):
                futures.append(
                    executor.submit(
                        BenchmarkingExamples._sklearn_nested_parallelism__pip_evaluate,
                        pip,
                        self.model,
                        i,
                    )
                )

            for future in as_completed(futures):
                mi_score, i = future.result()
                print(f"Future #{i}: {mi_score:.3f}")
        print("\n" * 5 + f"Took {time.time() - start:.3f}s" + "\n" * 5)

        # Nested Parallelism
        import dask
        from dask.distributed import Client
        from sklearn.utils import register_parallel_backend
        import logging

        client = Client(silence_logs=logging.INFO)
        dask.config.set(scheduler="processes")
        register_parallel_backend("distributed", client)

        pip.set_parallel()
        start = time.time()
        futures = []
        for i in range(trials):
            futures.append(
                client.submit(
                    BenchmarkingExamples._sklearn_nested_parallelism__pip_evaluate,
                    pip,
                    self.model,
                    i,
                )
            )
        res = client.gather(futures)
        print(res)
        print("\n" * 5 + f"Took {time.time() - start:.3f}s" + "\n" * 5)

    def genetic_multiprocessing_overhead(self):
        """
        Check that copy-on-write occurs during multiprocessing
        in the genetic algorithm runner. That is, large numpy arrays
        are not deep copied.
        """
        states = 4
        population, iterations = 10, 2

        mutations = [MutationOperator.add_noise]
        crossover = CrossoverOperator.subgraph_swap
        select = SelectionOperator.roulette_wheel
        runner = GeneticRunner(self.data, mutations, crossover, select)

        start = time.time()
        runner.run(
            states=states,
            population=population,
            iterations=iterations,
            model_generator_params={"one_active_state": False},
            verbose=False,
            debug=False,
        )
        print(f"Before large property: {time.time() - start:.3f}s")

        # Add some large overhead
        runner.pip.some_large_property = np.zeros((1000, 1000, 1000))

        start = time.time()
        runner.run(
            states=states,
            population=population,
            iterations=iterations,
            model_generator_params={"one_active_state": False},
            verbose=False,
            debug=False,
        )
        print(f"After large property: {time.time() - start:.3f}s")
