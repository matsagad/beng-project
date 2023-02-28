from concurrent.futures import ThreadPoolExecutor

from evolution.genetic.runner import GeneticRunner
from evolution.genetic.operators.mutation import MutationOperator
from evolution.genetic.operators.crossover import CrossoverOperator
from evolution.genetic.operators.selection import SelectionOperator
from mi_estimation.decoding import DecodingEstimator
from models.generator import ModelGenerator
from models.model import PromoterModel
from models.rates.function import RateFunction as RF
from optimisation.grid_search import GridSearch
from optimisation.particle_swarm import ParticleSwarm
from pipeline.one_step_decoding import OneStepDecodingPipeline
from ssa.one_step import OneStepSimulator
from utils.process import get_tf_data
import json
import numpy as np
import time


class Examples:
    CACHE_FOLDER = "cache/latestv2"
    tf_index = 2  # dot6
    a, b, c = 1.0e0, 1.0e0, 1.0e0
    # Parameters found from particle swarm
    m, n, p, q = (
        10 ** (1.98472999),
        10 ** (0.80802623),
        10 ** (-1.86577579),
        10 ** (1.41144266),
    )
    models = {
        2: PromoterModel(
            rate_fn_matrix=[
                [None, RF.Linear([a], [tf_index])],
                [RF.Constant([b]), None],
            ]
        ).with_active_states([1]),
        3: PromoterModel(
            rate_fn_matrix=[
                [None, None, RF.Constant([m])],
                [None, None, RF.Constant([n])],
                [
                    RF.Linear([p], [tf_index - 1]),
                    RF.Linear([q], [tf_index]),
                    None,
                ],
            ]
        ).with_active_states([0, 1]),
        4: PromoterModel(
            rate_fn_matrix=[
                [None, None, None, RF.Constant([a])],
                [None, None, None, RF.Constant([b])],
                [None, None, None, RF.Constant([c])],
                [
                    RF.Linear([a], [tf_index]),
                    RF.Linear([b], [tf_index + 1]),
                    RF.Linear([c], [tf_index + 2]),
                    None,
                ],
            ]
        ).with_active_states([0, 1, 2]),
    }

    class UsingThePipeline:
        def pipeline_example():
            # Import data
            ## Batched data (~119 replicates)
            data, origin, time_delta, _ = get_tf_data()
            print(data.shape)  # num envs, num tfs, replicates, time stamps

            ## Examples for debugging batching process
            single_replicate = np.array(data[:, :, :1])
            single_environment = np.array(data[:1])

            # Set-up model
            model = Examples.models[3]

            # Simulate and evaluate
            pipeline = OneStepDecodingPipeline(data, origin=origin, tau=time_delta)
            pipeline.evaluate(model, verbose=True)

    class PlottingVisuals:
        def visualise_model_example():
            # Set-up model
            tf_index = 0
            a, b, c = 1.0e-3, 1.0e-2, 1.0e-2

            # Changes in matrix will reflect in visualisation!
            model = PromoterModel(
                rate_fn_matrix=[
                    [None, None, None, RF.Constant([a])],
                    [None, None, None, RF.Constant([b])],
                    [None, None, None, RF.Constant([c])],
                    [
                        RF.Linear([a], [tf_index]),
                        RF.Linear([b], [tf_index + 1]),
                        RF.Linear([c], [tf_index + 2]),
                        None,
                    ],
                ]
            ).with_active_states([0, 1, 2])

            # Visualise
            model.visualise(save=True, fname="cache/model10.png")

        def visualise_trajectory_example():
            data, _, time_delta, _ = get_tf_data()
            model = Examples.models[4]

            # Simulate
            sim = OneStepSimulator(data, tau=time_delta, realised=True)
            trajectories = sim.simulate(model)
            print(trajectories.shape)

            # Visualise single cell
            print("Single-cell time series")
            OneStepSimulator.visualise_trajectory(
                trajectories, model=model, average=False, batch_num=1
            )

            # Visualise average
            print("Average-cell time series")
            OneStepSimulator.visualise_trajectory(
                trajectories, model=model, average=True
            )

        def visualise_realised_probabilistic_trajectories():
            data, _, time_delta, _ = get_tf_data()
            model = Examples.models[2]

            print("Realised (initial: [0.5, 0.5])")
            OneStepSimulator.visualise_trajectory(
                OneStepSimulator(
                    data, tau=time_delta, realised=True, replicates=3
                ).simulate(model),
                model=model,
            )

            print("Probabilistic (initial: [0.5, 0.5])")
            OneStepSimulator.visualise_trajectory(
                OneStepSimulator(data, tau=time_delta, realised=False).simulate(model),
                model=model,
            )

        def visualise_activity():
            data, origin, time_delta, _ = get_tf_data()

            # model = Examples.models[2]
            model = PromoterModel(
                [[None, RF.Linear([10], [4])], [RF.Linear([4.7], [2]), None]]
            )
            model = PromoterModel(
                [
                    [None, RF.Linear([1], [2]), RF.Linear([10], [4])],
                    [RF.Linear([1], [2]), None, RF.Linear([10], [4])],
                    [RF.Linear([1], [2]), RF.Linear([1], [2]), None],
                ]
            ).with_active_states([0, 1])
            replicates = 10
            interval = OneStepDecodingPipeline.FIXED_INTERVAL
            est = DecodingEstimator(origin, interval, "naive_bayes")

            # Raw Nuclear Translocation Trajectory
            TIME_AXIS = 2
            raw_data = np.moveaxis(data[:, 2], TIME_AXIS, 0)
            raw_data = raw_data.reshape((*raw_data.shape, 1))
            split_raw = est._split_classes(
                PromoterModel([[RF.Constant([1])]]), raw_data
            )

            # Simulated Trajectory
            sim = OneStepSimulator(
                data, tau=time_delta, realised=True, replicates=replicates
            )
            trajectories = sim.simulate(model)
            print(est.estimate(model, trajectories))
            split_sim = est._split_classes(model, trajectories)

            num_envs = data.shape[0]

            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(2, num_envs + 1, sharey="row", figsize=(5, 10))

            for i, pair in enumerate(zip(split_sim, split_raw)):
                res_sim, res_raw = pair
                im = axes[0][i].imshow(
                    res_sim,
                    cmap="rainbow",
                    aspect="auto",
                    interpolation="none",
                    vmin=0,
                    vmax=1,
                )
                im = axes[1][i].imshow(
                    res_raw,
                    cmap="rainbow",
                    aspect="auto",
                    interpolation="none",
                    vmin=0,
                    vmax=1,
                )

            fig.colorbar(im, ax=axes, location="bottom")

            plt.savefig(f"{Examples.CACHE_FOLDER}/binary_vis_best2.png", dpi=200)

    class Benchmarking:
        def matrix_exponentials():
            data, _, _, _ = get_tf_data()
            model = Examples.models[2]

            start = time.time()
            mat = model.get_matrix_exp(data, 2.5)
            print(time.time() - start)

        def trajectory():
            data, _, time_delta, _ = get_tf_data()
            model = Examples.models[2]

            # Simulate
            for replicates in [1, 1, 5, 10, 50]:
                repeats, total = 10, 0
                for _ in range(repeats):
                    start = time.time()

                    sim = OneStepSimulator(
                        data, tau=time_delta, realised=True, replicates=replicates
                    )
                    trajectories = sim.simulate(model)

                    total += time.time() - start

                print(f"{replicates} replicates: {total / repeats}")

        def mi_estimation():
            data, origin, time_delta, _ = get_tf_data()
            model = Examples.models[2]
            interval = OneStepDecodingPipeline.FIXED_INTERVAL
            est = DecodingEstimator(origin, interval, "naive_bayes")
            est.parallel = True

            for replicates in [1, 5, 10, 50, 100, 200]:
                # Simulate
                sim = OneStepSimulator(
                    data, tau=time_delta, realised=True, replicates=replicates
                )
                trajectories = sim.simulate(model)

                # Estimate MI
                print("Estimating MI...")
                start = time.time()
                mi_score = est.estimate(model, trajectories)
                print(f"{replicates} replicates: {time.time() - start}")

                print(f"MI: {mi_score}")

        def max_mi_estimation():
            data, origin, _, tf_names = get_tf_data()
            interval = OneStepDecodingPipeline.FIXED_INTERVAL
            dummy_model = PromoterModel([RF.Constant([1])])

            est = DecodingEstimator(origin, interval, "naive_bayes")
            for tf_index, tf in enumerate(tf_names):
                for replicates in [1, 1, 1, 1, 1, 1]:
                    TIME_AXIS = 2
                    raw_data = np.moveaxis(data[:, tf_index], TIME_AXIS, 0)
                    raw_data = raw_data.reshape((*raw_data.shape, 1))
                    # To test repeated samples are handled
                    rep_raw_data = np.tile(raw_data, (replicates, 1))

                    # Estimate MI
                    split_data = est._split_classes(dummy_model, raw_data)

                    start = time.time()
                    mi_score = est._estimate(split_data, add_noise=False)
                    print(time.time() - start)

                    print(f"{tf} {replicates} MI: {mi_score}")

        def mi_estimation_table():
            # (for simple model only)
            data, origin, time_delta, _ = get_tf_data()
            model = Examples.models[2]
            interval = OneStepDecodingPipeline.FIXED_INTERVAL

            replicates = [1, 2, 5, 10, 20, 50]
            classifiers = ["svm", "random_forest", "decision_tree", "naive_bayes"]
            TIMEOUT = 300

            res_times = np.zeros((len(classifiers), len(replicates))) + float("inf")
            res_mi = np.zeros((len(classifiers), len(replicates))) - 1

            def _benchmark(cls_index: int, rep_index: int) -> None:
                classifier, replicate = classifiers[cls_index], replicates[rep_index]
                print(f"{classifier} - {replicate} reps")
                sim = OneStepSimulator(
                    data, tau=time_delta, realised=True, replicates=replicate
                )
                trajectory = sim.simulate(model)
                est = DecodingEstimator(origin, interval, classifier)
                start = time.time()
                mi_score = est.estimate(model, trajectory)
                res_times[cls_index, rep_index] = time.time() - start
                res_mi[cls_index, rep_index] = mi_score

            with ThreadPoolExecutor(max_workers=1) as executor:
                for i in range(len(classifiers)):
                    for j in range(len(replicates)):
                        try:
                            executor.submit(_benchmark, i, j).result(TIMEOUT)
                        except Exception as e:
                            print(e)

            print(res_times)
            print(res_mi)

        def mi_vs_interval():
            data, origin, time_delta, _ = get_tf_data()
            model = Examples.models[2]
            reps = 3
            scores = []

            for interval in range(1, origin):
                replicates = 20
                est = DecodingEstimator(origin, interval, "naive_bayes")
                est.parallel = True

                # Simulate
                sim = OneStepSimulator(
                    data, tau=time_delta, realised=True, replicates=replicates
                )
                trajectories = sim.simulate(model)

                # Estimate MI
                print("Estimating MI...")
                start = time.time()

                mi_score = 0
                for _ in range(reps):
                    mi_score += est.estimate(model, trajectories)
                mi_score = mi_score / reps

                print(f"{interval} interval: {time.time() - start}")
                print(f"MI: {mi_score}")
                scores.append(mi_score)

            print(scores)

            import matplotlib.pyplot as plt

            plt.plot([i for i in range(1, origin)], scores)
            plt.ylabel("MI")
            plt.xlabel("Length of time interval from origin")
            plt.savefig(f"{Examples.CACHE_FOLDER}/mi_vs_interval.png", dpi=100)

        def mi_vs_repeated_intervals():
            data, origin, time_delta, _ = get_tf_data()
            interval = OneStepDecodingPipeline.FIXED_INTERVAL

            # Scale controls number of repeated intervals
            scale = 2

            origin *= scale
            time_delta /= scale
            interval *= scale
            data = np.repeat(data, scale, axis=3)

            model = Examples.models[2]
            est = DecodingEstimator(origin, interval, "svm")

            for replicates in [1]:
                # Simulate
                sim = OneStepSimulator(
                    data[:1], tau=time_delta, realised=True, replicates=replicates
                )
                trajectories = sim.simulate(model)

                # Estimate MI
                print("Estimating MI...")
                start = time.time()
                mi_score = est.estimate(model, trajectories)
                print(f"{replicates} replicates: {time.time() - start}")

                print(f"MI: {mi_score}")

        def _evaluate(estimator, model, trajectories, i):
            mi = estimator.estimate(model, trajectories)
            return mi, i

        def mi_distribution():
            data, origin, time_delta, _ = get_tf_data()
            model = Examples.models[2]

            interval = OneStepDecodingPipeline.FIXED_INTERVAL
            est = DecodingEstimator(origin, interval, "naive_bayes")
            est.parallel = False

            iters = 25
            rep_count = [1, 2, 5, 10, 20]
            n_processors = 10

            fname = f"mi_dist_rand_{iters}_{'_'.join(map(str,rep_count))}.dat"
            import pickle, os
            from concurrent.futures import ProcessPoolExecutor, as_completed

            if os.path.isfile(fname):
                print("Using cached MI distribution.")
                with open(fname, "rb") as f:
                    hist_map = pickle.load(f)
            else:
                hist_map = dict()

                for reps in rep_count:
                    hist_map[reps] = []

                    # Simulate
                    sim = OneStepSimulator(
                        data, tau=time_delta, realised=True, replicates=reps
                    )
                    trajectories = sim.simulate(model)

                    # Estimate MI
                    with ProcessPoolExecutor(
                        max_workers=min(n_processors, iters),
                    ) as executor:
                        futures = []
                        for i in range(iters):
                            futures.append(
                                executor.submit(
                                    Examples.Benchmarking._evaluate,
                                    est,
                                    model,
                                    trajectories,
                                    i,
                                )
                            )

                        for future in as_completed(futures):
                            mi_score, i = future.result()
                            hist_map[reps].append(mi_score)
                            print(f"{reps}-{i}: {mi_score:.3f}")

                with open(fname, "wb") as f:
                    pickle.dump(hist_map, f)
                    print("Cached best MI distribution.")

            import matplotlib.pyplot as plt

            # plt.style.use("seaborn-deep")

            bins = np.linspace(0, 0.6, 60)

            for reps, hist in hist_map.items():
                plt.hist(hist, bins, alpha=0.5, label=f"{reps} reps", edgecolor="black")

            # plt.hist(list(hist_map.values()), bins, label=list(hist_map.keys()))
            plt.legend(loc="upper right")
            plt.savefig(f"{Examples.CACHE_FOLDER}/mi_distribution.png", dpi=200)

    class Optimisation:
        def grid_search():
            data, _, _, tf_names = get_tf_data()
            gs = GridSearch()
            gs.optimise_simple(data, tf_names)

        def particle_swarm():
            data, _, _, _ = get_tf_data()
            ps = ParticleSwarm()
            ps.optimise_simple(data)

    class Evolution:
        def genetic_simple():
            model = Examples.models[3]
            print(
                [
                    rate_fn.rates
                    for row in model.rate_fn_matrix
                    for rate_fn in row
                    if rate_fn is not None
                ]
            )
            for _ in range(10):
                MutationOperator.add_noise(model)
                print(
                    [
                        rate_fn.rates
                        for row in model.rate_fn_matrix
                        for rate_fn in row
                        if rate_fn is not None
                    ]
                )

        def model_generation():
            for i in range(1):
                ModelGenerator.get_random_model(10).visualise(
                    save=True, fname=f"cache/latestv2/model{i}.png"
                )

        def evolutionary_run():
            import pickle

            data, _, _, _ = get_tf_data()

            states = 5
            population, iterations = 20, 20
            fname = (
                f"best_models_roulette_simple_{states}_{population}_{iterations}.dat"
            )

            mutations = [
                MutationOperator.edit_edge,
                MutationOperator.add_edge,
                MutationOperator.flip_tf,
                MutationOperator.add_noise,
            ]
            crossover = CrossoverOperator.one_point_triangular_row_swap
            select = SelectionOperator.roulette_wheel
            runner = GeneticRunner(data, mutations, crossover, select)

            models = runner.run(
                states=states,
                population=population,
                iterations=iterations,
                verbose=True,
                debug=True,
            )

            with open(fname, "wb") as f:
                pickle.dump(models, f)
                print("Cached best models")

        def load_best_models():
            import pickle

            data, _, _, _ = get_tf_data()
            states = 5
            population, iterations = 40, 20
            fname = (
                f"best_models_tournament_simple_{states}_{population}_{iterations}.dat"
            )

            with open(fname, "rb") as f:
                models = pickle.load(f)

            pip = OneStepDecodingPipeline(
                data, realised=True, replicates=10, classifier_name="naive_bayes"
            )

            for i, model in enumerate(models[:3]):
                print(pip.evaluate(model))
                model.visualise(
                    save=True,
                    fname=f"cache/evolution/{states}_{population}_{iterations}_{i}.png",
                )

        def crossover_no_side_effects():
            model1 = PromoterModel(
                [[None, RF.Constant([1.23])], [RF.Linear([2.345], [1]), None]]
            )
            model2 = PromoterModel(
                [[None, RF.Linear([4.56], [2])], [RF.Constant([0.123]), None]]
            )
            for model in (model1, model2):
                print(model.hash()[:6])

            data, _, _, _ = get_tf_data()
            mutations = [
                MutationOperator.add_noise,
                MutationOperator.add_edge,
                MutationOperator.edit_edge,
                MutationOperator.flip_tf,
            ]
            crossover = CrossoverOperator.one_point_triangular_row_swap
            select = SelectionOperator.tournament
            runner = GeneticRunner(data, mutations, crossover, select)

            # Model1 can change hash but model2 must remain the same
            children = runner.crossover(model1, model2, False, True)
            for child in children:
                for _ in range(100):
                    runner.mutate(child)

            for model in (model1, model2):
                print(model.hash()[:6])

        def models_generated_are_valid():
            # Random models are valid
            for _ in range(100):
                model = ModelGenerator.get_random_model(10, p_edge=0.5)
                if not ModelGenerator.is_valid(model, verbose=True):
                    print("not valid yoo")

            data, _, _, _ = get_tf_data()
            mutations = [
                MutationOperator.add_noise,
                MutationOperator.add_edge,
                MutationOperator.edit_edge,
                MutationOperator.flip_tf,
            ]
            crossover = CrossoverOperator.one_point_triangular_row_swap
            select = SelectionOperator.tournament
            runner = GeneticRunner(data, mutations, crossover, select)

            models = [
                ModelGenerator.get_random_model(10, p_edge=0.5),
                ModelGenerator.get_random_model(10, p_edge=0.5),
            ]

            # Crossover maintains reversibility
            for _ in range(100):
                models = runner.crossover(*models, False, False)
                for model in models:
                    ModelGenerator.is_valid(model, verbose=True)

            # Mutations maintain reversibility
            model = models[0]
            for _ in range(100):
                model = runner.mutate(model)
                ModelGenerator.is_valid(model, verbose=True)

        def test_random_model_variance():
            data, _, _, _ = get_tf_data()
            pip = OneStepDecodingPipeline(
                data, replicates=10, classifier_name="naive_bayes"
            )
            pip.set_parallel()

            for _ in range(10):
                model = ModelGenerator.get_random_model(2, p_edge=0.5)
                trajectories = pip.simulator.simulate(model)
                mi_score = pip.estimator.estimate(model, trajectories, add_noise=False)
                print(mi_score)

                # If close to boundary, then repeat to ensure no errors occur
                if mi_score < 0.1:
                    for _ in range(10):
                        mi_score = pip.estimator.estimate(
                            model, trajectories, add_noise=False
                        )
                        print(f"\t{mi_score}")

        def test_hypothetical_perfect_model():
            data, origin, _, _ = get_tf_data()
            pip = OneStepDecodingPipeline(
                data, replicates=10, classifier_name="svm"
            )
            pip.set_parallel()

            dummy_model = PromoterModel(
                [[None, RF.Constant([1])], [RF.Constant([1]), None]]
            ).with_active_states([0])
            dummy_traj = np.zeros(pip.simulator.simulate(dummy_model).shape)

            # Rich state left as is. Note: first state is active state
            dummy_traj[origin : origin + origin // 4, 0, :, 0] = 1
            dummy_traj[origin : origin + 2 * origin // 4, 1, :, 0] = 1
            dummy_traj[origin : origin + 3 * origin // 4, 2, :, 0] = 1

            mi_score = pip.estimator.estimate(dummy_model, dummy_traj)
            print(mi_score)

    class Data:
        def find_labels():
            fpath = "data/"
            fnames = ["fig2_stress_type_expts.json", "figS1_nuclear_marker_expts.json"]

            # DFS on JSON structure
            for fname in fnames:
                print(fname)
                with open(fpath + fname, "r") as f:
                    data = json.load(f)
                    stack = [(k, v, "\t") for (k, v) in data.items()]

                    while stack:
                        label, items, prefix = stack.pop()
                        print(prefix + label)
                        if not isinstance(items, dict):
                            continue
                        stack.extend((k, v, "\t" + prefix) for (k, v) in items.items())

        def load_data():
            ts, origin, time_delta, tf_names = get_tf_data()
            print(ts.shape)
            print(origin)
            print(time_delta)
            print(tf_names)


def main():
    # Examples.Benchmarking.trajectory()
    # Examples.Benchmarking.mi_estimation()
    # Examples.Benchmarking.max_mi_estimation()
    # Examples.Benchmarking.mi_estimation_table()
    # Examples.Benchmarking.mi_vs_interval()
    # Examples.Benchmarking.mi_vs_repeated_intervals()
    # Examples.Benchmarking.mi_distribution()

    # Examples.PlottingVisuals.visualise_model_example()
    # Examples.PlottingVisuals.visualise_trajectory_example()
    # Examples.PlottingVisuals.visualise_realised_probabilistic_trajectories()
    # Examples.PlottingVisuals.visualise_activity()

    # Examples.UsingThePipeline.pipeline_example()

    # Examples.Optimisation.grid_search()
    # Examples.Optimisation.particle_swarm()

    # Examples.Evolution.genetic_simple()
    # Examples.Evolution.model_generation()
    # Examples.Evolution.evolutionary_run()
    # Examples.Evolution.load_best_models()
    # Examples.Evolution.crossover_no_side_effects()
    # Examples.Evolution.models_generated_are_valid()
    # Examples.Evolution.test_random_model_variance()
    Examples.Evolution.test_hypothetical_perfect_model()

    # Examples.Data.find_labels()
    # Examples.Data.load_data()


if __name__ == "__main__":
    main()
