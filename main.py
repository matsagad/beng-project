from concurrent.futures import ThreadPoolExecutor
from typing import Tuple
from evolution.genetic.runner import GeneticRunner
from evolution.genetic.penalty import ModelPenalty
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
from utils.process import get_tf_data as get_data
import json
import numpy as np
import sys
import time
import pickle

def unpickle(fname):
    object = None
    with open(fname, "rb") as f:
        object = pickle.load(f)
    return object

def get_tf_data(**kwargs):
    args = sys.argv[1:]
    if not args:
        return get_data(**kwargs)
    return get_data(cache_folder=args[0], **kwargs)


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
        ).with_equal_active_states([1]),
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
        ).with_equal_active_states([0, 1]),
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
        ).with_equal_active_states([0, 1, 2]),
        "best_2": PromoterModel(
            [[None, RF.Linear([10], [4])], [RF.Linear([4.7], [2]), None]]
        ),
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
            ).with_equal_active_states([0, 1, 2])

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

        def visualise_tf_concentration():
            import matplotlib.pyplot as plt

            data, origin, time_delta, tf_names = get_tf_data(
                scale=True, local_scale=True
            )

            num_envs, num_tfs = data.shape[:2]

            fig, axes = plt.subplots(
                num_tfs, num_envs + 1, sharey="row", sharex=True, figsize=(5, 10)
            )
            fig.tight_layout()
            interval = OneStepDecodingPipeline.FIXED_INTERVAL
            est = DecodingEstimator(origin, interval, "naive_bayes")
            plt.subplots_adjust(wspace=0.1, hspace=0.1)

            # Raw Nuclear Translocation Trajectory
            TIME_AXIS = 2
            for i, row in enumerate(axes):
                raw_data = np.moveaxis(data[:, i], TIME_AXIS, 0)
                raw_data = raw_data.reshape((*raw_data.shape, 1))
                split_raw = est._split_classes(
                    PromoterModel([[RF.Constant([1])]]), raw_data
                )

                for j, ax in enumerate(row):
                    ax.imshow(
                        split_raw[j],
                        cmap="rainbow",
                        aspect="auto",
                        interpolation="none",
                        vmin=0,
                        vmax=1,
                    )
                    # ax.yaxis.set_ticks_position("right")
                    ax.set_yticks([])
                row[0].set_ylabel(tf_names[i])

            fst_row = axes[0]
            env_labels = ["rich", "carbon", "osmotic", "oxidative"]

            for ax, label in zip(fst_row, env_labels):
                ax.xaxis.set_label_position("top")
                ax.set_xlabel(label)

            plt.savefig(f"{Examples.CACHE_FOLDER}/tf_conc.png", dpi=200, pad_inches=0.2)

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
            ).with_equal_active_states([0, 1])
            model = ModelGenerator.get_random_model(5, one_active_state=False)

            fname = "jobs/7324363_models.dat"

            models = unpickle(fname)
            _, _, _, model = models[0]

            print(model.num_states)
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

        def visualise_crossover():
            import matplotlib.pyplot as plt

            model1 = ModelGenerator.get_random_model(
                5, p_edge=0.1, one_active_state=False
            )
            model2 = ModelGenerator.get_random_model(
                3, p_edge=0.1, one_active_state=False
            )
            crossover = CrossoverOperator.subgraph_swap

            print("beginning swap")
            model_pairs = ((model1, model2), crossover(model1, model2))
            print("finished swap")

            fig, axes = plt.subplots(2, 2, sharey=True, sharex=True, figsize=(6, 6))
            fig.tight_layout()

            for row, model_pair in zip(axes, model_pairs):
                for ax, model in zip(row, model_pair):
                    model.visualise(target_ax=ax)

            for ax, label in zip(axes[:, 0], ("Parents", "Children")):
                ax.set_ylabel(label)

            plt.subplots_adjust(wspace=0, hspace=0)

            plt.savefig(
                f"{Examples.CACHE_FOLDER}/crossover_vis.png",
                dpi=200,
                bbox_inches="tight",
                pad_inches=0,
            )

        def visualise_crossover_chart():
            import matplotlib.pyplot as plt

            rows, cols = 5, 5
            crossover = CrossoverOperator.subgraph_swap

            scale = 8 / max(rows, cols)
            fig = plt.figure(figsize=(scale * cols, scale * rows))

            subfigs = fig.subfigures(
                rows,
                cols,
                height_ratios=np.ones(rows),
                width_ratios=np.ones(cols),
            )

            for dim in (rows, cols):
                if dim == 1:
                    subfigs = [subfigs]

            for row in subfigs:
                for subfig in row:
                    axes = subfig.subplots(
                        2,
                        2,
                        sharey=True,
                        sharex=True,
                    )
                    parents = [
                        ModelGenerator.get_random_model(states=states, p_edge=p_edge)
                        for states, p_edge in zip(
                            2 + np.random.choice(8, size=2), np.random.uniform(size=2)
                        )
                    ]

                    models = (parents, crossover(*parents))

                    for model_pair, subrow in zip(models, axes):
                        for model, ax in zip(model_pair, subrow):
                            ax.set_aspect("equal")
                            model.visualise(target_ax=ax, small_size=True)
                            ax.axis("off")

                    for ax, label in zip(axes[:, 0], ("Parents", "Children")):
                        ax.set_ylabel(label)
                    parents = models[1]

                    subfig.subplots_adjust(wspace=0, hspace=0)

            plt.axis("off")
            print("saving figure...")
            plt.savefig(
                f"{Examples.CACHE_FOLDER}/crossover_chart.png",
                dpi=200,
                bbox_inches="tight",
                pad_inches=0.1,
            )

        def visualise_crossbreeding():
            import matplotlib.pyplot as plt

            iterations = 10
            crossover = CrossoverOperator.one_point_triangular_row_swap

            scale = 8 / iterations

            fig, axes = plt.subplots(
                nrows=2,
                ncols=iterations,
                sharex=True,
                sharey=True,
                figsize=(scale * iterations, scale),
            )
            fig.tight_layout()

            models = [
                ModelGenerator.get_random_model(states=states, p_edge=p_edge)
                for states, p_edge in zip((5, 5), (0.1, 0.5))
            ]

            for col in axes.T:
                for model, ax in zip(models, col):
                    ax.set_aspect("equal")
                    ax.axis("off")
                    model.visualise(target_ax=ax, small_size=True)

                models = crossover(*models)

            plt.axis("off")
            print("saving figure...")
            plt.savefig(
                f"{Examples.CACHE_FOLDER}/crossbreeding.png",
                dpi=200,
                bbox_inches="tight",
                pad_inches=0.1,
            )

        def visualise_tf_grid_activity():
            data, _, time_delta, tf_names = get_tf_data()
            num_envs, num_tfs, num_cells, num_times = data.shape

            from matplotlib.animation import FuncAnimation
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(
                num_tfs, num_envs, sharex=True, sharey=True, figsize=(num_envs, num_tfs)
            )
            fig.tight_layout()
            plt.subplots_adjust(wspace=0, hspace=0)
            for row in axes:
                for ax in row:
                    ax.imshow(np.zeros((10, 10)), aspect="auto", vmin=0, vmax=1)
                    plt.setp(ax.get_xticklabels(), visible=False)
                    plt.setp(ax.get_yticklabels(), visible=False)
                    ax.tick_params(axis="both", which="both", length=0)

            env_labels = ["carbon", "osmotic", "oxidative"]

            for ax, label in zip(axes[0], env_labels):
                ax.axis("on")
                ax.xaxis.set_label_position("top")
                ax.set_xlabel(label)

            for i, row in enumerate(axes):
                row[0].set_ylabel(tf_names[i])

            im = fig.show()

            def _init():
                return [im]

            def _animate(t):
                for i, row in enumerate(axes):
                    for j, ax in enumerate(row):
                        ax.imshow(
                            data[j, i, :100, t].reshape((10, 10)),
                            aspect="auto",
                            vmin=0,
                            vmax=1,
                        )
                return [im]

            animation = FuncAnimation(
                fig,
                init_func=_init,
                func=_animate,
                frames=num_times,
                interval=20,
                blit=False,
            )
            animation.save("tf_grid_activity.gif", writer="imagemagick")

        def visualise_grid_activity():
            data, _, time_delta, tf_names = get_tf_data()
            num_envs, num_tfs, num_cells, num_times = data.shape
            replicates = 10
            grid_dims = int((num_cells * replicates) ** 0.5)

            model = Examples.models[2]
            _, _, _, model = unpickle("jobs/7324363_models.dat")[0]

            trajectories = OneStepSimulator(
                data, time_delta, replicates=replicates
            ).simulate(model)
            activity_weights = model.activity_weights / np.sum(model.activity_weights)
            activity = np.sum(activity_weights * trajectories, axis=3)

            from matplotlib.animation import FuncAnimation
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(
                1, num_envs, sharex=True, sharey=True, figsize=(2 * num_envs, 2)
            )
            fig.tight_layout()
            plt.subplots_adjust(wspace=0, hspace=0)

            for i, ax in enumerate(axes):
                ax.imshow(np.zeros((grid_dims, grid_dims)))
                plt.setp(ax.get_xticklabels(), visible=False)
                plt.setp(ax.get_yticklabels(), visible=False)
                ax.tick_params(axis="both", which="both", length=0)

            env_labels = ["carbon", "osmotic", "oxidative"]

            for ax, label in zip(axes, env_labels):
                ax.xaxis.set_label_position("top")
                ax.set_xlabel(label)

            im = fig.show()

            def _animate(t):
                for i, ax in enumerate(axes):
                    ax.imshow(
                        activity[t, i, : grid_dims * grid_dims].reshape(
                            (grid_dims, grid_dims)
                        ),
                        aspect="auto",
                        vmin=0,
                        vmax=1,
                    )
                return [im]

            def _init():
                return _animate(0)

            animation = FuncAnimation(
                fig,
                init_func=_init,
                func=_animate,
                frames=num_times,
                interval=20,
                blit=False,
            )
            animation.save("grid_activity.gif", writer="imagemagick")

    class Benchmarking:
        def matrix_exponentials():
            data, _, _, _ = get_tf_data()
            model = Examples.models[2]

            start = time.time()
            mat = model.get_matrix_exp(data, 2.5)
            print(time.time() - start)

        def trajectory():
            data, _, time_delta, _ = get_tf_data()

            states = [i for i in range(2, 20)]
            repeats = 10

            sim = OneStepSimulator(data, tau=time_delta, realised=True, replicates=10)
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

            plt.savefig(f"{Examples.CACHE_FOLDER}/ssa_comparison.png")
            print(res)

        def mi_estimation():
            data, origin, time_delta, _ = get_tf_data()
            model = Examples.models["best_2"]

            interval = OneStepDecodingPipeline.FIXED_INTERVAL
            est = DecodingEstimator(origin, interval, "naive_bayes")
            est.parallel = True

            for replicates in [1, 5, 10, 50, 100]:
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
            import itertools

            data, origin, _, tf_names = get_tf_data()
            dummy_model = PromoterModel.dummy()

            reps = 10
            interval = 30
            classifier = "naive_bayes"
            random_mesh = True

            est = DecodingEstimator(origin, interval, classifier)
            est.parallel = True
            tf_split_data = []
            num_tfs = len(tf_names)

            for tf_index in range(num_tfs):
                TIME_AXIS = 2
                raw_data = np.moveaxis(data[:, tf_index], TIME_AXIS, 0)
                raw_data = raw_data.reshape((*raw_data.shape, 1))
                tf_split_data.append(est._split_classes(dummy_model, raw_data))

            print(f"{classifier}: interval {interval}, {reps} reps")
            print(f"|\033[1m{'TF GROUP':^25}\033[0m|\033[1m{'MI':^25}\033[0m|")
            print(("|" + "-" * 25) * 2 + "|")
            for group_size in range(1, num_tfs + 1):
                for comb in itertools.combinations(list(range(num_tfs)), group_size):
                    comb_split_data = np.concatenate(
                        [tf_split_data[tf] for tf in comb], axis=2
                    )
                    if random_mesh:
                        num_cells = comb_split_data.shape[1]
                        indices = np.random.choice(
                            group_size, (num_cells, interval)
                        ) * interval + np.arange(interval)

                        ## To randomly choose across all cell-samples
                        # num_envs = comb_split_data.shape[0]
                        # env_indices, cell_indices, _ = np.indices((num_envs, num_cells, interval))
                        # comb_split_data = comb_split_data[env_indices, cell_indices, indices]

                        comb_split_data = comb_split_data[:, :, indices]
                    total = 0
                    for _ in range(reps):
                        total += est._estimate(comb_split_data, halving=False)
                    print(
                        f"|{','.join(tf_names[tf] for tf in comb):^25}|{(total/reps):^25.3f}|"
                    )

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
            import os
            from concurrent.futures import ProcessPoolExecutor, as_completed

            if os.path.isfile(fname):
                print("Using cached MI distribution.")
                hist_map = unpickle(fname)
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

        def _pip_evaluate(
            pip: OneStepDecodingPipeline, model: PromoterModel, index: int
        ) -> Tuple[float, int]:
            mi = pip.evaluate(model, verbose=False)
            return index, mi

        def sklearn_nested_parallelism():
            data, origin, time_delta, _ = get_tf_data()
            model = Examples.models[4]

            reps = 20
            n_processors = 5
            iters = 2

            pip = OneStepDecodingPipeline(data, tau=time_delta, replicates=reps)

            # Parallelised n_job=1 tasks
            from concurrent.futures import ProcessPoolExecutor, as_completed

            start = time.time()
            with ProcessPoolExecutor(
                max_workers=min(n_processors, iters),
            ) as executor:
                futures = []
                for i in range(iters):
                    futures.append(
                        executor.submit(
                            Examples.Benchmarking._pip_evaluate,
                            pip,
                            model,
                            i,
                        )
                    )

                for future in as_completed(futures):
                    mi_score, i = future.result()
                    print(f"{reps}-{i}: {mi_score:.3f}")
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
            for i in range(iters):
                futures.append(
                    client.submit(
                        Examples.Benchmarking._pip_evaluate,
                        pip,
                        model,
                        i,
                    )
                )
            res = client.gather(futures)
            print(res)
            print("\n" * 5 + f"Took {time.time() - start:.3f}s" + "\n" * 5)

        def genetic_multiprocessing_overhead():
            data, _, _, _ = get_tf_data()

            states = 4
            population, iterations = 10, 100

            mutations = [
                MutationOperator.edit_edge,
                MutationOperator.add_edge,
                MutationOperator.flip_tf,
                MutationOperator.add_noise,
            ]
            crossover = CrossoverOperator.one_point_triangular_row_swap
            select = SelectionOperator.roulette_wheel
            runner = GeneticRunner(data, mutations, crossover, select)

            start = time.time()
            models, stats = runner.run(
                states=states,
                population=population,
                iterations=iterations,
                model_generator_params={"one_active_state": False},
                verbose=True,
                debug=True,
            )
            print(time.time() - start)

    class Optimisation:
        def grid_search_simple():
            data, _, _, tf_names = get_tf_data()
            gs = GridSearch()
            gs.optimise_simple(data, tf_names)

        def particle_swarm_simple():
            data, _, _, _ = get_tf_data()
            ps = ParticleSwarm()
            ps.optimise_simple(data)
        
        def particle_swarm():
            data, _, _, _ = get_tf_data()
            ps = ParticleSwarm()

            fname = "jobs/7324363_models.dat"
            models = unpickle(fname)
            _, _, _, model = models[0]

            ps.optimise(data, model, start_at_pos=True)

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
            data, _, _, _ = get_tf_data()

            states = 8
            population, iterations = 10, 3
            fname = f"best_models_roulette_new_{states}_{population}_{iterations}.dat"

            mutations = [
                MutationOperator.edit_edge,
                MutationOperator.add_edge,
                MutationOperator.flip_tf,
                MutationOperator.add_noise,
                MutationOperator.flip_activity,
                MutationOperator.add_activity_noise,
            ]
            scale_fitness = ModelPenalty.state_penalty()
            crossover = CrossoverOperator.subgraph_swap
            select = SelectionOperator.tournament
            runner = GeneticRunner(data, mutations, crossover, select, scale_fitness)

            models, stats = runner.run(
                states=states,
                elite_ratio=0.1,
                population=population,
                iterations=iterations,
                n_processors=min(10, population),
                model_generator_params={"one_active_state": False},
                verbose=True,
                debug=True,
            )
            print(stats["non_elite"])

            with open(fname, "wb") as f:
                pickle.dump(models, f)
                print("Cached best models")

        def load_best_models():
            data, _, _, _ = get_tf_data()

            states = 4
            population, iterations = 10, 10
            # fname = f"best_models_roulette_new_{states}_{population}_{iterations}.dat"

            reps = 10
            fname = "jobs/7324363_models.dat"
            models = unpickle(fname)

            pip = OneStepDecodingPipeline(
                data, realised=True, replicates=10, classifier_name="svm"
            )
            pip.set_parallel()

            for i, (_, _, _, model) in enumerate(models[:3]):
                avg_mi = 0
                for _ in range(reps):
                    avg_mi += pip.evaluate(model)
                avg_mi /= reps
                print(f"Model: {model.hash()}, MI: {avg_mi:.3f}")
                # model.visualise(
                #     save=True,
                #     fname=f"cache/evolution/{states}_{population}_{iterations}_{i}.png",
                # )

        def show_best_models():
            import matplotlib.pyplot as plt

            data, _, _, _ = get_tf_data()

            rows, cols = 2, 5
            num_models = rows * cols
            reps = 3

            states = 6
            penalty = False
            penalty_coeff = 0
            population, iterations = 500, 1000
            fname = "jobs/7324363_models.dat"

            scale_fitness = (
                ModelPenalty.state_penalty(penalty_coeff)
                if penalty
                else lambda _, mi: mi
            )

            models = unpickle(fname)

            fig, axes = plt.subplots(
                rows, cols, sharex=True, sharey=True, figsize=(4 * cols, 4 * rows)
            )
            fig.tight_layout()
            fig.suptitle(
                f"GA: {states}-states start, population: {population}, generations: {iterations}"
            )

            pip = OneStepDecodingPipeline(
                data, realised=True, replicates=10, classifier_name="naive_bayes"
            )
            pip.set_parallel()

            plt.subplots_adjust(wspace=0.1, hspace=0.1)

            if rows == 1:
                axes = [axes]

            for i, row in enumerate(axes):
                for (_, _, _, model), ax in zip(models[cols * i : cols * (i + 1)], row):
                    avg_mi = 0
                    for _ in range(reps):
                        avg_mi += pip.evaluate(model)
                    avg_mi /= reps
                    ax.set_xlabel(
                        f"Fitness: {scale_fitness(model, avg_mi):.3f}, MI: {avg_mi:.3f}"
                    )
                    model.visualise(target_ax=ax, transparent=True)
                    ax.set_xticklabels([])
                    ax.set_yticklabels([])
                    ax.set_aspect("equal")

            f_output = f"{Examples.CACHE_FOLDER}/best_{num_models}_{states}_{penalty_coeff}_{population}_{iterations}.png"
            plt.savefig(
                fname=f_output,
                dpi=200,
                bbox_inches="tight",
                pad_inches=0.25,
            )
            print(f"Saved to: {f_output}")

        def examine_evolutionary_run_stats():
            import matplotlib.pyplot as plt

            job_id = "7324363"
            fname = f"jobs/{job_id}_stats_models.dat"

            population = 500
            init_states = 6
            iterations = 1000
            selection = "4-tournament replace"
            penalty = ""

            stats = unpickle(fname)

            include_duration = False
            model_groups = ("elite", "population", "non_elite")
            stat_labels = ("avg_fitness", "avg_mi", "avg_num_states")
            group_colors = ("firebrick", "seagreen", "royalblue")

            fig, axes = plt.subplots(
                len(stat_labels) + int(include_duration), 1, sharex=True
            )
            fig.tight_layout()

            for label, ax in zip(stat_labels, axes):
                for group, color in zip(model_groups, group_colors):
                    ys = stats[group][label]
                    ax.plot(range(len(ys)), ys, label=group, color=color)
                    ax.set_xticks(list(range(1, len(ys), max(1, (len(ys) - 1) // 10))))
                ax.set_ylabel(label)
            plt.xlabel("Number of generations")

            if include_duration:
                duration_states = stats["avg_time_duration"]
                axes[-1].plot(
                    range(len(duration_states)), duration_states, color=group_colors[-1]
                )
                axes[-1].set_ylabel("duration")

            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            fig.legend(
                by_label.values(),
                by_label.keys(),
                loc="lower center",
                ncol=len(model_groups),
                bbox_to_anchor=[0.5, -0.1],
            )

            axes[0].set_title(
                f"{job_id}: init: {population} x {init_states} states x {iterations} iters, {selection}, {penalty}",
                loc="center",
            )

            plt.savefig(
                f"{Examples.CACHE_FOLDER}/{job_id}_evolutionary_run_stats.png",
                dpi=200,
                bbox_inches="tight",
                pad_inches=0.15,
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
                MutationOperator.add_activity_noise,
                MutationOperator.flip_activity,
            ]
            crossover = CrossoverOperator.one_point_triangular_row_swap
            select = SelectionOperator.tournament
            runner = GeneticRunner(data, mutations, crossover, select)

            # Model1 can change hash but model2 must remain the same
            children = runner.crossover(model1, model2, False, True)
            for child in children:
                for _ in range(1000):
                    runner.mutate(child)

            for model in (model1, model2):
                print(model.hash()[:6])

        def models_generated_are_valid():
            # Random models are valid
            for _ in range(100):
                model = ModelGenerator.get_random_model(10, p_edge=0.5)
                ModelGenerator.is_valid(model, verbose=True)

            data, _, _, _ = get_tf_data()
            mutations = [
                MutationOperator.add_noise,
                MutationOperator.add_edge,
                MutationOperator.edit_edge,
                MutationOperator.flip_tf,
            ]
            # crossover = CrossoverOperator.one_point_triangular_row_swap
            crossover = CrossoverOperator.subgraph_swap
            select = SelectionOperator.tournament
            runner = GeneticRunner(data, mutations, crossover, select)

            # Crossover maintains reversibility
            for _ in range(1000):
                models = [
                    ModelGenerator.get_random_model(states=states, p_edge=p_edge)
                    for states, p_edge in zip(
                        2 + np.random.choice(20, size=2), np.random.uniform(size=2)
                    )
                ]
                models = runner.crossover(*models, False, False)
                for model in models:
                    ModelGenerator.is_valid(model, verbose=True)

            # Mutations maintain reversibility
            model = ModelGenerator.get_random_model(10, p_edge=0.1)
            for _ in range(1000):
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
                data, replicates=10, classifier_name="naive_bayes"
            )
            pip.set_parallel()

            dummy_model = PromoterModel.dummy()
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
    # Examples.Benchmarking.sklearn_nested_parallelism()
    # Examples.Benchmarking.genetic_multiprocessing_overhead()
    # Examples.Benchmarking.test_crossover()

    # Examples.PlottingVisuals.visualise_model_example()
    # Examples.PlottingVisuals.visualise_trajectory_example()
    # Examples.PlottingVisuals.visualise_realised_probabilistic_trajectories()
    # Examples.PlottingVisuals.visualise_tf_concentration()
    # Examples.PlottingVisuals.visualise_activity()
    # Examples.PlottingVisuals.visualise_crossover()
    # Examples.PlottingVisuals.visualise_crossover_chart()
    # Examples.PlottingVisuals.visualise_crossbreeding()
    # Examples.PlottingVisuals.visualise_tf_grid_activity()
    # Examples.PlottingVisuals.visualise_grid_activity()

    # Examples.UsingThePipeline.pipeline_example()

    # Examples.Optimisation.grid_search_simple()
    # Examples.Optimisation.particle_swarm_simple()
    Examples.Optimisation.particle_swarm()

    # Examples.Evolution.genetic_simple()
    # Examples.Evolution.model_generation()
    # Examples.Evolution.evolutionary_run()
    # Examples.Evolution.load_best_models()
    # Examples.Evolution.show_best_models()
    # Examples.Evolution.examine_evolutionary_run_stats()
    # Examples.Evolution.crossover_no_side_effects()
    # Examples.Evolution.models_generated_are_valid()
    # Examples.Evolution.test_random_model_variance()
    # Examples.Evolution.test_hypothetical_perfect_model()

    # Examples.Data.find_labels()
    # Examples.Data.load_data()


if __name__ == "__main__":
    main()
