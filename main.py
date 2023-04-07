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
    curr_job_id = "7361710"
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
            model = model = PromoterModel([
                [None, RF.Linear([1], [1])],
                [RF.Linear([1], [2]), None],
            ])

            # Simulate
            sim = OneStepSimulator(data, tau=time_delta, realised=True)
            trajectories = sim.simulate(model)
            print(trajectories.shape)

            # Visualise single cell
            print("Single-cell time series")
            OneStepSimulator.visualise_trajectory(
                trajectories, model=model, average=False, batch_num=1, fname="single_traj.png"
            )

            # Visualise average
            print("Average-cell time series")
            OneStepSimulator.visualise_trajectory(
                trajectories, model=model, average=True, fname="avg_traj.png"
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

            # model = ModelGenerator.get_random_model(5, one_active_state=False)

            # fname = f"jobs/{Examples.curr_job_id}_models.dat"

            # models = unpickle(fname)
            # _, _, _, model = models[0]
            model = model = PromoterModel([
                [None, RF.Linear([2], [1])],
                [RF.Linear([1], [2]), None],
            ])


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
            fname = f"jobs/{Examples.curr_job_id}_models.dat"
            _, _, _, model = unpickle(fname)[0]

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

        def visualise_trajectory_weaving():
            data, _, _, _ = get_tf_data()
            model = PromoterModel([
                [None, RF.Linear([1], [1])],
                [RF.Linear([1], [2]), None],
            ])

    class Optimisation:
        def grid_search_simple():
            data, _, _, tf_names = get_tf_data()
            gs = GridSearch()
            gs.optimise_simple(data, tf_names)

        def particle_swarm_simple():
            data, _, _, _ = get_tf_data()
            ps = ParticleSwarm()
            ps._optimise_simple(data)

        def particle_swarm():
            data, _, _, _ = get_tf_data()
            ps = ParticleSwarm()

            fname = f"jobs/{Examples.curr_job_id}_models.dat"
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
            fname = f"jobs/{Examples.curr_job_id}_models.dat"
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

            rows, cols = 2, 2
            num_models = rows * cols
            reps = 1

            states = 6
            penalty = False
            penalty_coeff = 0
            population, iterations = 2000, 2000
            fname = f"jobs/{Examples.curr_job_id}_models.dat"

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

            plt.subplots_adjust(wspace=0.05, hspace=0.05)

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
                    model.visualise(with_rates=False, target_ax=ax, transparent=False)
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

            job_id = Examples.curr_job_id
            fname = f"jobs/{job_id}_stats_models.dat"

            population = 500
            init_states = 6
            iterations = 1000
            selection = "roulette noreplace"
            penalty = "6-target-8"

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

        def pickle_best_model():
            job_id = "7324363"
            fname = f"jobs/{job_id}_models.dat"
            models = unpickle(fname)

            with open("best_model.dat", "wb") as f:
                pickle.dump(models[0][3], f)


def main():
    # from examples.benchmarking import BenchmarkingExamples
    # bm_examples = BenchmarkingExamples()
    # bm_examples.matrix_exponentials()
    # bm_examples.trajectory_simulation()
    # bm_examples.mi_estimation()
    # bm_examples.mi_estimation_table()
    # bm_examples.sklearn_nested_parallelism()
    # bm_examples.genetic_multiprocessing_overhead()

    # from examples.mi_trends import MITrendsExamples
    # mi_examples = MITrendsExamples()
    # mi_examples.mi_vs_interval()
    # mi_examples.mi_distribution()
    # mi_examples.max_mi_estimation()
    # mi_examples.mi_vs_repeated_intervals()

    
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
    # Examples.Optimisation.particle_swarm()

    # Examples.Evolution.genetic_simple()
    # Examples.Evolution.model_generation()
    # Examples.Evolution.evolutionary_run()
    # Examples.Evolution.load_best_models()
    # Examples.Evolution.show_best_models()
    # Examples.Evolution.examine_evolutionary_run_stats()
    # Examples.Evolution.evaluate_tf_presence_in_models()
    # Examples.Evolution.test_random_model_variance()

    # Examples.Data.find_labels()
    # Examples.Data.load_data()
    # Examples.Data.pickle_best_model()


if __name__ == "__main__":
    main()
