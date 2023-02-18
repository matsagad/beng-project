from concurrent.futures import ThreadPoolExecutor
from mi_estimation.decoding import DecodingEstimator
from models.model import PromoterModel
from models.rates.function import RateFunction as RF
from optimisation.grid_search import GridSearch
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
    models = {
        2: PromoterModel(
            rate_fn_matrix=[[None, RF.Linear(a, tf_index)], [RF.Constant(b), None]]
        ).with_active_states([1]),
        3: PromoterModel(
            rate_fn_matrix=[
                [None, None, RF.Constant(a)],
                [None, None, RF.Constant(b)],
                [
                    RF.Linear(a, tf_index),
                    RF.Linear(b, tf_index + 1),
                    None,
                ],
            ]
        ).with_active_states([0, 1]),
        4: PromoterModel(
            rate_fn_matrix=[
                [None, None, None, RF.Constant(a)],
                [None, None, None, RF.Constant(b)],
                [None, None, None, RF.Constant(c)],
                [
                    RF.Linear(a, tf_index),
                    RF.Linear(b, tf_index + 1),
                    RF.Linear(c, tf_index + 2),
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
                    [None, None, None, RF.Constant(a)],
                    [None, None, None, RF.Constant(b)],
                    [None, None, None, RF.Constant(c)],
                    [
                        RF.Linear(a, tf_index),
                        RF.Linear(b, tf_index + 1),
                        RF.Linear(c, tf_index + 2),
                        None,
                    ],
                ]
            ).with_active_states([0, 1, 2])

            # Visualise
            model.visualise()

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

            model = Examples.models[2]
            replicates = 10
            interval = OneStepDecodingPipeline.FIXED_INTERVAL
            est = DecodingEstimator(origin, interval, "naive_bayes")

            # Raw Nuclear Translocation Trajectory
            TIME_AXIS = 2
            raw_data = np.moveaxis(data[:, 0], TIME_AXIS, 0)
            raw_data = raw_data.reshape((*raw_data.shape, 1))
            split_raw = est._split_classes(PromoterModel([[RF.Constant(1)]]), raw_data)

            # Simulated Trajectory
            sim = OneStepSimulator(
                data, tau=time_delta, realised=True, replicates=replicates
            )
            trajectories = sim.simulate(model)
            split_sim = est._split_classes(model, trajectories)

            split = split_sim

            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, len(split), sharey=True)

            for _, pair in enumerate(zip(axes, split)):
                ax, res = pair
                im = ax.imshow(
                    res,  # reverse to move index 0 to bottom
                    cmap="rainbow",
                    aspect="auto",
                    interpolation="none",
                    vmin=0,
                    vmax=1,
                )

            fig.colorbar(im, ax=axes, location="bottom")

            plt.savefig(f"{Examples.CACHE_FOLDER}/binary_vis.png", dpi=100)

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

            for replicates in [1, 2, 5, 10]:
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
            dummy_model = PromoterModel([RF.Constant(1)])

            est = DecodingEstimator(origin, interval, "naive_bayes")
            for tf_index, tf in enumerate(tf_names):
                for replicates in [1]:
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

    class Optimisation:
        def grid_search():
            data, _, _, tf_names = get_tf_data()
            gs = GridSearch()
            gs.optimise_simple(data, tf_names)

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
    Examples.Benchmarking.max_mi_estimation()
    # Examples.Benchmarking.mi_estimation_table()
    # Examples.Benchmarking.mi_vs_interval()
    # Examples.PlottingVisuals.visualise_model_example()
    # Examples.PlottingVisuals.visualise_trajectory_example()
    # Examples.PlottingVisuals.visualise_realised_probabilistic_trajectories()
    # Examples.PlottingVisuals.visualise_activity()
    # Examples.UsingThePipeline.pipeline_example()
    # Examples.Optimisation.grid_search()
    # Examples.Data.find_labels()
    # Examples.Data.load_data()


if __name__ == "__main__":
    main()
