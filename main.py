import json
from data.process import get_tf_data
from models.model import PromoterModel
from models.rates.function import RateFunction as RF
from optimisation.grid_search import GridSearch
from pipeline.one_step_decoding import OneStepDecodingPipeline
from ssa.one_step import OneStepSimulator
from mi_estimation.decoding import DecodingEstimator
from utils.data_processing import scaleTSall
from utils.data_processing import scaleTS
import numpy as np
import time


def _normalise(data):
    min_trace = np.expand_dims(np.min(data, axis=-1), axis=-1)
    max_trace = np.expand_dims(np.max(data, axis=-1), axis=-1)
    return (data - min_trace) / (max_trace - min_trace)


def import_gluc_data(fname="cache/gluc_data_all.npy", save=True):
    try:
        return np.load(fname)
    except:
        full_data = []
        for tf in ("msn2", "sfp1", "dot6", "maf1", "mig1", "hog1", "yap1"):
            try:
                ts, _, _ = scaleTS(tf)
                full_data.append(ts)
            except:
                print(f"{tf} not in ncdata")

        min_count = min(ts.shape[0] for ts in full_data)
        full_data = np.array([[ts[:min_count] for ts in full_data]])
        if save:
            np.save(fname, full_data)
        return full_data


def import_data(fname="cache/data_all.npy", save=True, normalise=True):
    try:
        full_data = np.load(fname)
    except:
        full_data = []
        for tf in ("msn2", "sfp1", "dot6", "maf1", "mig1", "hog1", "yap1"):
            try:
                ts, _, _ = scaleTSall(tf)
                full_data.append(ts)
            except:
                print(f"{tf} not in ncdata")

        min_count = min(len(stress_test) for ts in full_data for stress_test in ts)
        full_data = np.moveaxis(
            np.array(
                [[stress_test[:min_count] for stress_test in ts] for ts in full_data]
            ),
            0,
            1,
        )

        if save:
            np.save(fname, full_data)

    if not normalise:
        return full_data

    return _normalise(full_data)


class Examples:
    tf_index = 0
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
            data = import_data()
            print(data.shape)  # num envs, num tfs, replicates, time stamps

            ## Examples for debugging batching process
            single_replicate = np.array(data[:, :, :1])
            single_environment = np.array(data[:1])

            # Set-up model
            model = Examples.models[3]

            # Simulate and evaluate
            pipeline = OneStepDecodingPipeline(data)
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
            data = import_data()
            model = Examples.models[4]

            # Simulate
            sim = OneStepSimulator(data, tau=2.5, realised=True)
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
            data = import_data()
            model = Examples.models[2]

            print("Realised (initial: [0.5, 0.5])")
            OneStepSimulator.visualise_trajectory(
                OneStepSimulator(data, tau=2.5, realised=True, replicates=3).simulate(
                    model
                ),
                model=model,
            )

            print("Probabilistic (initial: [0.5, 0.5])")
            OneStepSimulator.visualise_trajectory(
                OneStepSimulator(data, tau=2.5, realised=False).simulate(model),
                model=model,
            )

        def visualise_activity():
            data = import_data()

            model = Examples.models[2]
            replicates = 10
            origin = OneStepDecodingPipeline.FIXED_ORIGIN
            interval = OneStepDecodingPipeline.FIXED_INTERVAL
            est = DecodingEstimator(origin, interval, "naive_bayes")

            # Raw Nuclear Translocation Trajectory
            TIME_AXIS = 2
            raw_data = np.moveaxis(data[:, 0], TIME_AXIS, 0)
            raw_data = raw_data.reshape((*raw_data.shape, 1))
            split_raw = est._split_classes(PromoterModel([[RF.Constant(1)]]), raw_data)

            # Simulated Trajectory
            sim = OneStepSimulator(data, tau=2.5, realised=True, replicates=replicates)
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

            plt.savefig(f"cache/updated/binary_vis.png", dpi=100)

    class Benchmarking:
        def matrix_exponentials():
            data = import_data()
            model = Examples.models[2]

            start = time.time()
            mat = model.get_matrix_exp(data, 2.5)
            print(time.time() - start)

        def trajectory():
            data = import_data()
            model = Examples.models[2]

            # Simulate
            for replicates in [1, 1, 5, 10, 50]:
                repeats, total = 10, 0
                for _ in range(repeats):
                    start = time.time()

                    sim = OneStepSimulator(
                        data, tau=2.5, realised=True, replicates=replicates
                    )
                    trajectories = sim.simulate(model)

                    total += time.time() - start

                print(f"{replicates} replicates: {total / repeats}")

        def mi_estimation():
            data = import_data()[:1]
            model = Examples.models[2]
            origin = OneStepDecodingPipeline.FIXED_ORIGIN
            interval = OneStepDecodingPipeline.FIXED_INTERVAL
            est = DecodingEstimator(origin, interval, "naive_bayes")

            for replicates in [1, 5, 10, 50]:
                # Simulate
                sim = OneStepSimulator(
                    data, tau=2.5, realised=True, replicates=replicates
                )
                trajectories = sim.simulate(model)

                # Estimate MI
                print("Estimating MI...")
                start = time.time()
                mi_score = est.estimate(model, trajectories)
                print(f"{replicates} replicates: {time.time() - start}")

                print(f"MI: {mi_score}")

        def max_mi_estimation():
            origin = OneStepDecodingPipeline.FIXED_ORIGIN
            interval = OneStepDecodingPipeline.FIXED_INTERVAL
            dummy_model = PromoterModel([RF.Constant(1)])
            tfs = ["msn2", "sfp1", "dot6", "maf1"]

            est = DecodingEstimator(origin, interval, "random_forest")
            for tf_index, tf in enumerate(tfs):
                for replicates in [2, 5, 10, 20, 30]:
                    TIME_AXIS = 2
                    raw_data = np.moveaxis(import_data()[:, tf_index], TIME_AXIS, 0)
                    raw_data = raw_data.reshape((*raw_data.shape, 1))
                    # To test repeated samples are handled
                    rep_raw_data = np.tile(raw_data, (replicates, 1))

                    # Estimate MI
                    data = est._split_classes(dummy_model, raw_data)

                    start = time.time()
                    mi_score = est._estimate(data, add_noise=False)
                    print(time.time() - start)

                    print(f"{tf} {replicates} MI: {mi_score}")

    class Optimisation:
        def grid_search():
            data = import_data()
            gs = GridSearch()
            gs.optimise_simple(data)

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
            print(get_tf_data().shape)

def main():
    # Examples.Benchmarking.trajectory()
    # Examples.Benchmarking.mi_estimation()
    # Examples.Benchmarking.max_mi_estimation()
    # Examples.PlottingVisuals.visualise_model_example()
    # Examples.PlottingVisuals.visualise_trajectory_example()
    # Examples.PlottingVisuals.visualise_realised_probabilistic_trajectories()
    # Examples.PlottingVisuals.visualise_activity()
    # Examples.UsingThePipeline.pipeline_example()
    # Examples.Optimisation.grid_search()
    # Examples.Data.find_labels()
    Examples.Data.load_data()


if __name__ == "__main__":
    main()
