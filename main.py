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


def import_data(fname="cache/data_all.npy", save=True):
    try:
        return np.load(fname)
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
        return full_data


class Examples:
    tf_index = 0
    a, b, c = 1.0e-2, 1.0e-2, 1.0e-2
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
            start = time.time()

            sim = OneStepSimulator(data, tau=2.5, realised=False)
            trajectories = sim.simulate(model)

            print(time.time() - start)

        def mi_estimation():
            data = import_data()
            model = Examples.models[2]

            # Simulate
            sim = OneStepSimulator(data, tau=2.5, realised=False, replicates=1)
            trajectories = sim.simulate(model)

            # Estimate MI
            origin = OneStepDecodingPipeline.FIXED_ORIGIN
            interval = OneStepDecodingPipeline.FIXED_INTERVAL
            est = DecodingEstimator(origin, interval)

            print("Estimating MI...")
            start = time.time()
            mi_score = est.estimate(model, trajectories)
            print(time.time() - start)
            print(f"MI: {mi_score}")

    class Optimisation:
        def grid_search():
            data = import_data()
            gs = GridSearch()
            gs.optimise(data)


def main():
    # Examples.Benchmarking.trajectory()
    Examples.Benchmarking.mi_estimation()
    # Examples.PlottingVisuals.visualise_model_example()
    # Examples.PlottingVisuals.visualise_trajectory_example()
    # Examples.PlottingVisuals.visualise_realised_probabilistic_trajectories()
    # Examples.UsingThePipeline.pipeline_example()
    # Examples.Optimisation.grid_search()


if __name__ == "__main__":
    main()
