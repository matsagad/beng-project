from models.model import PromoterModel
from models.rates.function import RateFunction as RF
from pipeline.one_step_decoding import OneStepDecodingPipeline
from ssa.one_step import OneStepSimulator
from utils.data_processing import scaleTSall
from utils.data_processing import scaleTS
import numpy as np


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
            tf_index = 0
            a, b, c = 1.0e-3, 1.0e-2, 1.0e-2

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

            # Simulate and evaluate
            pipeline = OneStepDecodingPipeline(data)
            pipeline.evaluate(model, verbose=True)

    class PlottingVisuals:
        def visualise_model_example():
            # Set-up model
            tf_index = 0
            a, b, c = 1.0e-3, 1.0e-2, 1.0e-2

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
            # Import data
            data = import_data()

            # Set-up model
            tf_index = 0
            a, b, c = 1.0e-3, 1.0e-2, 1.0e-2
    
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

            # Simulate
            sim = OneStepSimulator(data, tau=2.5)
            trajectories = sim.simulate(model)
            print(trajectories.shape)

            # Visualise average
            OneStepSimulator.visualise_trajectory(trajectories, average=True)

            # Visualise single cell
            OneStepSimulator.visualise_trajectory(
                trajectories, average=False, batch_num=1
            )


def main():
    Examples.PlottingVisuals.visualise_trajectory_example()


if __name__ == "__main__":
    main()
