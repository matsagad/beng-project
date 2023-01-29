import json
from models.simple import SimpleModel
from models.model import PromoterModel
from models.rates.function import RateFunction
from utils.data_processing import scaleTSall
from utils.MIdecodingSimp import estimateMIS
from ssa.one_step import OneStepSimulator
from utils.data_processing import scaleTS
import time
import numpy as np


def import_gluc_data(fname="gluc_data_all.npy", save=True):
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
        full_data = np.array([ts[:min_count] for ts in full_data])
        if save:
            np.save(fname, full_data)
        return full_data


def import_data(fname="data_all.npy", save=True):
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
        full_data = np.array([[stress_test[:min_count] for stress_test in ts] for ts in full_data])
        if save:
            np.save(fname, full_data)
        return full_data

def test_gluc():
    # Load data
    _, origin, _ = scaleTS("dot6")

    ## Batched data (~119 replicates)
    data = import_gluc_data()
    print(data.shape)  # num tfs, replicates, time stamps
    single_data = np.array(data[:, :1])

    # Set-up model
    tf_index = 1
    a, b, c = 1.0e-3, 1.0e-2, 1.0e-2

    model = PromoterModel(
        rate_fn_matrix=[
            [None, None, RateFunction.linear(a, tf_index)],
            [RateFunction.constant(b), None, None],
            [None, RateFunction.constant(c), None],
        ]
    )

    # Simulate
    start = time.time()
    sim = OneStepSimulator(data, 2.5, deterministic=False)
    ts = sim.simulate(model)
    print(ts.shape)
    print(time.time() - start)

    active_state = -1
    rich = ts[origin - 20 : origin, :, active_state].T
    stress = ts[origin : origin + 20, :, active_state].T
    print(f"rich: {np.sum(rich, axis=1)}")
    print(f"stress: {np.sum(stress, axis=1)}")

    # Evaluate
    print("calculating")
    miscore = estimateMIS([[rich, stress]])[0]
    print(f"MI: {miscore}")


if __name__ == "__main__":
    test_gluc()
