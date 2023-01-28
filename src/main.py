from models.simple import SimpleModel
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
        np.save(fname, full_data)
        return full_data


def main():
    # Batched data (~119 replicates)
    data = import_gluc_data()
    print(data.shape)  # num tfs, replicates, time stamps

    # single_data = np.array(data[:, :1])
    # print(single_data.shape)

    model = SimpleModel(0, 1.0e0, 1.0e0)

    # Simulate
    start = time.time()
    sim = OneStepSimulator(data, 2.5)
    ts = sim.simulate(model)
    print(time.time() - start)

    print(ts)


if __name__ == "__main__":
    main()
