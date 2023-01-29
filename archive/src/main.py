import json
from models.simple import SimpleModel
from utils.extrande import Extrande
from utils.data_processing import *
import matplotlib.pyplot as plt
import time

def main():
    """
    TODO: either figure out why B bound is too large making times longer
    or allow batched extrande runs or integrate MI estimation
    """
    model = SimpleModel.simple_with_rates([1.0e-02, 1.0e-02, 1.0e-01, 1.0e-02])
    time_series, origin, time_stamps = scaleTS("dot6")
    cell_num = 0
    print("start sim")
    start = time.time()
    sim_times, sim_states = Extrande.simulate(
        model=model,
        max_time=time_stamps[cell_num][-1],
        initial_state=np.array([0, 1, 0, 0]),
        exogenous_times=time_stamps[cell_num],
        exogenous_states=np.array([time_series[cell_num]]),
        scaled=True
    )
    print("before:")
    print(sim_states[:origin])
    print("after:")
    print(sim_states[origin:])
    print(time.time() - start)

    return sim_times, sim_states

if __name__ == "__main__":
    main()