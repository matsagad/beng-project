import json
from models.simple import SimpleModel
from utils.extrande import Extrande
from utils.data_processing import *


def main():
    model = SimpleModel.simple_with_rates([1.0e-02, 1.0e-02, 1.0e-01, 1.0e-02])
    time_series, origin, time_stamps = scaleTS("dot6")
    cell_num = 0

    sim_time_series = Extrande.simulate(
        model=model,
        max_time=time_stamps[cell_num][-1],
        initial_state=np.array([0, 10, 0, 0]),
        exogenous_times=time_stamps[cell_num],
        exogenous_states=np.array([time_series[cell_num]]),
    )
    print("before:")
    print(sim_time_series[:origin])
    print("after:")
    print(sim_time_series[origin:])
