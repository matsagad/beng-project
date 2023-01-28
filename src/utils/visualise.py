import json
import matplotlib.pyplot as plt

def plot_heatmap(data, cmap="plasma"):
    plt.imshow(data, cmap=cmap, interpolation="nearest")
    plt.colorbar()
    plt.show()


def data_visualisation(tf_name="dot6"):
    time_series, origin, sampleTimes = scaleTS(tf_name)
    plot_heatmap(time_series)

with open("data/figS1_nuclear_marker_expts.json", "r") as read_file:
    ncdata = json.load(read_file)
    plot_heatmap(ncdata["dot6"]["GFP"]["times"])


