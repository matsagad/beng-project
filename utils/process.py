from typing import Tuple
from nptyping import NDArray, Shape, Float
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
import os.path
import json


def _min_max_scale(data, local: bool = False):
    """
    Helper function for min-max scaling multidimensional data.
    """
    if local:
        min_trace = np.expand_dims(np.min(data, axis=-1), axis=-1)
        max_trace = np.expand_dims(np.max(data, axis=-1), axis=-1)
    else:
        min_trace, max_trace = np.min(data), np.max(data)

    return (data - min_trace) / (max_trace - min_trace)


def _max_scale(data, local: bool = True):
    """
    Helper function for max scaling multidimensional data.
    """
    max_trace = (
        np.expand_dims(np.max(data, axis=-1), axis=-1) if local else np.max(data)
    )
    return data / max_trace


def get_tf_data(
    f_data: str = "data_tf.npy",
    f_params: str = "params_tf.json",
    scale: bool = True,
    local_scale: bool = True,
    cache_folder: str = "cache",
    data_folder: str = "data",
) -> Tuple[NDArray[Shape["Any, Any, Any, Any"], Float], int, int]:
    """
    Yields data from the stress type experiments.
    Returns a 4D array with dimensions: # environments, # TFs, batch size, # times.
    """
    # Load from cache if exists
    path_to_data = f"{cache_folder}/{f_data}"
    path_to_params = f"{cache_folder}/{f_params}"

    scale_fn = lambda x: _min_max_scale(x, local=local_scale)

    if os.path.isfile(path_to_data) and os.path.isfile(path_to_params):
        print("Using cached data!")

        ts = np.load(path_to_data)
        ts = scale_fn(ts) if scale else ts

        with open(path_to_params) as f:
            params = json.load(f)

        return ts, params["origin"], params["avg_time_delta"], params["tf_names"]

    # Check if primary data sources exist
    NUCLEAR_MARKER_EXPTS = f"{data_folder}/figS1_nuclear_marker_expts.json"
    STRESS_TYPE_EXPTS = f"{data_folder}/fig2_stress_type_expts.json"

    for experiment in (NUCLEAR_MARKER_EXPTS, STRESS_TYPE_EXPTS):
        if not os.path.isfile(experiment):
            raise Exception(f"Data for experiment not found: {experiment}.")

    # Find inhomogeneity in fluorescence from nuclear markers
    tf_names = []
    tf_ts = []
    params = dict()
    kernel = DotProduct() + WhiteKernel()

    with open(NUCLEAR_MARKER_EXPTS, "r") as f_nuc:
        with open(STRESS_TYPE_EXPTS, "r") as f_stress:
            nm_data = json.load(f_nuc)
            stress_data = json.load(f_stress)
            min_batch_size = float("inf")
            min_time_interval = float("inf")
            avg_time_delta = 0

            # Consider all TFs with nuclear markers
            for tf in nm_data.keys():
                gfp = nm_data[tf]["GFP"]

                metrics = [
                    np.array(gfp[label]).astype("float")
                    for label in ["max5", "nucInnerMedian", "median", "imBackground"]
                ]
                invalid = np.logical_or.reduce(list(map(np.isnan, metrics))).any(axis=1)
                max5, inner_median, median, bg = [arr[~invalid] for arr in metrics]

                # Avoid division by zero warnings
                nuc_loc = np.mean(
                    np.divide(
                        (max5 - bg),
                        (median - bg),
                        where=(median - bg != 0),
                    ),
                    axis=0,
                )
                marker_intensity = np.mean(inner_median - bg, axis=0)

                # Fit Gaussian Process
                X_train = np.expand_dims(nuc_loc, -1)
                y_train = np.expand_dims(marker_intensity, -1)

                input_scaler = StandardScaler().fit(X_train)
                output_scaler = MinMaxScaler().fit(y_train)

                X_scaled = input_scaler.transform(X_train)
                y_scaled = output_scaler.transform(y_train)

                reg = GaussianProcessRegressor(kernel=kernel, random_state=0).fit(
                    X_scaled, y_scaled
                )

                tf_nuc_ts = []

                for env in sorted(stress_data[tf].keys()):
                    origin = stress_data[tf][env]["general"]["origin"]
                    times = stress_data[tf][env]["general"]["times"]
                    num_times = len(times[0])

                    # Find largest delta neighborhood around origin
                    min_time_interval = min(
                        min_time_interval, origin, num_times - origin - 1
                    )

                    # Find average time delta
                    avg_time_delta += np.mean(
                        np.array(times)[:, 1:] - np.array(times)[:, :-1]
                    )

                    # Load and predict data using regressor
                    st_gfp = stress_data[tf][env]["GFP"]
                    st_max5, st_median, st_bg = [
                        np.array(st_gfp[label]).astype("float")
                        for label in ["max5", "median", "imBackground"]
                    ]

                    # Avoid division by zero warnings and NaNs
                    st_nuc_loc = np.divide(
                        (st_max5 - st_bg),
                        (st_median - st_bg),
                        where=(st_median - st_bg != 0),
                    )
                    st_nuc_loc = st_nuc_loc[np.isfinite(st_nuc_loc).all(axis=1)]

                    min_batch_size = min(min_batch_size, len(st_nuc_loc))
                    st_nuc_loc = st_nuc_loc[
                        :min_batch_size,
                        origin - min_time_interval : origin + min_time_interval + 1,
                    ]

                    X = input_scaler.transform(st_nuc_loc.reshape((-1, 1)))
                    y_unscaled = reg.predict(X).reshape(st_nuc_loc.shape)
                    y = output_scaler.inverse_transform(y_unscaled)

                    tf_nuc_ts.append(y)

                tf_names.append(tf)
                ind_tf_ts = []
                for tf_env_ts in tf_nuc_ts:
                    mid = (tf_env_ts.shape[1] - 1) // 2
                    ind_tf_ts.append(
                        tf_env_ts[
                            :min_batch_size,
                            mid - min_time_interval : mid + min_time_interval + 1,
                        ]
                    )
                tf_ts.append(np.array(ind_tf_ts))

    params["tf_names"] = tf_names
    params["origin"] = min_time_interval
    params["avg_time_delta"] = avg_time_delta / (len(tf_names) * len(tf_ts[0]))

    ts = []
    for tf_nuc_ts in tf_ts:
        mid = (tf_nuc_ts.shape[2] - 1) // 2
        ts.append(
            tf_nuc_ts[
                :,
                :min_batch_size,
                mid - min_time_interval : mid + min_time_interval + 1,
            ]
        )

    # Set leading dimension to be environmental identity
    ts = np.moveaxis(np.array(ts), 1, 0)

    np.save(path_to_data, ts)
    with open(path_to_params, "w") as f:
        json.dump(params, f)
    print("Cached data successfully!")

    ts = scale_fn(ts) if scale else ts

    return ts, params["origin"], params["avg_time_delta"], params["tf_names"]
