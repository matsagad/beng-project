from typing import Tuple
from nptyping import NDArray, Shape, Float
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
import os.path
import json


def get_tf_data(
    f_data: str = "cache/data_tf.npy",
    f_params: str = "cache/params_tf.json",
    scaled: bool = True,
    folder: str = "data/",
) -> Tuple[NDArray[Shape["Any, Any, Any, Any"], Float], int, int]:

    # Load from cache if exists
    if os.path.isfile(f_data) and os.path.isfile(f_params):
        print("Using cached data!")
        ts = np.load(f_data)
        with open(f_params) as f:
            params = json.load(f)
        return ts, params["origin"], params["tf_names"]

    # Check if primary data sources exist
    NUCLEAR_MARKER_EXPTS = folder + "figS1_nuclear_marker_expts.json"
    STRESS_TYPE_EXPTS = folder + "fig2_stress_type_expts.json"

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
                    times = len(stress_data[tf][env]["general"]["times"][0])

                    # Find largest delta neighborhood around origin
                    min_time_interval = min(
                        min_time_interval, origin, times - origin - 1
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

    np.save(f_data, ts)
    with open(f_params, "w") as f:
        json.dump(params, f)
    print("Cached data successfully!")

    return ts, params["origin"], params["tf_names"]
