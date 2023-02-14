from typing import Tuple
from nptyping import NDArray, Shape, Float
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
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
        ts = np.load(f_data)
        params = json.load(f_params)
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

            # Consider all TFs with nuclear markers
            for tf in ["sfp1"]:  # nm_data.keys():
                gfp = nm_data[tf]["GFP"]

                max5, inner_median, median, bg = [
                    np.array(gfp[label]).astype("float")
                    for label in ["max5", "nucInnerMedian", "median", "imBackground"]
                ]

                nuc_loc = (max5 - bg) / (median - bg)
                marker_intensity = inner_median - bg

                invalid = np.logical_or(np.isnan(nuc_loc), np.isnan(marker_intensity)).any(axis=0)

                nuc_loc = np.mean(nuc_loc[:, ~invalid], axis=0)
                marker_intensity = np.mean(marker_intensity[:, ~invalid], axis=0)

                # Fit Gaussian Process
                X, y = np.expand_dims(marker_intensity, -1), nuc_loc
                reg = GaussianProcessRegressor(kernel=kernel, random_state=0).fit(X, y)
                
                tf_nuc_ts = []
                for env in sorted(stress_data[tf].keys()):
                    st_gfp = stress_data[tf][env]["GFP"]
                    st_max5, st_median, st_bg = [
                        np.array(st_gfp[label]).astype("float")
                        for label in ["max5", "median", "imBackground"]
                    ]
                    st_nuc_loc = (st_max5 - st_bg) / (st_median - st_bg)
                    st_nuc_loc = np.expand_dims(np.mean(st_nuc_loc[:, ~np.isnan(st_nuc_loc).any(axis=0)], axis=0), -1)
                    tf_nuc_ts.append(reg.predict(st_nuc_loc))
                    print(stress_data[tf][env]["general"]["origin"])

                tf_names.append(tf)
                # TODO: handle case when different number of batches
                tf_ts.append(np.array(tf_nuc_ts))

    ts = np.array(ts)
    params["tf_names"] = tf_names

    np.save(f_data, ts)
    np.save(f_params, params)

    return ts, params["origin"], params["tf_names"]

