from concurrent.futures import ProcessPoolExecutor, as_completed
from models.rates.function import RateFunction as RF
from models.model import PromoterModel
from multiprocessing import get_context
from nptyping import NDArray, Shape, Float
from pipeline.one_step_decoding import OneStepDecodingPipeline
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import os.path
import time


class GridSearch:
    def _simple_grid_search(tf: int, params: Dict[str, any]) -> Tuple[int, NDArray]:
        pip = params["pipeline"]
        on_count, off_count, low_bound, up_bound = params["bounds"]

        print(f"Starting grid search for TF{tf}")
        res = np.zeros((on_count, off_count))

        for i, k_on in enumerate(np.logspace(low_bound, up_bound, num=on_count)):
            for j, k_off in enumerate(np.logspace(low_bound, up_bound, num=off_count)):
                model = PromoterModel(
                    rate_fn_matrix=[
                        [None, RF.Linear([k_on], [tf])],
                        [RF.Constant([k_off]), None],
                    ]
                )
                # print(f"TF{tf}-{'%.2f' % k_on}-{'%.2f' % k_off}")
                res[i, j] = pip.evaluate(model)
                print(
                    f"TF{tf}-{'%.2f' % (100 * (i * off_count + j + 1) / (on_count * off_count))}%"
                )

        return tf, res

    def optimise_simple(
        self,
        exogenous_data: NDArray[Shape["Any, Any, Any, Any"], Float],
        tf_names: List[str],
        grid_size: Tuple[int, int] = (10, 10),
        log_bounds: Tuple[int, int] = (-2, 2),
        classifier: str = "svm",
        single_env: bool = False,
        replicates: int = 1,
        fname: str = "optimise_simple.png",
        cache_folder: str = "cache",
    ) -> None:
        on_count, off_count = grid_size
        low_bound, up_bound = log_bounds
        tfs_to_test = tf_names

        data = exogenous_data[:1] if single_env else exogenous_data
        fname_temp = (
            f"{cache_folder}/{fname.split('/')[-1].split('.')[0]}__"
            + f"tf{{tf_index}}_{classifier}_reps{replicates}_"
            + f"{low_bound}-{up_bound}_{on_count}_{off_count}.npy"
        )
        pip = OneStepDecodingPipeline(
            data,
            realised=True,
            replicates=replicates,
            classifier_name=classifier,
        )

        num_tfs = len(tfs_to_test)
        tfs = [tf_names.index(tf) for tf in tfs_to_test]
        res = np.zeros((num_tfs, on_count, off_count))

        params = {
            "bounds": (on_count, off_count, low_bound, up_bound),
            "pipeline": pip,
        }

        start = time.time()
        print("0.00%")

        if num_tfs == 1:
            # Don't spawn processes that interfere with sklearn's joblib parallelism
            tf = tfs[0]
            cache_fname = fname_temp.format(tf_index=tf)
            if os.path.isfile(cache_fname):
                print(f"Using cached data for TF{tf}")
                res[tfs.index(tf)] = np.load(cache_fname)
            else:
                pip.set_parallel()
                tf, data = GridSearch._simple_grid_search(tfs[0], params)
                np.save(fname_temp.format(tf_index=tf), data)
                print(f"Cached TF{tf} data")
                res[tfs.index(tf)] = data
        else:
            with ProcessPoolExecutor(
                max_workers=num_tfs, mp_context=get_context("spawn")
            ) as executor:
                futures = []
                for i, tf in enumerate(tfs):
                    cache_fname = fname_temp.format(tf_index=tf)
                    if os.path.isfile(cache_fname):
                        print(f"Using cached data for TF{tf}")
                        res[tfs.index(tf)] = np.load(cache_fname)
                        continue
                    futures.append(
                        executor.submit(GridSearch._simple_grid_search, tf, params)
                    )

                for future in as_completed(futures):
                    tf, data = future.result()
                    np.save(fname_temp.format(tf_index=tf), data)
                    print(f"Cached TF{tf} data")
                    res[tfs.index(tf)] = data

        print(f"{time.time() - start}s elapsed")

        fig, axes = plt.subplots(1, num_tfs, sharey=True)

        if num_tfs == 1:
            axes = [axes]

        for i, pair in enumerate(zip(axes, res)):
            ax, res = pair
            ax.title.set_text(tf_names[tfs[i]])
            im = ax.imshow(
                res[::-1],  # reverse to move index 0 to bottom
                cmap="rainbow",
                extent=(low_bound, up_bound, low_bound, up_bound),
                aspect="equal",
                interpolation="none",
                vmin=0,
                vmax=2,
            )

        fig.colorbar(im, ax=axes, location="bottom")

        plt.savefig(fname, dpi=100)
