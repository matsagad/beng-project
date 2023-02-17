from concurrent.futures import ProcessPoolExecutor, as_completed
from models.rates.function import RateFunction as RF
from models.model import PromoterModel
from nptyping import NDArray, Shape, Float
from pipeline.one_step_decoding import OneStepDecodingPipeline
from typing import Dict, List
import matplotlib.pyplot as plt
import numpy as np
import os.path
import time


class GridSearch:
    def _simple_grid_search(tf: int, params: Dict[str, any]) -> Tuple[int, NDArray]:
        fname_temp = params["template"]
        pip = params["pipeline"]
        on_count, off_count, low_bound, up_bound = params["bounds"]

        fname = fname_temp.format(tf_index=tf)
        print(f"Starting grid search for TF{tf}")
        if os.path.isfile(fname):
            res = np.load(fname)
            print(f"Used cached TF{tf} data")
            return tf, res

        res = np.zeros((on_count, off_count))

        for i, k_on in enumerate(np.logspace(low_bound, up_bound, num=on_count)):
            for j, k_off in enumerate(np.logspace(low_bound, up_bound, num=off_count)):
                model = PromoterModel(
                    rate_fn_matrix=[
                        [None, RF.Linear(k_on, tf)],
                        [RF.Constant(k_off), None],
                    ]
                )
                # print(f"TF{tf}-{'%.2f' % k_on}-{'%.2f' % k_off}")
                res[i, j] = pip.evaluate(model)
                print(
                    f"TF{tf}-{'%.2f' % (100 * (i * off_count + j + 1) / (on_count * off_count))}%"
                )

        np.save(fname_temp.format(tf_index=tf), res[tf])
        print(f"Cached TF{tf} data")
        return tf, res

    def optimise_simple(
        self,
        exogenous_data: NDArray[Shape["Any, Any, Any, Any"], Float],
        tf_names: List[str],
    ) -> None:
        on_count, off_count = 10, 10
        low_bound, up_bound = -2, 2
        replicates = 10
        classifier = "naive_bayes"
        single_env, single_tf = False, False

        data = exogenous_data[:1] if single_env else exogenous_data
        fname_temp = f"cache/latestv2/res_real_tf{{tf_index}}_{classifier}_reps{replicates}_{on_count}_{off_count}.npy"
        pip = OneStepDecodingPipeline(
            data,
            realised=True,
            replicates=replicates,
            classifier_name=classifier,
        )

        num_tfs = 1 if single_tf else data.shape[1]  # 5
        res = np.zeros((num_tfs, on_count, off_count))

        start = time.time()
        print("0.00%")

        with ProcessPoolExecutor(max_workers=num_tfs) as executor:
            params = {
                "template": fname_temp,
                "bounds": (on_count, off_count, low_bound, up_bound),
                "pipeline": pip,
            }

            futures = [
                executor.submit(GridSearch._simple_grid_search, i, params)
                for i in range(num_tfs)
            ]

            for future in as_completed(futures):
                tf, data = future.result()
                res[tf] = data

            executor.shutdown()

        print(f"{time.time() - start}s elapsed")

        fig, axes = plt.subplots(1, num_tfs, sharey=True)

        if num_tfs == 1:
            axes = [axes]

        for i, pair in enumerate(zip(axes, res)):
            ax, res = pair
            ax.title.set_text(tf_names[i])
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

        plt.savefig(f"cache/grid_tfs_{on_count}_{off_count}", dpi=100)
        print(f"Cached data to {fname_temp.format(tf_index='x')}")
