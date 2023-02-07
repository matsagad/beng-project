import numpy as np
from models.rates.function import RateFunction as RF
from models.model import PromoterModel
from pipeline.one_step_decoding import OneStepDecodingPipeline
import matplotlib.pyplot as plt
import time
from multiprocessing.pool import ThreadPool
import os.path


class GridSearch:
    def optimise_simple(self, exogenous_data):
        on_count, off_count = 10, 10
        low_bound, up_bound = -2, 2
        replicates = 5
        classifier = "decision_tree"
        single_env, single_tf = False, False

        data = exogenous_data[:1] if single_env else exogenous_data
        fname_temp = f"cache/res_real_tf{{tf_index}}_{classifier}_reps{replicates}_{on_count}_{off_count}.npy"
        pip_real = OneStepDecodingPipeline(
            data,
            realised=True,
            replicates=replicates,
            classifier_name=classifier,
        )

        tf_names = ["msn2", "sfp1", "dot6", "maf1"]
        num_tfs = 1 if single_tf else data.shape[1]  # 4
        res_real = np.zeros((num_tfs, on_count, off_count))

        def single_grid_search(tf: int):
            fname = fname_temp.format(tf_index=tf)
            print(f"Starting grid search for TF{tf}")
            if os.path.isfile(fname):
                res_real[tf] = np.load(fname)
                print(f"Used cached TF{tf} data")
                return
            for i, k_on in enumerate(np.logspace(low_bound, up_bound, num=on_count)):
                for j, k_off in enumerate(
                    np.logspace(low_bound, up_bound, num=off_count)
                ):
                    model = PromoterModel(
                        rate_fn_matrix=[
                            [None, RF.Linear(k_on, tf)],
                            [RF.Constant(k_off), None],
                        ]
                    )
                    res_real[tf, i, j] = pip_real.evaluate(model)
                    print(
                        f"TF{tf}-{'%.2f' % (100 * (i * off_count + j + 1) / (on_count * off_count))}%"
                    )
            np.save(fname_temp.format(tf_index=tf), res_real[tf])
            print(f"Cached TF{tf} data")

        start = time.time()
        print("0.00%")

        with ThreadPool(4) as pool:
            pool.map(single_grid_search, [i for i in range(num_tfs)])

        print(f"{time.time() - start}s elapsed")

        fig, axes = plt.subplots(1, num_tfs, sharey=True)

        if num_tfs == 1:
            axes = [axes]

        for i, pair in enumerate(zip(axes, (res_real))):
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
