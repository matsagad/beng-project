import numpy as np
from models.rates.function import RateFunction as RF
from models.model import PromoterModel
from pipeline.one_step_decoding import OneStepDecodingPipeline
import matplotlib.pyplot as plt
import time
from multiprocessing.pool import ThreadPool

class GridSearch:
    
    def optimise(self, exogenous_data):
        on_count, off_count = 10, 10
        fname_temp = f"cache/res_real_tf{{tf_index}}_{on_count}_{off_count}.npy"
        pip_real = OneStepDecodingPipeline(exogenous_data, realised=True)

        tf_names = ["msn2", "sfp1", "dot6", "maf1"]
        num_tfs = exogenous_data.shape[1]  # 4
        res_real = np.zeros((num_tfs, on_count, off_count))

        def single_grid_search(tf: int):
            print(f"Starting grid search for TF{tf}")
            try:
                res_real[tf] = np.load(fname_temp.format(tf_index=tf))
                print(f"Used cached TF{tf} data")
            except:
                for i, k_on in enumerate(np.logspace(-2, 2, num=on_count)):
                    for j, k_off in enumerate(np.logspace(-2, 2, num=off_count)):
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

        with ThreadPool(10) as pool:
            pool.map(single_grid_search, [i for i in range(num_tfs)])
        
        print(f"{time.time() - start}s elapsed")

        fig, axes = plt.subplots(1, num_tfs, sharey=True)

        if num_tfs == 1:
            axes = [axes]

        for i, pair in enumerate(zip(axes, (res_real))):
            ax, res = pair
            ax.title.set_text(tf_names[i])
            im = ax.imshow(
                res,
                cmap="rainbow",
                extent=(-2, 2, -2, 2),
                aspect="equal",
                interpolation="none",
                vmin=0,
                vmax=2,
            )

        fig.colorbar(im, ax=axes, location="bottom")

        plt.savefig(f"cache/grid_tfs_{on_count}_{off_count}", dpi=100)
