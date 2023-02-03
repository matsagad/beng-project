import numpy as np
from models.rates.function import RateFunction as RF
from models.model import PromoterModel
from pipeline.one_step_decoding import OneStepDecodingPipeline
import matplotlib.pyplot as plt
import time


class GridSearch:
    def optimise(self, exogenous_data):
        on_count, off_count = 3, 3
        pip_real = OneStepDecodingPipeline(exogenous_data, realised=True)
        pip_prob = OneStepDecodingPipeline(exogenous_data, realised=False)

        res_real = np.zeros((on_count, off_count))
        res_prob = np.zeros((on_count, off_count))

        tf_index = 0

        start = time.time()
        print("0.00%")

        for i, k_on in enumerate(np.logspace(-2, 2, num=on_count)):
            for j, k_off in enumerate(np.logspace(-2, 2, num=off_count)):
                model = PromoterModel(
                    rate_fn_matrix=[
                        [None, RF.Linear(k_on, tf_index)],
                        [RF.Constant(k_off), None],
                    ]
                )
                res_real[i, j] = pip_real.evaluate(model)
                res_prob[i, j] = pip_prob.evaluate(model)
                print(
                    f"{'%.2f' % (100 * (i * off_count + j + 1) / (on_count * off_count))}%"
                )

        print(f"{time.time() - start}s elapsed")

        np.save(f"cache/res_real_tf{tf_index}_{on_count}_{off_count}.npy", res_real)
        np.save(f"cache/res_prob_tf{tf_index}_{on_count}_{off_count}.npy", res_prob)

        fig, axes = plt.subplots(1, 2, sharey=True)
        for ax, res in zip(axes, (res_real, res_prob)):
            im = ax.imshow(
                res,
                cmap="magma",
                extent=(-2, 2, -2, 2),
                aspect="equal",
                interpolation="none",
                vmin=0,
                vmax=2,
            )

        fig.colorbar(im, ax=axes, location="bottom")

        plt.savefig(f"cache/grid_tf{tf_index}_{on_count}_{off_count}", dpi=100)
