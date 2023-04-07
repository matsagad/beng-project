from ssa.one_step import OneStepSimulator
from utils.data import ClassWithData
from mi_estimation.decoding import DecodingEstimator
from ssa.one_step import OneStepSimulator
from pipeline.one_step_decoding import OneStepDecodingPipeline
from models.model import PromoterModel
from models.preset import Preset
import numpy as np
import os
import pickle


class MITrendsExamples(ClassWithData):
    def __init__(self):
        super().__init__()
        self.model = Preset.simple(2, 1, 1)

        self.default_sim_args = {
            "exogenous_data": self.data,
            "tau": self.time_delta,
            "realised": True,
            "replicates": 10,
        }

        self.default_est_args = {
            "origin": self.origin,
            "interval": self.interval,
            "replicates": 10,
            "classifier_name": "naive_bayes",
        }

        self.default_pip_args = {
            **self.default_sim_args,
            **self.default_est_args,
            "classifier_name": "naive_bayes",
        }
        self.SAVE_FOLDER = f"{self.CACHE_FOLDER}/mi_trends_examples/saves"
        self.CACHING_FOLDER = f"{self.CACHE_FOLDER}/mi_trends_examples/cache"

    def mi_vs_interval(self):
        """
        Plot the MI estimates as the interval size from
        the origin is varied.
        """
        repeats = 10
        scores = []

        for interval in range(1, self.origin):
            pip = OneStepDecodingPipeline(
                **{**self.default_pip_args, "interval": interval}
            )
            pip.set_parallel()

            mi_score = 0
            for _ in range(repeats):
                mi_score += pip.evaluate(self.model)
            mi_score = mi_score / repeats

            print(f"MI at interval={interval}: {mi_score:.3f}")
            scores.append(mi_score)

        print(scores)

        import matplotlib.pyplot as plt

        plt.plot([i for i in range(1, self.origin)], scores)
        plt.xlabel("Length of time interval from origin")
        plt.ylabel("MI")

        plt.savefig(f"{self.SAVE_FOLDER}/mi_vs_interval.png", dpi=100)

    def _mi_distribution__evaluate(estimator, model, trajectories, i):
        mi = estimator.estimate(model, trajectories)
        return mi, i

    def mi_distribution(self):
        """
        Plot the distribution of MI estimates as the number
        of cell replicates is varied.
        """
        trials = 2
        replicate_counts = [1, 2, 5, 10, 20]
        n_processors = max(1, os.cpu_count() - 1)

        fname = f"{self.CACHING_FOLDER}/mi_distribution__{trials}_{'_'.join(map(str,replicate_counts))}.dat"

        from concurrent.futures import ProcessPoolExecutor, as_completed

        if os.path.isfile(fname):
            print("Using cached MI distribution.")
            hist_map = self.unpickle(fname)
        else:
            hist_map = dict()

            for replicates in replicate_counts:
                hist_map[replicates] = []

                sim = OneStepSimulator(
                    **{**self.default_sim_args, "replicates": replicates}
                )
                trajectories = sim.simulate(self.model)

                est = DecodingEstimator(
                    **{**self.default_est_args, "replicates": replicates}
                )
                est.parallel = False

                with ProcessPoolExecutor(
                    max_workers=min(n_processors, trials),
                ) as executor:
                    futures = []
                    for i in range(trials):
                        futures.append(
                            executor.submit(
                                MITrendsExamples._mi_distribution__evaluate,
                                est,
                                self.model,
                                trajectories,
                                i,
                            )
                        )

                    for future in as_completed(futures):
                        mi_score, i = future.result()
                        hist_map[replicates].append(mi_score)
                        print(f"{replicates}-{i}: {mi_score:.3f}")

            with open(fname, "wb") as f:
                pickle.dump(hist_map, f)
                print("Cached best MI distribution.")

        import matplotlib.pyplot as plt

        bins = np.linspace(0, 0.6, 60)

        for reps, hist in hist_map.items():
            plt.hist(hist, bins, alpha=0.5, label=f"{reps} reps", edgecolor="black")

        # plt.hist(list(hist_map.values()), bins, label=list(hist_map.keys()))
        plt.legend(loc="upper right")
        plt.savefig(f"{self.SAVE_FOLDER}/mi_distribution.png", dpi=200)

    def max_mi_estimation(self):
        """
        Find the maximum MI that can be achieved by estimating
        directly on the nuclear localisation trajectory of TFs.

        When random_mesh=True, we attempt to mesh their trajectories
        together - an attempt to simulate mixing of trajectories. When set
        to False, their trajectories are merely appended together.
        """
        import itertools

        repeats = 10
        random_mesh = True
        classifier = "naive_bayes"

        dummy_model = PromoterModel.dummy()
        est = DecodingEstimator(
            self.origin, self.interval, classifier_name=classifier, replicates=1
        )
        est.parallel = True
        tf_split_data = []
        num_tfs = len(self.tf_names)

        for tf_index in range(num_tfs):
            TIME_AXIS = 2
            raw_data = np.moveaxis(self.data[:, tf_index], TIME_AXIS, 0)
            raw_data = raw_data.reshape((*raw_data.shape, 1))
            tf_split_data.append(est._split_classes(dummy_model, raw_data))

        print(f"{classifier}: interval {self.interval}, {repeats} reps")
        print(f"|\033[1m{'TF GROUP':^25}\033[0m|\033[1m{'MI':^25}\033[0m|")
        print(("|" + "-" * 25) * 2 + "|")
        interval = self.interval
        for group_size in range(1, num_tfs + 1):
            for comb in itertools.combinations(list(range(num_tfs)), group_size):
                comb_split_data = np.concatenate(
                    [tf_split_data[tf] for tf in comb], axis=2
                )
                split_data = comb_split_data

                total = 0
                for _ in range(repeats):
                    if random_mesh:
                        num_envs, num_cells = comb_split_data.shape[:2]

                        ## Predetermine which TFs to use regardless of cell
                        # indices = np.random.choice(
                        #     group_size, interval
                        # ) * interval + np.arange(interval)
                        # split_data = comb_split_data[:, :, indices]

                        ## Predetermine which TFs each cell uses regardless of environment
                        indices = np.random.choice(
                            group_size, (num_cells, interval)
                        ) * interval + np.arange(interval)
                        cell_indices, _ = np.indices((num_cells, interval))
                        split_data = comb_split_data[:, cell_indices, indices]

                        ## Randomly choose TF for each cell in all environments
                        # indices = np.random.choice(
                        #     group_size, (num_envs, num_cells, interval)
                        # ) * interval + np.arange(interval)
                        # env_indices, cell_indices, _ = np.indices((num_envs, num_cells, interval))
                        # split_data = comb_split_data[env_indices, cell_indices, indices]

                    total += est._estimate(split_data, halving=False)
                print(
                    f"|{','.join(self.tf_names[tf] for tf in comb):^25}|{(total/repeats):^25.3f}|"
                )

    def mi_vs_repeated_intervals(self):
        """
        Plot the MI estimates as intervals are repeated.
        """
        pip = OneStepDecodingPipeline(**self.default_pip_args)

        for scale in np.arange(1, 5):
            origin *= scale
            time_delta /= scale
            interval *= scale

            data = np.repeat(data, scale, axis=3)

            mi_score = pip.estimate(self.model)
            print(f"MI at {scale}x repeats: {mi_score:.3f}")
