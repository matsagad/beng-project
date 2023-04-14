from collections import Counter
from pipeline.one_step_decoding import OneStepDecodingPipeline
from utils.data import ClassWithData
import matplotlib.pyplot as plt


class GeneticAnalysisExamples(ClassWithData):
    def __init__(self, f_models: str = None, f_stats: str = None, job_id: str = "7361710"):
        super().__init__()

        self.save_pictures = True

        if f_models is not None and f_stats is not None:
            self.models = self.unpickle(f_models)
            self.stats = self.unpickle(f_stats)
            self.job_id = f_stats.split(".")[0]
            return
        
        # Models and stats saved from a genetic algorithm job
        # (Replace with path to where such files exist)
        _job_folder = "jobs"
        self.job_id = job_id
        self.models = self.unpickle(f"{_job_folder}/{self.job_id}_models.dat")
        self.stats = self.unpickle(f"{_job_folder}/{self.job_id}_stats_models.dat")

    def evaluate_models(self):
        """
        Estimate MI of models under different classifiers
        and number of cell replicates.
        """
        num_models = 10
        repeats = 3

        for classifier in ("naive_bayes", "svm"):
            print(f"Classifier: {classifier}")

            for num_replicates in [10, 20, 30]:
                print(f"\tNumber of replicates: {num_replicates}")
                pip = OneStepDecodingPipeline(
                    **{
                        **self.default_pip_args,
                        "classifier_name": classifier,
                        "replicates": num_replicates,
                    }
                )
                pip.set_parallel()

                for _, _, _, model in self.models[:num_models]:
                    total = 0
                    for _ in range(repeats):
                        total += pip.evaluate(model)
                    print(f"\t\t{model.hash(short=True)}: {total/repeats:.3f}")

    def visualise_models(self):
        """
        Visualise the top models found and their MI estimates.
        """
        rows, cols = 2, 5
        calculate = False
        repeats = 1

        fig, axes = plt.subplots(
            rows, cols, sharex=True, sharey=True, figsize=(4 * cols, 4 * rows)
        )
        fig.tight_layout()

        if calculate:
            pip = OneStepDecodingPipeline(**self.default_pip_args)
            pip.set_parallel()

        if rows == 1:
            axes = [axes]

        for i, row in enumerate(axes):
            for (fitness, mi, _, model), ax in zip(
                self.models[cols * i : cols * (i + 1)], row
            ):
                if calculate:
                    total_mi = 0
                    for _ in range(repeats):
                        total_mi += pip.evaluate(model)
                    mi = total_mi / repeats
                    fitness = mi

                model.visualise(target_ax=ax, transparent=True)
                ax.set_xlabel(
                    f"Fitness: {fitness:.3f}, MI: {mi:.3f}, {model.hash(short=True)}"
                )

                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_aspect("equal")

        plt.subplots_adjust(wspace=0.05, hspace=0.05)

        if self.save_pictures:
            plt.savefig(
                fname=f"{self.SAVE_FOLDER}/visualise_models__{self.job_id}.png",
                bbox_inches="tight",
                pad_inches=0.25,
            )
            return
        plt.show()

    def examine_run_stats(self):
        """
        Examine statistics from running the genetic algorithm.
        """
        include_duration = False
        model_groups = ("elite", "population", "non_elite")
        stat_labels = ("avg_fitness", "avg_mi", "std_mi", "avg_num_states")
        group_colors = ("firebrick", "seagreen", "royalblue")

        fig, axes = plt.subplots(
            len(stat_labels) + int(include_duration), 1, sharex=True
        )
        fig.tight_layout()

        for label, ax in zip(stat_labels, axes):
            for group, color in zip(model_groups, group_colors):
                ys = self.stats[group][label]
                ax.plot(range(len(ys)), ys, label=group, color=color)
                ax.set_xticks(list(range(1, len(ys), max(1, (len(ys) - 1) // 10))))
            ax.set_ylabel(label)
        plt.xlabel("Number of generations")

        if include_duration:
            duration_states = self.stats["avg_time_duration"]
            axes[-1].plot(
                range(len(duration_states)), duration_states, color=group_colors[-1]
            )
            axes[-1].set_ylabel("duration")

        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        fig.legend(
            by_label.values(),
            by_label.keys(),
            loc="lower center",
            ncol=len(model_groups),
            bbox_to_anchor=[0.5, -0.1],
        )

        axes[0].set_title(f"{self.job_id}", loc="center")

        if self.save_pictures:
            plt.savefig(
                f"{self.SAVE_FOLDER}/evolutionary_run_stats__{self.job_id}.png",
                dpi=200,
                bbox_inches="tight",
                pad_inches=0.15,
            )
            return
        plt.show()

    def evaluate_tf_presence_in_models(self):
        """
        Evaluate the distribution of models that use certain TFs and
        TF combination groups. Also evaluate the distribution of model
        edges that depend on certain TFs.

        Analyses on 10% slices of the population as well as the entire
        population are carried out.
        """
        from itertools import combinations
        from matplotlib.cm import get_cmap
        from matplotlib.colors import rgb2hex

        n_rows = 10
        total = len(self.models)
        indices = list(range(0, total + 1, total // n_rows))

        model_classes = [
            self.models[start:end] for start, end in zip(indices, indices[1:])
        ]
        groups = [
            f"{100 * start // total}-{100 * end // total}%"
            for start, end in zip(indices, indices[1:])
        ]

        cmap = get_cmap("viridis_r")
        colors = [rgb2hex(cmap(i / n_rows)) for i in range(n_rows + 1)]

        num_tfs = len(self.tf_names)
        total_combinations = [
            group
            for size in range(0, num_tfs + 1)
            for group in combinations(range(num_tfs), size)
        ]

        model_groups = {
            group: {
                "models": models,
                "counters": {
                    "edge_freq": Counter(),
                    "freq": Counter(),
                    "group_freq": Counter(total_combinations),
                },
                "color": color,
            }
            for group, models, color in zip(groups, model_classes, colors)
        }

        for group, stats in model_groups.items():
            freq = stats["counters"]["freq"]
            group_freq = stats["counters"]["group_freq"]
            edge_freq = stats["counters"]["edge_freq"]

            for _, _, _, model in stats["models"]:
                tfs = set()
                for row in model.rate_fn_matrix:
                    for fn in row:
                        if fn is None:
                            continue
                        edge_freq.update(fn.tfs)
                        tfs.update(fn.tfs)
                group_freq[tuple(sorted(tfs))] += 1
                freq.update(tfs)

        fig, axes = plt.subplots(
            1 + len(groups),
            3,
            sharex="col",
            figsize=(18, 9),
            gridspec_kw={"width_ratios": [1, 1, 6]},
        )
        fig.tight_layout()

        totals = {}
        for group, stats in model_groups.items():
            for counter, counter_obj in stats["counters"].items():
                if counter not in totals:
                    totals[counter] = {}
                if "total" not in totals[counter]:
                    totals[counter]["total"] = 0
                group_total = counter_obj.total()
                totals[counter][group] = group_total
                totals[counter]["total"] += group_total

        group_results = {
            group: {
                counter: (
                    [
                        100 * cnt / totals[counter][group]
                        for cnt in counter_obj.values()
                    ],
                    [
                        100 * cnt / totals[counter]["total"]
                        for cnt in counter_obj.values()
                    ],
                )
                for counter, counter_obj in stats["counters"].items()
            }
            for group, stats in model_groups.items()
        }

        shared_row = axes[0]
        tf_group_labels = [
            "\n".join(self.tf_names[tf] for tf in group) for group in total_combinations
        ]
        tf_group_labels[0] = "w/o"
        tick_labels = [self.tf_names, self.tf_names, tf_group_labels]

        for group, results in group_results.items():
            for counter, tick_label, ax in zip(results.keys(), tick_labels, shared_row):
                ax.bar(
                    tick_label,
                    results[counter][1],
                    label=group,
                    color=model_groups[group]["color"],
                    edgecolor="black",
                    linewidth=0.5,
                )
        shared_row[0].set_ylabel("population")

        for (group, results), row in zip(group_results.items(), axes[1:]):
            for counter, tick_label, ax in zip(results.keys(), tick_labels, row):
                ax.bar(
                    tick_label,
                    results[counter][0],
                    label=group,
                    color=model_groups[group]["color"],
                    edgecolor="black",
                    linewidth=0.5,
                )
            row[0].set_ylabel(group)

        axes[0][0].set_title("% of model edges using TF")
        axes[0][1].set_title("% of models using TF")
        axes[0][2].set_title("% of models using TF group")

        axes[-1][0].set_xlabel("Transcription Factor")
        axes[-1][1].set_xlabel("Transcription Factor")
        axes[-1][2].set_xlabel("TF Group")

        if self.save_pictures:
            plt.savefig(
                f"{self.SAVE_FOLDER}/evaluate_tf_presence_in_models.png",
                bbox_inches="tight",
                pad_inches=1,
            )
            return
        plt.show()
