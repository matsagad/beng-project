from evolution.genetic.operators.crossover import CrossoverOperator
from mi_estimation.decoding import DecodingEstimator
from models.generator import ModelGenerator
from models.model import PromoterModel
from ssa.one_step import OneStepSimulator
from ssa.one_step import OneStepSimulator
from utils.data import ClassWithData
import matplotlib.pyplot as plt
import numpy as np


class VisualisationExamples(ClassWithData):
    def __init__(self):
        super().__init__()
        self.model = ModelGenerator.get_random_model(3)
        self.save_pictures = True

    def visualise_model(self):
        """
        Visualise a given model.
        """
        self.model.visualise(
            save=True,
            fname=f"{self.SAVE_FOLDER}/visualise_model.png"
            if self.save_pictures
            else None,
        )

    def visualise_trajectory(self):
        """
        Visualise a single cell's activity, the population's average activity,
        and the probability distribution of activity for each state given a
        specified model. (Line chart)
        """
        sim = OneStepSimulator(**self.default_sim_args)
        trajectories = sim.simulate(self.model)

        # Visualise activity of a single cell
        OneStepSimulator.visualise_trajectory(
            trajectories,
            self.model,
            average=False,
            batch_num=27,
            fname=f"{self.SAVE_FOLDER}/visualise_trajectory__single.png"
            if self.save_pictures
            else None,
        )

        # Visualise average activity across entire cell population
        OneStepSimulator.visualise_trajectory(
            trajectories,
            self.model,
            average=True,
            fname=f"{self.SAVE_FOLDER}/visualise_trajectory__average.png"
            if self.save_pictures
            else None,
        )

        # Visualise probability distribution of states chosen
        sim.realised = False
        prob_trajectories = sim.simulate(self.model)

        OneStepSimulator.visualise_trajectory(
            prob_trajectories,
            self.model,
            average=True,
            fname=f"{self.SAVE_FOLDER}/visualise_trajectory__probabilistic.png"
            if self.save_pictures
            else None,
        )

    def visualise_activity(self):
        """
        Visualise the activity of a given model in different environments, as
        stress is introduced. (2D Heatmap)
        """
        sim = OneStepSimulator(**self.default_sim_args)
        est = DecodingEstimator(**self.default_est_args)

        trajectories = sim.simulate(self.model)
        slices = est._split_classes(self.model, trajectories)

        num_envs = self.data.shape[0]
        fig, axes = plt.subplots(1, num_envs + 1, sharey="row", figsize=(5, 10))

        for slice, ax in zip(slices, axes):
            im = ax.imshow(
                slice,
                cmap="rainbow",
                aspect="auto",
                interpolation="none",
                vmin=0,
                vmax=1,
            )

        fig.colorbar(im, ax=axes, location="bottom")

        if self.save_pictures:
            plt.savefig(f"{self.SAVE_FOLDER}/visualise_activity.png", dpi=200)
            return
        plt.show()

    def visualise_tf_concentration(self):
        """
        Visualise the trajectory of TF concentrations in different environments,
        as stress is introduced. (2D Heatmap)
        """
        num_envs, num_tfs = self.data.shape[:2]

        fig, axes = plt.subplots(
            num_tfs, num_envs + 1, sharey="row", sharex=True, figsize=(5, 10)
        )
        fig.tight_layout()

        est = DecodingEstimator(**self.default_est_args)
        plt.subplots_adjust(wspace=0.1, hspace=0.1)

        TIME_AXIS = 2
        for i, row in enumerate(axes):
            raw_data = np.moveaxis(self.data[:, i], TIME_AXIS, 0)
            raw_data = raw_data.reshape((*raw_data.shape, 1))
            split_raw = est._split_classes(PromoterModel.dummy(), raw_data)

            for j, ax in enumerate(row):
                ax.imshow(
                    split_raw[j],
                    cmap="rainbow",
                    aspect="auto",
                    interpolation="none",
                    vmin=0,
                    vmax=1,
                )
                ax.set_yticks([])
            row[0].set_ylabel(self.tf_names[i])

        fst_row = axes[0]
        env_labels = ["rich", "carbon", "osmotic", "oxidative"]

        for ax, label in zip(fst_row, env_labels):
            ax.xaxis.set_label_position("top")
            ax.set_xlabel(label)

        if self.save_pictures:
            plt.savefig(
                f"{self.SAVE_FOLDER}/visualise_tf_concentration.png",
                dpi=200,
                pad_inches=0.2,
            )
            return
        plt.show()

    def visualise_activity_grid(self):
        """
        Visualise the animated activity of cells on a grid at different
        environments, as stress is introduced.
        """
        num_envs, _, num_cells, num_times = self.data.shape
        replicates = 10
        grid_dims = int((num_cells * replicates) ** 0.5)

        trajectories = OneStepSimulator(**self.default_sim_args).simulate(self.model)
        activity_weights = self.model.activity_weights / np.sum(
            self.model.activity_weights
        )
        activity = np.sum(activity_weights * trajectories, axis=3)

        from matplotlib.animation import FuncAnimation
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(
            1, num_envs, sharex=True, sharey=True, figsize=(2 * num_envs, 2)
        )
        fig.tight_layout()
        plt.subplots_adjust(wspace=0, hspace=0)

        for i, ax in enumerate(axes):
            ax.imshow(np.zeros((grid_dims, grid_dims)))
            plt.setp(ax.get_xticklabels(), visible=False)
            plt.setp(ax.get_yticklabels(), visible=False)
            ax.tick_params(axis="both", which="both", length=0)

        env_labels = ["carbon", "osmotic", "oxidative"]

        for ax, label in zip(axes, env_labels):
            ax.xaxis.set_label_position("top")
            ax.set_xlabel(label)

        im = fig.show()

        def _animate(t):
            for i, ax in enumerate(axes):
                ax.imshow(
                    activity[t, i, : grid_dims * grid_dims].reshape(
                        (grid_dims, grid_dims)
                    ),
                    aspect="auto",
                    vmin=0,
                    vmax=1,
                )
            return [im]

        def _init():
            return _animate(0)

        animation = FuncAnimation(
            fig,
            init_func=_init,
            func=_animate,
            frames=num_times,
            interval=20,
            blit=False,
        )
        animation.save(
            f"{self.SAVE_FOLDER}/visualise_activity_grid.gif", writer="imagemagick"
        )

    def visualise_tf_concentration_grid(self):
        """
        Visualise the animated concentration of TFs in cells on a grid at
        different environments, as stress is introduced.
        """
        num_envs, num_tfs, _, num_times = self.data.shape

        from matplotlib.animation import FuncAnimation

        fig, axes = plt.subplots(
            num_tfs, num_envs, sharex=True, sharey=True, figsize=(num_envs, num_tfs)
        )
        fig.tight_layout()
        plt.subplots_adjust(wspace=0, hspace=0)
        for row in axes:
            for ax in row:
                ax.imshow(np.zeros((10, 10)), aspect="auto", vmin=0, vmax=1)
                plt.setp(ax.get_xticklabels(), visible=False)
                plt.setp(ax.get_yticklabels(), visible=False)
                ax.tick_params(axis="both", which="both", length=0)

        env_labels = ["carbon", "osmotic", "oxidative"]

        for ax, label in zip(axes[0], env_labels):
            ax.axis("on")
            ax.xaxis.set_label_position("top")
            ax.set_xlabel(label)

        for i, row in enumerate(axes):
            row[0].set_ylabel(self.tf_names[i])

        im = fig.show()

        def _init():
            return [im]

        def _animate(t):
            for i, row in enumerate(axes):
                for j, ax in enumerate(row):
                    ax.imshow(
                        self.data[j, i, :100, t].reshape((10, 10)),
                        aspect="auto",
                        vmin=0,
                        vmax=1,
                    )
            return [im]

        animation = FuncAnimation(
            fig,
            init_func=_init,
            func=_animate,
            frames=num_times,
            interval=20,
            blit=False,
        )
        animation.save(
            f"{self.SAVE_FOLDER}/visualise_tf_concentration_grid.gif",
            writer="imagemagick",
        )

    def visualise_crossover(self):
        """
        Visualise the offsprings produced by a crossover between two random models.
        """
        model1 = ModelGenerator.get_random_model(5, p_edge=0.1, one_active_state=False)
        model2 = ModelGenerator.get_random_model(3, p_edge=0.1, one_active_state=False)
        crossover = CrossoverOperator.subgraph_swap

        model_pairs = ((model1, model2), crossover(model1, model2))

        fig, axes = plt.subplots(2, 2, sharey=True, sharex=True, figsize=(6, 6))
        fig.tight_layout()

        for row, model_pair in zip(axes, model_pairs):
            for ax, model in zip(row, model_pair):
                model.visualise(target_ax=ax)

        for ax, label in zip(axes[:, 0], ("Parents", "Children")):
            ax.set_ylabel(label)

        plt.subplots_adjust(wspace=0, hspace=0)

        if self.save_pictures:
            plt.savefig(
                f"{self.SAVE_FOLDER}/visualise_crossover.png",
                dpi=200,
                bbox_inches="tight",
                pad_inches=0,
            )
            return
        plt.show()

    def visualise_crossover_chart(self):
        """
        Visualise the architectures of offpsrings produced by a crossover between
        many different models.
        """
        rows, cols = 5, 5
        crossover = CrossoverOperator.subgraph_swap

        scale = 8 / max(rows, cols)
        fig = plt.figure(figsize=(scale * cols, scale * rows))

        subfigs = fig.subfigures(
            rows,
            cols,
            height_ratios=np.ones(rows),
            width_ratios=np.ones(cols),
        )

        for dim in (rows, cols):
            if dim == 1:
                subfigs = [subfigs]

        for row in subfigs:
            for subfig in row:
                axes = subfig.subplots(
                    2,
                    2,
                    sharey=True,
                    sharex=True,
                )
                parents = [
                    ModelGenerator.get_random_model(states=states, p_edge=p_edge)
                    for states, p_edge in zip(
                        2 + np.random.choice(8, size=2), np.random.uniform(size=2)
                    )
                ]

                models = (parents, crossover(*parents))

                for model_pair, subrow in zip(models, axes):
                    for model, ax in zip(model_pair, subrow):
                        ax.set_aspect("equal")
                        model.visualise(target_ax=ax, small_size=True)
                        ax.axis("off")

                for ax, label in zip(axes[:, 0], ("Parents", "Children")):
                    ax.set_ylabel(label)
                parents = models[1]

                subfig.subplots_adjust(wspace=0, hspace=0)

        plt.axis("off")
        if self.save_pictures:
            plt.savefig(
                f"{self.SAVE_FOLDER}/visualise_crossover_chart.png",
                dpi=200,
                bbox_inches="tight",
                pad_inches=0.1,
            )
            return
        plt.show()

    def visualise_crossbreeding(self):
        """
        Visualise crossbreeding between two models and their successive
        offsprings for many iterations.
        """
        iterations = 10
        crossover = CrossoverOperator.one_point_triangular_row_swap

        scale = 8 / iterations

        fig, axes = plt.subplots(
            nrows=2,
            ncols=iterations,
            sharex=True,
            sharey=True,
            figsize=(scale * iterations, scale),
        )
        fig.tight_layout()

        models = [
            ModelGenerator.get_random_model(states=states, p_edge=p_edge)
            for states, p_edge in zip((5, 5), (0.1, 0.5))
        ]

        for col in axes.T:
            for model, ax in zip(models, col):
                ax.set_aspect("equal")
                ax.axis("off")
                model.visualise(target_ax=ax, small_size=True)

            models = crossover(*models)

        plt.axis("off")
        if self.save_pictures:
            plt.savefig(
                f"{self.SAVE_FOLDER}/visualise_crossbreeding.png",
                dpi=200,
                bbox_inches="tight",
                pad_inches=0.1,
            )
            return
        plt.show()
