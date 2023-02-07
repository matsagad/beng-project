from abc import ABC, abstractmethod
from models.model import PromoterModel
from nptyping import NDArray, Shape, Float


class StochasticSimulator(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def simulate(
        self, model: PromoterModel
    ) -> NDArray[Shape["Any, Any, Any, Any"], Float]:
        """Simulates the trajectory of the model.

        Args:
            model : a promoter model

        Returns:
            A time series of the model's state. Its dimensions are given
            by: # of times, # of classes, batch size, # of states.
        """
        pass

    @staticmethod
    def visualise_trajectory(
        trajectory: NDArray[Shape["Any, Any, Any, Any"], Float],
        model: PromoterModel = None,
        average: bool = True,
        batch_num: int = 0,
        show_inactive: bool = True,
    ) -> None:
        import matplotlib.cm as cm
        import matplotlib.pyplot as plt
        from matplotlib.colors import rgb2hex
        from matplotlib import rcParams
        import numpy as np

        # rcParams["text.usetex"] = True
        
        num_times, num_classes, batch_size, num_states = trajectory.shape
        x = np.arange(num_times)

        fig, axes = plt.subplots(1, num_classes, sharey=True)

        if num_classes == 1:
            axes = [axes]

        labels, colors = None, None
        if model is not None:
            labels = np.zeros(num_states, dtype=object)
            labels[model.active_states] = [
                f"$A_{i}$" for i in range(sum(model.active_states))
            ]
            labels[~model.active_states] = [
                f"$I_{i}$" for i in range(sum(~model.active_states))
            ]

            colors = np.zeros(num_states, dtype=object)
            for (cmap_name, states) in zip(
                ["summer", "autumn"], [model.active_states, ~model.active_states]
            ):
                cmap = cm.get_cmap(cmap_name, 2 * sum(states))
                colors[states] = [
                    rgb2hex(cmap(i), keep_alpha=True) for i in range(sum(states))
                ]

        for (env_class, ax) in enumerate(axes):
            for state in range(num_states):
                if not show_inactive and model is not None and not model.active_states[state]:
                    continue
                ax.plot(
                    x,
                    np.average(trajectory[:, env_class, :, state], axis=1)
                    if average
                    else trajectory[:, env_class, batch_num, state],
                    label=(labels[state] if labels is not None else None),
                    color=(colors[state] if colors is not None else None),
                )

        if labels is not None:
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            fig.legend(
                by_label.values(),
                by_label.keys(),
                loc="upper center",
                ncol=num_states,
                fancybox=True,
            )

        plt.show()
