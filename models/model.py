from typing import List
from models.rates.function import RateFunction
from nptyping import NDArray, Shape, Float, Int
from scipy.linalg import expm
import numpy as np


class PromoterModel:
    def __init__(self, rate_fn_matrix: List[List[RateFunction.Function]]):
        """
        Args:
            rate_fn_matrix  : a 2D matrix of rate functions; diagonals are
                              expected to be None and adjusted later to get
                              a row sum of zero.
        """
        self.rate_fn_matrix = rate_fn_matrix

        # Initial state is zero by default
        self.init_state = np.zeros(len(rate_fn_matrix), dtype=int)
        self.init_state[0] = 1

        # Only active state is last state by default
        self.active_states = np.zeros(len(rate_fn_matrix), dtype=bool)
        self.active_states[-1] = True
        # (should be changed for models with more than one active
        # state, e.g. competing activator)

    def with_init_state(
        self, init_state: NDArray[Shape["Any"], Int]
    ) -> "PromoterModel":
        self.init_state = init_state
        return self

    def with_active_states(
        self, active_states: NDArray[Shape["Any"], Int]
    ) -> "PromoterModel":
        self.active_states = np.zeros(len(self.rate_fn_matrix), dtype=bool)
        self.active_states[active_states] = True
        return self

    def get_generator(
        self, exogenous_data: NDArray[Shape["Any, Any, Any"], Float]
    ) -> NDArray[Shape["Any, Any, Any"], Float]:
        batch_size = exogenous_data.shape[1]
        rate_fn_len = exogenous_data.shape[1:]
        length = exogenous_data.shape[-1]

        generator = np.stack(
            [
                np.stack(
                    [
                        np.zeros(rate_fn_len)
                        if not rate_fn
                        else rate_fn.evaluate(exogenous_data)
                        for rate_fn in row
                    ],
                    axis=2,
                )
                for row in self.rate_fn_matrix
            ],
            axis=2,
        )

        for j in range(batch_size):
            for i in range(length):
                np.fill_diagonal(generator[j, i], -np.sum(generator[j, i], axis=1))
        return generator

    def get_matrix_exp(
        self, exogenous_data: NDArray[Shape["Any, Any, Any"], Float], tau: float
    ) -> NDArray[Shape["Any, Any, Any"], Float]:
        return expm(tau * self.get_generator(exogenous_data))

    def visualise(self, save: bool = False) -> None:
        from igraph import Graph, plot
        import matplotlib.pyplot as plt
        from matplotlib import rcParams

        rcParams["text.usetex"] = True

        # Colors are from the "marumaru gum" palette by sukinapan!
        palette = ["#96beb1", "#fda9a9", "#f3eded"]

        num_states = len(self.active_states)

        graph = Graph(directed=True)
        graph.add_vertices(num_states)

        properties = np.zeros((num_states, 2), dtype=object)
        properties[self.active_states, 0] = [
            f"$A_{i}$" for i in range(sum(self.active_states))
        ]
        properties[self.active_states, 1] = palette[0]
        properties[~self.active_states, 0] = [
            f"$I_{i}$" for i in range(sum(1 - self.active_states))
        ]
        properties[~self.active_states, 1] = palette[1]

        graph.vs["label"] = properties[:, 0]
        graph.vs["color"] = properties[:, 1]

        edges = [
            [(i, j), entry.str()]
            for (i, row) in enumerate(self.rate_fn_matrix)
            for (j, entry) in enumerate(row)
            if entry is not None
        ]
        graph.add_edges([edge for edge, _ in edges])
        graph.es["label"] = [label for _, label in edges]
        graph.es["label_size"] = 12

        visual_style = {
            # "edge_curved": 0,
            "background": None,
            "edge_label": graph.es["label"],
            "edge_background": palette[2],
        }

        fig, ax = plt.subplots()
        ax.set_facecolor(palette[2])
        ax.set_aspect(1)
        fig.set_facecolor(palette[2])

        plot(graph, target=ax, **visual_style)

        if save:
            plt.savefig("model.png", dpi=180, bbox_inches="tight", pad_inches=0)
            return

        plt.show()
