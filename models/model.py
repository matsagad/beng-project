from typing import List
from models.rates.function import RateFunction
from nptyping import NDArray, Shape, Float, Int
from scipy.linalg import expm, norm
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

        # Probabilistic initial state is uniform distribution
        self.init_state = np.ones(len(rate_fn_matrix), dtype=int) / len(rate_fn_matrix)

        # Only active state is first state by default
        self.activity_weights = np.zeros(len(rate_fn_matrix))
        self.activity_weights[0] = 1
        # state, e.g. competing activator)

        self.num_states = len(rate_fn_matrix)
        self.num_edges = sum(sum(map(bool, row)) for row in self.rate_fn_matrix)
        self.numba = False

    def with_init_state(
        self, init_state: NDArray[Shape["Any"], Int]
    ) -> "PromoterModel":
        self.init_state = init_state
        return self

    def with_equal_active_states(
        self, states: NDArray[Shape["Any"], Int]
    ) -> "PromoterModel":
        self.activity_weights = np.zeros(self.num_states)
        self.activity_weights[states] = 1
        return self

    def with_activity_weights(
        self, weights: NDArray[Shape["Any"], Float]
    ) -> "PromoterModel":
        self.activity_weights = weights
        return self

    def get_generator(
        self, exogenous_data: NDArray[Shape["Any, Any, Any, Any"], Float]
    ) -> NDArray[Shape["Any, Any, Any, Any, Any"], Float]:
        """
        Args:
            exogenous_data: an array of exogenous data with dimensions:
                            # of classes, # of TFs, batch size, # of times

        Returns:
            An array of generator matrices with dimensions:
            # of classes, batch size, # of times, # of states, # of states
        """
        TIME_AXIS = len(exogenous_data.shape) - 1

        # swap # of tfs and # of classes axes
        tf_iterable_data = np.moveaxis(exogenous_data, 1, 0)
        batched_rate_shape = tf_iterable_data.shape[1:]

        generator = np.stack(
            [
                np.stack(
                    [
                        np.zeros(batched_rate_shape)
                        if not rate_fn
                        else rate_fn.evaluate(tf_iterable_data)
                        for rate_fn in row
                    ],
                    axis=TIME_AXIS,
                )
                for row in self.rate_fn_matrix
            ],
            axis=TIME_AXIS,
        )
        # generator shape is: # of classes, batch size, # of times, # of states, # of states

        RATE_MATRIX_ROW_AXIS = len(generator.shape) - 1
        row, col = np.diag_indices_from(generator[0, 0, 0])
        generator[:, :, :, row, col] = -generator.sum(axis=RATE_MATRIX_ROW_AXIS)

        return generator

    def get_matrix_exp(
        self, exogenous_data: NDArray[Shape["Any, Any, Any, Any"], Float], tau: float
    ) -> NDArray[Shape["Any, Any, Any, Any, Any"], Float]:
        Q = tau * self.get_generator(exogenous_data)
        scale = 1 << max(0, int((np.log2(norm(Q)))))
        # if self.numba:
        #     return np.linalg.matrix_power(numba_expm(Q / scale), scale)
        return np.linalg.matrix_power(expm(Q / scale), scale)

    def visualise(
        self,
        save: bool = False,
        fname: str = "cache/model.png",
        transparent: bool = False,
        small_size: bool = False,
        target_ax: any = None,
    ) -> None:
        from igraph import Graph, plot
        import matplotlib.pyplot as plt
        from matplotlib import rcParams

        # rcParams["text.usetex"] = True

        # Colors are from the "marumaru gum" palette by sukinapan!
        palette = ["#96beb1", "#fda9a9", "#f3eded"]

        graph = Graph(directed=True)
        graph.add_vertices(self.num_states)
        
        active_states = self.activity_weights > 0
        activity_weights = self.activity_weights[active_states] / np.sum(self.activity_weights)

        properties = np.zeros((self.num_states, 3), dtype=object)
        properties[active_states, 0] = [f"$\\underset{{ {activity_weights[i]:.2f} }}{{ A_{{ {i} }} }}$" for i in range(sum(active_states))]
        properties[active_states, 1] = palette[0]
        properties[active_states, 2] = 9
        properties[~active_states, 0] = [f"$I_{i}$" for i in range(sum(~active_states))]
        properties[~active_states, 1] = palette[1]
        properties[~active_states, 2] = 12

        graph.vs["label"] = properties[:, 0]
        graph.vs["color"] = properties[:, 1]
        graph.vs["label_size"] = properties[:, 2]

        edges = [
            [(i, j), entry.str()]
            for (i, row) in enumerate(self.rate_fn_matrix)
            for (j, entry) in enumerate(row)
            if entry is not None
        ]
        graph.add_edges([edge for edge, _ in edges])
        graph.es["label"] = [label for _, label in edges]
        graph.es["label_size"] = 7

        visual_style = {
            # "edge_curved": 0,
            "background": None,
            "edge_label": None if small_size else graph.es["label"],
            "edge_width": 1,
            "edge_background": "white" if transparent else palette[2],
            "vertex_label": None if small_size else graph.vs["label"],
            "bbox": (200, 200),
            "layout": graph.layout("kk"),
        }

        if target_ax is not None:
            if not transparent:
                target_ax.set_facecolor(palette[2])
            plot(graph, target=target_ax, **visual_style)
            return

        fig, ax = plt.subplots()
        ax.set_aspect(1)

        if not transparent:
            ax.set_facecolor(palette[2])
            fig.set_facecolor(palette[2])

        plot(graph, target=ax, **visual_style)

        if save:
            plt.savefig(fname, dpi=180, bbox_inches="tight", pad_inches=0)
            return

        plt.show()

    def __hash__(self) -> int:
        return hash(
            (
                *(tuple(row) for row in self.rate_fn_matrix),
                *tuple(self.activity_weights),
            )
        )

    def hash(self) -> str:
        return hex((self.__hash__() + (1 << 64)) % (1 << 64))

    @staticmethod
    def dummy():
        return PromoterModel([[]])
