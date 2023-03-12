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
        self.active_states = np.zeros(len(rate_fn_matrix), dtype=bool)
        self.active_states[0] = True
        # (should be changed for models with more than one active
        # state, e.g. competing activator)

        self.num_states = len(rate_fn_matrix)
        self.num_edges = sum(sum(map(bool, row)) for row in self.rate_fn_matrix)

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
        graph.es["label_size"] = 8

        visual_style = {
            # "edge_curved": 0,
            "background": None,
            "edge_label": None if small_size else graph.es["label"],
            "edge_width": 1,
            "edge_background": "white" if transparent else palette[2],
            "vertex_label": None if small_size else graph.vs["label"],
            "bbox": (100, 100),
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
                *tuple(self.active_states),
            )
        )

    def hash(self) -> str:
        return hex((self.__hash__() + (1 << 64)) % (1 << 64))

    @staticmethod
    def dummy():
        return PromoterModel([[]])