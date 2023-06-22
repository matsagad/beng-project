from typing import List, Tuple
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

        self.num_states = len(rate_fn_matrix)
        self.num_edges = sum(sum(map(bool, row)) for row in self.rate_fn_matrix)

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
        self,
        exogenous_data: NDArray[Shape["Any, Any, Any, Any"], Float],
        axes_permutation: Tuple[int, int, int] = (2, 0, 1),
    ) -> NDArray[Shape["Any, Any, Any, Any, Any"], Float]:
        """
        Args:
            exogenous_data: an array of exogenous data with dimensions:
                            # of classes, # of TFs, batch size, # of times

        Returns:
            (By default) an array of generator matrices with dimensions:
            # of times, # of classes, batch size, # of states, # of states
        """
        # Set the no. of TFs to be the leading axis and  permute the rest of the axes.
        relative_permutation = np.array([0, 2, 3])[list(axes_permutation)]
        tf_iterable_data = np.transpose(exogenous_data, (1, *relative_permutation))

        generator = np.zeros(
            (*tf_iterable_data.shape[1:], self.num_states, self.num_states),
            dtype=np.float32,
        )

        for i, row in enumerate(self.rate_fn_matrix):
            for j, rate_fn in enumerate(row):
                if rate_fn:
                    generator[:, :, :, i, j] = rate_fn.evaluate(tf_iterable_data)

        # Default generator shape is: no. times, no. classes, batch size, no, states, no. states

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

    def get_matrix_exp_iterable(
        self, exogenous_data: NDArray[Shape["Any, Any, Any, Any"], Float], tau: float
    ) -> NDArray[Shape["Any, Any, Any, Any, Any"], Float]:
        # Lazily evaluate matrix exponentials to save memory.
        Q = tau * self.get_generator(exogenous_data)
        scale = 1 << max(0, int((np.log2(norm(Q)))))
        return (np.linalg.matrix_power(expm(Q_t / scale), scale) for Q_t in Q)

    def visualise(
        self,
        with_rates: bool = True,
        discrete: bool = False,
        save: bool = False,
        fname: str = "cache/model.png",
        white_bg: bool = True,
        transparent: bool = False,
        small_size: bool = False,
        target_ax: any = None,
        no_pgf: bool = False,
    ) -> None:
        from igraph import Graph, plot
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap, rgb2hex
        from matplotlib import rcParams

        # rcParams["text.usetex"] = True
        # plt.rc('text', usetex=True)
        # plt.rc("text.latex", preamble=r"\usepackage{amsmath}")
        rcParams["pgf.preamble"] = r"\usepackage{amsmath} \usepackage{xcolor}"

        # Colors are from the "curiosities" palette by sukinapan!
        palette = ["#00b9be", "#ffb0a3", "#ffeecc", "#46425e"]
        bg = ("#ffffff00" if transparent else "white") if white_bg else palette[2]

        num_colors = 10
        cmap = LinearSegmentedColormap.from_list(
            "redToGreen", [(0, palette[1]), (1, palette[0])], N=num_colors
        )

        graph = Graph(directed=True)
        graph.add_vertices(self.num_states)

        active_states = self.activity_weights > 0
        activity_weights = self.activity_weights[active_states] / np.sum(
            self.activity_weights
        )

        properties = np.zeros((self.num_states, 4), dtype=object)
        properties[active_states, 0] = [
            (
                f"$A_{{ {i} }}$"
                if discrete
                else f"$\\underset{{ {activity_weights[i]:.2f} }}{{ A_{{ {i} }} }}$"
            )
            for i, weight in enumerate(activity_weights)
        ]

        properties[active_states, 1] = [
            palette[0] if discrete else rgb2hex(cmap(weight))
            for weight in activity_weights
        ]
        properties[active_states, 3] = [
            cmap(0.0) if discrete or weight > 0.5 else cmap(1.0)
            for weight in activity_weights
        ]

        properties[active_states, 2] = 14 if discrete else 12
        properties[~active_states, 0] = [f"$I_{i}$" for i in range(sum(~active_states))]
        properties[~active_states, 1] = palette[1]
        properties[~active_states, 2] = 14

        graph.vs["label"] = properties[:, 0]
        graph.vs["color"] = properties[:, 1]
        graph.vs["label_size"] = properties[:, 2]

        edges = [
            [(i, j), rate_fn.str(with_rates=with_rates)]
            for (i, row) in enumerate(self.rate_fn_matrix)
            for (j, rate_fn) in enumerate(row)
            if rate_fn is not None
        ]
        graph.add_edges([edge for edge, _ in edges])
        graph.es["label"] = [label for _, label in edges]
        graph.es["label_size"] = 12

        graph.es["color"] = graph.vs["frame_color"] = "#46425e"

        if hasattr(self, "color_matrix"):
            graph.es["color"] = graph.es["label_color"] = [
                color for row in self.color_matrix for color in row if color is not None
            ]
            graph.vs["frame_color"] = [
                max(set([c for c in row if c is not None]), key=row.count)
                for row in self.color_matrix
            ]

        visual_style = {
            # "edge_curved": 0,
            "background": None,
            "edge_label": None if small_size else graph.es["label"],
            "edge_width": 1.25,
            "edge_background": bg,
            "edge_arrow_width": 0.75,
            "vertex_label": None if small_size else graph.vs["label"],
            "vertex_frame_width": 1.25,
            "bbox": (200, 200),
            "layout": graph.layout("kk"),
        }

        if target_ax is not None:
            if not transparent:
                target_ax.set_facecolor(bg)
            plot(graph, target=target_ax, **visual_style)
            return

        fig, ax = plt.subplots()
        ax.set_aspect(1)

        if not transparent:
            ax.set_facecolor(bg)
            fig.set_facecolor(bg)

        plot(graph, target=ax, **visual_style)

        if save:
            if no_pgf:
                plt.savefig(fname, dpi=180, bbox_inches="tight", pad_inches=0)
                return
            plt.savefig(
                fname, dpi=180, bbox_inches="tight", pad_inches=0, backend="pgf"
            )
            return

        plt.show()

    def __hash__(self) -> int:
        return hash(
            (
                *(tuple(filter(None, row)) for row in self.rate_fn_matrix),
                *tuple(self.activity_weights),
            )
        )

    def hash(self, short: bool = False) -> str:
        _hash = hex((self.__hash__() + (1 << 64)) % (1 << 64))
        return _hash[2:8] if short else _hash

    @staticmethod
    def dummy():
        return PromoterModel([[]])
