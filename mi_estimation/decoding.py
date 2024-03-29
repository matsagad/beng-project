from models.model import PromoterModel
from mi_estimation.estimator import MIEstimator
from nptyping import NDArray, Shape, Float
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import confusion_matrix
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import GridSearchCV, HalvingGridSearchCV, train_test_split
from sklearn.neural_network import MLPClassifier
from typing import Any, Dict, Tuple
import numpy as np
import os
import warnings


class DecodingEstimator(MIEstimator):
    classifiers = ["svm", "decision_tree", "random_forest", "naive_bayes"]
    default_classifier_cfgs = {
        "svm": {
            "classifier": svm.SVC(),
            "params": {
                "kernel": ["rbf"],
                "C": np.logspace(-3, 3, 8),
                "gamma": np.logspace(-3, 3, 8),
            },
        },
        "decision_tree": {
            "classifier": DecisionTreeClassifier(random_state=0),
            "params": {
                "criterion": ["gini", "entropy"],
                "ccp_alpha": [0.1, 0.01, 0.001],
                "max_depth": [16, 64, 128],
                "min_samples_leaf": [1, 2, 4],
                "min_samples_split": [2, 4, 8],
            },
        },
        "random_forest": {
            "classifier": RandomForestClassifier(random_state=0),
            "params": {
                "criterion": ["gini", "entropy"],
                "ccp_alpha": [0.1, 0.01, 0.001],
                "max_depth": [16, 64, 128],
                "min_samples_leaf": [1, 2, 4],
                "min_samples_split": [2, 4, 8],
                "n_estimators": [10],
            },
        },
        "naive_bayes": {
            "classifier": GaussianNB(),
            "params": {
                "var_smoothing": np.logspace(-11, 0, num=100),
            },
        },
        "sgd": {
            "classifier": SGDClassifier(max_iter=1000),
            "params": {
                "alpha": [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3],
                "loss": ["hinge", "log_loss", "modified_huber"],
                "penalty": ["l2"],
            },
        },
        "mlp": {
            "classifier": MLPClassifier(max_iter=1000),
            "params": {
                "hidden_layer_sizes": [(32, 32, 32), (32, 64, 32), (128,)],
                "activation": ["tanh", "relu"],
                "solver": ["sgd", "adam"],
                "alpha": np.logspace(-4, -1, 10),
                "learning_rate": ["constant", "adaptive"],
            },
        },
    }
    RANDOM_STATE = 1

    def __init__(
        self,
        origin: int,
        interval: int,
        classifier_name: str = "svm",
        replicates: int = 1,
        classifier: any = None,
        classifier_params: Dict[str, Any] = None,
    ):
        self.origin = origin
        self.interval = interval

        if classifier_name not in self.default_classifier_cfgs:
            classifier_name = "svm"

        if classifier is not None:
            self.classifier = classifier
            self.classifier_params = classifier_params
            return

        cfg = self.default_classifier_cfgs[classifier_name]
        self.classifier = cfg["classifier"]
        self.classifier_params = cfg["params"]
        self.parallel = False

        self.replicates = replicates
        self.variance_threshold = 0.05

        # Hide runtime warnings from parallel backend jobs
        os.environ["PYTHONWARNINGS"] = "ignore::RuntimeWarning"

    def _split_classes(
        self,
        model: PromoterModel,
        trajectory: NDArray[Shape["Any, Any, Any, Any"], Float],
        remove_duplicates: bool = False,
    ) -> NDArray[Shape["Any, Any, Any"], Float]:
        CLASS_AXIS = 1
        trajectory = np.moveaxis(trajectory, CLASS_AXIS, 0)

        activity_weights = model.activity_weights / np.sum(model.activity_weights)
        rich_states = []
        states = []

        for env_class in trajectory:
            # Sum probabilities of the active states
            rich_trajectory = np.sum(
                activity_weights
                * env_class[self.origin - self.interval : self.origin, :, :],
                axis=2,
            ).T
            rich_states.extend(rich_trajectory)

            stress_trajectory = np.sum(
                activity_weights
                * env_class[self.origin + 1 : self.origin + self.interval + 1, :, :],
                axis=2,
            ).T
            states.append(
                np.unique(stress_trajectory, axis=0)
                if remove_duplicates
                else stress_trajectory
            )

        if remove_duplicates:
            rich_states = np.unique(rich_states, axis=0)
            samples_per_class = min(len(state) for state in states + [rich_states])
            for i, state in enumerate(states):
                states[i] = state[:samples_per_class]
        else:
            rich_states = np.array(rich_states)
            samples_per_class = len(states[0])

        rich_samples = rich_states[
            np.random.RandomState().choice(
                len(rich_states), samples_per_class, replace=False
            )
        ]

        return np.array([rich_samples] + states)

    def _flip_random(
        self, Xs: NDArray[Shape["Any, Any"], Float], prob: float = 0.05
    ) -> NDArray[Shape["Any, Any"], Float]:
        bit_mask = np.random.binomial(size=(Xs.shape), n=1, p=prob)
        return np.array(Xs != bit_mask, dtype=float)

    def _smoothen(self, Xs: NDArray[Shape["Any, Any"], Float], window_size: int = 6):
        smooth_Xs = np.zeros(Xs.shape)

        for i in range(smooth_Xs.shape[1]):
            smooth_Xs[:, i] = np.int8(
                np.average(Xs[:, i : i + window_size], axis=1) >= 0.5
            )
        return smooth_Xs

    def _estimate(
        self,
        data: NDArray[Shape["Any, Any, Any"], Float],
        n_bootstraps: int = 25,
        c_interval: int = [0.25, 0.75],
        verbose: bool = False,
        halving: bool = True,
        add_noise: bool = False,
        smoothen: bool = False,
        return_std: bool = False,
    ) -> float | Tuple[float, float]:
        """
        The MI estimation process is adapted from the method of Granados, Pietsch, et al,
        Proc Nat Acad Sci USA 115 (2008) 6088. It has been modified to suit the setting of
        the study and allow different classifiers to be used.
        """
        # Set-up data
        num_classes, num_samples, ts_duration = data.shape
        num_cells = num_samples // self.replicates

        _add_noise = self._flip_random if add_noise else lambda x: x
        _smoothen = self._smoothen if smoothen else lambda x: x
        pre_process = lambda x: _add_noise(_smoothen(x))

        Xs = np.split(
            pre_process(np.hstack(data)).reshape((-1, ts_duration)),
            num_cells,
        )
        ys = np.split(
            np.tile(np.arange(num_classes), num_samples),
            num_cells,
        )
        np_rs = np.random.RandomState()

        ## Split data into validation/training/testing ~ 15/70/15 split
        data_split = (0.15, 0.70, 0.15)

        _fst_split = data_split[0]
        _snd_split = data_split[2] / (data_split[1] + data_split[2])

        fst_split_count = int(_fst_split * num_cells)
        snd_split_count = int(_snd_split * (num_cells - fst_split_count))

        np_rs.shuffle(Xs)
        _X_val, _X = np.split(Xs, [fst_split_count])
        _y_val, _y = np.split(ys, [fst_split_count])
        X_val, y_val = np.vstack(_X_val), np.hstack(_y_val)

        ## If all features have a variance below a threshold, then prematurely return
        #  an MI of zero. This avoids division by zero within PCA and other calculations
        #  as well as speeding up estimation times.
        if np.all(
            np.var(MinMaxScaler((-1, 1)).fit_transform(X_val), axis=0)
            < self.variance_threshold
        ):
            return 0.0, -1.0

        # Tune pipeline hyperparameters
        num_samples = len(y_val)
        n_PCA_range = np.linspace(0.95, 0.99, 5)
        _prefix = "classifier__"
        params = [
            {"project__n_components": n_PCA_range},
            {_prefix + prop: value for (prop, value) in self.classifier_params.items()},
        ]

        ## The pipeline
        pipe = Pipeline(
            [
                ("rescale", StandardScaler()),
                ("project", PCA()),
                ("classifier", self.classifier),
            ]
        )

        ## Grid search
        grid_pipeline = (HalvingGridSearchCV if halving else GridSearchCV)(
            pipe,
            params,
            n_jobs=(-1 if self.parallel else 1),
            cv=5,
            # resource="n_samples",
            # min_resources=num_samples // 4,
            # factor=2,
            # error_score="raise",
            # scoring="f1_micro"
        )
        ## Suppress runtime warnings (e.g. division by zero)
        #  After all, these naturally tend to low MI scores as classifier performance
        #  is poor - which is in line with our expectation for low variance features.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            grid_pipeline.fit(X_val, y_val)
        if verbose:
            print(grid_pipeline.best_estimator_)
        pipe.set_params(**grid_pipeline.best_params_)

        # Find mutual information for each bootstrapped dataset
        mi = np.empty(n_bootstraps)
        for i in range(n_bootstraps):
            # Get random train-test partition
            perm = np_rs.permutation(len(_y))
            X_train, X_test = (
                np.vstack(arr) for arr in np.split(_X[perm], [snd_split_count])
            )
            y_train, y_test = (
                np.hstack(arr) for arr in np.split(_y[perm], [snd_split_count])
            )

            # Estimate mutual information
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                pipe.fit(X_train, y_train)
                y_pred = pipe.predict(X_test)

            p_y = 1 / num_classes
            p_yhat_given_y = confusion_matrix(y_test, y_pred, normalize="true")

            p_yhat = np.sum(p_y * p_yhat_given_y, 0)
            h_yhat = -np.sum(p_yhat[p_yhat > 0] * np.log2(p_yhat[p_yhat > 0]))

            h_yhat_given_y = -p_y * np.sum(
                p_yhat_given_y[p_yhat_given_y > 0]
                * np.log2(p_yhat_given_y[p_yhat_given_y > 0])
            )

            mi[i] = h_yhat - h_yhat_given_y

        # Summary statistics - median and confidence intervals
        ci_low, ci_high = c_interval
        sorted_mi = np.sort(mi)
        inter_quartile = sorted_mi[
            int(ci_low * n_bootstraps) : int(ci_high * n_bootstraps)
        ]

        mean_mi = np.mean(inter_quartile)
        std_mi = np.std(inter_quartile)

        if verbose:
            ci_low_value = sorted_mi[int(ci_low * n_bootstraps)]
            ci_high_value = sorted_mi[int(ci_high * n_bootstraps)]
            print(f"median MI= {mean_mi:.2f} [{ci_low_value:.2f}, {ci_high_value:.2f}]")

        if return_std:
            return mean_mi, std_mi

        return mean_mi

    def estimate(
        self,
        model: PromoterModel,
        trajectory: NDArray[Shape["Any, Any, Any, Any"], Float],
        verbose: bool = False,
        halving: bool = True,
        add_noise: bool = False,
        smoothen: bool = False,
        return_std: bool = False,
    ) -> float | Tuple[float, float]:
        data = self._split_classes(model, trajectory)
        return self._estimate(
            data,
            verbose=verbose,
            halving=halving,
            add_noise=add_noise,
            smoothen=smoothen,
            return_std=return_std,
        )
