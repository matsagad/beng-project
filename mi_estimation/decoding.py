from models.model import PromoterModel
from mi_estimation.estimator import MIEstimator
from nptyping import NDArray, Shape, Float
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
from typing import Any, Dict
import numpy as np


class DecodingEstimator(MIEstimator):
    classifiers = ["svm", "decision_tree", "random_forest", "naive_bayes"]
    default_classifier_cfgs = {
        "svm": {
            "classifier": svm.SVC(probability=True),
            "params": {
                "kernel": ["rbf"],
                "C": np.logspace(-3, 3, 12),
                "gamma": np.logspace(-3, 3, 12),
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
    }

    def __init__(
        self,
        origin: int,
        interval: int,
        classifier_name: str = "svm",
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
        print(self.classifier)
        print(classifier_name)

    def _split_classes(
        self,
        model: PromoterModel,
        trajectory: NDArray[Shape["Any, Any, Any, Any"], Float],
    ) -> float:
        CLASS_AXIS = 1
        trajectory = np.moveaxis(trajectory, CLASS_AXIS, 0)

        active_states = model.active_states
        rich_states = []
        states = []

        for env_class in trajectory:
            # Sum probabilities of the active states
            rich_trajectory = np.sum(
                env_class[self.origin - self.interval : self.origin, :, active_states],
                axis=2,
            ).T
            if rich_state is None:
                rich_state = rich_trajectory
            else:
                rich_state += rich_trajectory

            stress_trajectory = np.sum(
                env_class[self.origin : self.origin + self.interval, :, active_states],
                axis=2,
            ).T
            states.append(stress_trajectory)

        avg_rich_state = rich_state / len(trajectory)
        return [[avg_rich_state] + states]

    def _flatten(self, X):
        return X.reshape((X.shape[0], -1))

    def _estimate(
        self,
        data: int,
        overtime: bool = True,
        n_bootstraps: int = 25,
        c_interval: int = [0.25, 0.75],
        verbose: bool = False,
    ) -> float:
        """
        The MI estimation process is adapted from the method of Granados, Pietsch, et al,
        Proc Nat Acad Sci USA 115 (2008) 6088. It has been modified to allow different
        classifiers to be used.
        """
        for i in range(len(data)):
            for j in range(len(data[i])):
                data[i][j] = self._flatten(data[i][j])

        Xos = [0] * len(data)
        ys = [0] * len(data)
        n_classes = len(data[0])

        for i in range(len(data)):
            Xos[i] = np.vstack([timeseries for timeseries in data[i]])
            ys[i] = np.hstack(
                [
                    j * np.ones(timeseries.shape[0])
                    for j, timeseries in enumerate(data[i])
                ]
            )

        durations = np.arange(1, Xos[0].shape[1] + 1) if overtime else [Xos[0].shape[1]]
        duration = durations[-1]
        grid_pipelines = [0] * len(Xos)

        for k in range(len(Xos)):
            if verbose:
                print("duration of time series is", duration)
            X = Xos[k][:, :duration]

            # Hyperparameter space
            nPCArange = range(1, X.shape[1] + 1) if X.shape[1] < 5 else [3, 4, 5]

            _prefix = "classifier__"
            params = [
                {"project__n_components": nPCArange},
                {
                    _prefix + prop: value
                    for (prop, value) in self.classifier_params.items()
                },
            ]

            # Pipeline
            pipe = Pipeline(
                [
                    ("rescale", StandardScaler()),
                    ("project", PCA()),
                    ("classifier", self.classifier),
                ]
            )

            # Tune pipeline hyperparameters
            grid_pipelines[k] = HalvingGridSearchCV(pipe, params, n_jobs=-1, cv=5)
            grid_pipelines[k].fit(X, ys[k])
            if verbose:
                print(grid_pipelines[k].best_estimator_)
            pipe.set_params(**grid_pipelines[k].best_params_)

        # Find mutual information for each bootstrapped dataset
        mi = np.empty(n_bootstraps)
        for i in range(n_bootstraps):
            p = np.random.RandomState().permutation(len(ys[0]))
            test_size = np.round(0.3 * len(ys[0])).astype("int")
            y_pred = np.zeros((test_size, n_classes))

            for k in range(len(ys)):
                X, y = Xos[k][p], ys[k][p]
                test_size = np.round(0.3 * len(y)).astype("int")
                X_train, y_train = X[test_size:], y[test_size:]
                X_test, y_test = X[:test_size], y[:test_size]

                # Run classifier using optimal parameters
                pipe.fit(X_train, y_train)
                y_preds = pipe.predict_proba(X_test)
                y_pred += y_preds

            y_predict = np.argmax(y_pred, axis=1)

            # Estimate mutual information
            p_y = 1 / n_classes
            p_yhat_given_y = confusion_matrix(y_test, y_predict, normalize="true")

            p_yhat = np.sum(p_y * p_yhat_given_y, 0)
            h_yhat = -np.sum(p_yhat[p_yhat > 0] * np.log2(p_yhat[p_yhat > 0]))
            log2_p_yhat_given_y = np.ma.log2(p_yhat_given_y).filled(0)
            h_yhat_given_y = -np.sum(
                p_y * np.sum(p_yhat_given_y * log2_p_yhat_given_y, 1)
            )
            mi[i] = h_yhat - h_yhat_given_y

        # Summary statistics - median and confidence intervals
        ci_low, ci_high = c_interval
        sorted_mi = np.sort(mi)

        mean_mi = np.mean(
            sorted_mi[int(ci_low * n_bootstraps) : int(ci_high * n_bootstraps)]
        )

        if verbose:
            ci_low_value = sorted_mi[int(ci_low * n_bootstraps)]
            ci_high_value = sorted_mi[int(ci_high * n_bootstraps)]
            print(f"median MI= {mean_mi:.2f} [{ci_low_value:.2f}, {ci_high_value:.2f}]")

        return mean_mi

    def estimate(
        self,
        model: PromoterModel,
        trajectory: NDArray[Shape["Any, Any, Any, Any"], Float],
    ) -> float:
        data = self._split_classes(model, trajectory)
        return self._estimate(data)
