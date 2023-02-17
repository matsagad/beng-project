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
from sklearn.model_selection import GridSearchCV, HalvingGridSearchCV, train_test_split
from typing import Any, Dict
import numpy as np


class DecodingEstimator(MIEstimator):
    classifiers = ["svm", "decision_tree", "random_forest", "naive_bayes"]
    default_classifier_cfgs = {
        "svm": {
            "classifier": svm.SVC(),
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
    RANDOM_STATE = 1

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
        print(classifier_name)

    def _split_classes(
        self,
        model: PromoterModel,
        trajectory: NDArray[Shape["Any, Any, Any, Any"], Float],
        remove_duplicates: bool = False,
    ) -> NDArray[Shape["Any, Any, Any"], Float]:
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
            rich_states.extend(rich_trajectory)

            stress_trajectory = np.sum(
                env_class[
                    self.origin + 1 : self.origin + self.interval + 1, :, active_states
                ],
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
            np.random.choice(len(rich_states), samples_per_class, replace=False)
        ]

        return np.array([rich_samples] + states)

    def _flip_random(
        self, Xs: NDArray[Shape["Any, Any"], Float], prob: float = 0.05
    ) -> NDArray[Shape["Any, Any"], Float]:
        bit_mask = np.random.binomial(size=(Xs.shape), n=1, p=prob)
        return np.array(Xs != bit_mask, dtype=float)

    def _estimate(
        self,
        data: NDArray[Shape["Any, Any, Any"], Float],
        n_bootstraps: int = 25,
        c_interval: int = [0.25, 0.75],
        verbose: bool = False,
        add_noise: bool = True,
    ) -> float:
        """
        The MI estimation process is adapted from the method of Granados, Pietsch, et al,
        Proc Nat Acad Sci USA 115 (2008) 6088. It has been modified to suit the setting of
        the study and allow different classifiers to be used.
        """
        # Set-up data
        num_classes, num_samples, ts_duration = data.shape
        Xs, ys = np.vstack(data), np.repeat(np.arange(num_classes), num_samples)

        if add_noise:
            Xs = self._flip_random(Xs)

        ## Split data into validation/training/testing ~ 20/60/20 split
        data_split = (0.20, 0.65, 0.15)

        fst_split = data_split[0]
        snd_split = data_split[2] / (data_split[1] + data_split[2])

        _X, X_val, _y, y_val = train_test_split(
            Xs, ys, test_size=fst_split, stratify=ys
        )

        # Tune pipeline hyperparameters
        num_samples = len(y_val)
        n_PCA_range = range(1, 1 + min(num_samples, ts_duration))
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
        grid_pipeline = HalvingGridSearchCV(
            pipe,
            params,
            n_jobs=None,
            cv=5,
            resource="n_samples",
            # min_resources=num_samples // 4,
            # factor=2,
            # error_score="raise",
        )
        grid_pipeline.fit(X_val, y_val)
        if verbose:
            print(grid_pipeline.best_estimator_)
        pipe.set_params(**grid_pipeline.best_params_)

        # Find mutual information for each bootstrapped dataset
        mi = np.empty(n_bootstraps)
        for i in range(n_bootstraps):
            X_train, X_test, y_train, y_test = train_test_split(
                _X, _y, test_size=snd_split, stratify=_y
            )

            # Estimate mutual information
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
