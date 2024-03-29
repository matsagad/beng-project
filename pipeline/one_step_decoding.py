from mi_estimation.decoding import DecodingEstimator
from models.model import PromoterModel
from nptyping import NDArray, Shape, Float
from pipeline.pipeline import Pipeline
from ssa.one_step import OneStepSimulator


class OneStepDecodingPipeline(Pipeline):
    # Constants derived from data
    FIXED_TIME_DELTA = 2.525
    FIXED_ORIGIN = 47
    FIXED_INTERVAL = 30

    def __init__(
        self,
        exogenous_data: NDArray[Shape["Any, Any, Any, Any"], Float],
        tau: float = FIXED_TIME_DELTA,
        realised: bool = True,
        replicates: int = 1,
        origin: int = FIXED_ORIGIN,
        interval: int = FIXED_INTERVAL,
        classifier_name: str = "svm",
    ):
        super().__init__(
            simulator=OneStepSimulator(
                exogenous_data, tau, realised=realised, replicates=replicates
            ),
            estimator=DecodingEstimator(origin, interval, classifier_name, replicates=replicates),
        )
        # Optimisation: don't simulate trajectories past interval
        self.simulator.num_times = origin + interval

    def evaluate(self, model: PromoterModel, verbose: bool = False) -> float:
        return super().estimateMI(model, verbose)

    def set_parallel(self) -> None:
        self.estimator.parallel = True
