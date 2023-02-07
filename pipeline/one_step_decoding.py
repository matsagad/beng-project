from mi_estimation.decoding import DecodingEstimator
from models.model import PromoterModel
from nptyping import NDArray, Shape, Float
from pipeline.pipeline import Pipeline
from ssa.one_step import OneStepSimulator


class OneStepDecodingPipeline(Pipeline):
    FIXED_TIME_DELTA = 2.5
    FIXED_ORIGIN = 49
    FIXED_INTERVAL = 20

    def __init__(
        self,
        exogenous_data: NDArray[Shape["Any, Any, Any, Any"], Float],
        tau: float = FIXED_TIME_DELTA,
        realised: bool = False,
        replicates: int = 1,
        origin: int = FIXED_ORIGIN,
        interval: int = FIXED_INTERVAL,
    ):
        super().__init__(
            simulator=OneStepSimulator(
                exogenous_data, tau, realised=realised, replicates=replicates
            ),
            estimator=DecodingEstimator(origin, interval),
        )

    def evaluate(self, model: PromoterModel, verbose: bool = False) -> float:
        return super().estimateMI(model, verbose)
