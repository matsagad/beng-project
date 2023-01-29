from models.model import PromoterModel
from models.rates.function import RateFunction


class SimpleModel(PromoterModel):
    def __init__(self, tf_index: int, a: float, b: float):
        rate_fn_matrix = [
            [None, RateFunction.Linear(a, tf_index)],
            [RateFunction.Constant(b), None],
        ]
        super().__init__(rate_fn_matrix)
