from models.model import PromoterModel
from models.rates.function import RateFunction


class SimpleModel(PromoterModel):
    def __init__(self, a: float, b: float):
        rate_fn_matrix = [
            [None, RateFunction.linear(a, 0)],
            [RateFunction.constant(b), None]
        ]
        super().__init__(rate_fn_matrix)
    
