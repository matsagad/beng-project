from models.model import PromoterModel
from models.rates.function import RateFunction as RF


class Preset:
    """
    A collection of notable model configurations previously explored.
    """

    @staticmethod
    def simple(tf_index: int, a: float, b: float) -> PromoterModel:
        return PromoterModel(
            rate_fn_matrix=[
                [None, RF.Linear(a, tf_index)],
                [RF.Constant(b), None],
            ]
        ).with_active_states([1])

    @staticmethod
    def simple_hill(tf_index: int, a: float, b: float, c: float) -> PromoterModel:
        return PromoterModel(
            rate_fn_matrix=[
                [None, RF.Hill(a, b, tf_index)],
                [RF.Constant(c), None],
            ]
        ).with_active_states([1])

    @staticmethod
    def competing_activator(
        tf_fst: int, tf_snd: int, a: float, b: float
    ) -> PromoterModel:
        return PromoterModel(
            rate_fn_matrix=[
                [None, None, RF.Constant(a)],
                [None, None, RF.Constant(b)],
                [RF.Linear(a, tf_fst), RF.Linear(b, tf_snd), None],
            ]
        ).with_active_states([0, 1])

    @staticmethod
    def extended_competing_activator(
        tf_fst: int, tf_snd: int, tf_thd: int, a: float, b: float, c: float
    ) -> PromoterModel:
        return PromoterModel(
            rate_fn_matrix=[
                [None, None, None, RF.Constant(a)],
                [None, None, None, RF.Constant(b)],
                [None, None, None, RF.Constant(c)],
                [
                    RF.Linear(a, tf_fst),
                    RF.Linear(b, tf_snd),
                    RF.Linear(c, tf_thd),
                    None,
                ],
            ]
        ).with_active_states([0, 1, 2])
