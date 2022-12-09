from typing import Callable, Dict
import numpy as np
import re


class RatesEquation:
    def __init__(
        self,
        reactants: Dict[str, float],
        products: Dict[str, float],
        rate_fn: Callable[[np.ndarray], float],
    ):
        self.reactants = reactants
        self.products = products
        self.elements = set().union(*(reactants.keys(), products.keys()))
        self.rate_fn = rate_fn
        self.v_rate_fn = np.vectorize(rate_fn)

    @staticmethod
    def parse_str(
        equation_str: str,
        rate_fn: Callable[[np.ndarray], float] = lambda _: 1,
        add_delim: str = "+",
        react_delim: str = "->",
    ) -> "RatesEquation":
        """Parses rate equation strings (e.g. A + 2B -> C).

        Args:
          equation_str  : string representing the rate equation
          rate_fn       : function that yields the rate coefficient with
                          respect to the state of the system
          add_delim     : addition delimeter
          react_delim   : reaction delimeter
        """
        reactants_str, products_str = equation_str.split(react_delim)[:2]
        reactants, products = dict(), dict()

        for (terms, terms_str) in (
            (reactants, reactants_str),
            (products, products_str),
        ):
            for term in terms_str.split(add_delim):
                symbol_args = [
                    char for char in re.split("((\d|\.)+)", term.strip()) if char
                ]
                count = 1 if not symbol_args[1:] else float(symbol_args[0])
                symbol = symbol_args[-1]
                if symbol not in terms:
                    terms[symbol] = 0
                terms[symbol] += count

        return RatesEquation(reactants, products, rate_fn)

    def __str__(self) -> str:
        return " -> ".join(
            " + ".join(
                ("" if count == 1 else f"{count}") + symbol
                for (symbol, count) in freq_map.items()
            )
            for freq_map in (self.reactants, self.products)
        )
