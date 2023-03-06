from typing import Dict
from evolution.genetic.runner import GeneticRunner
from evolution.genetic.operators.mutation import MutationOperator
from evolution.genetic.operators.selection import SelectionOperator
from evolution.genetic.operators.crossover import CrossoverOperator
from utils.process import get_tf_data
from jobs.job import Job
import pickle
from math import exp


class GeneticAlgorithmJob(Job):
    def __init__(self, verbose: bool):
        self.default_args = {
            "states": 2,
            "population": 10,
            "iterations": 10,
            "fix_states": False,
            "reversible": True,
            "one_active_state": True,
            "n_processors": 1,
            "cache_folder": "cache",
            "output_file": "models.dat",
        }
        self.name = "Genetic Algorithm"
        super().__init__(verbose)

    @Job.timed
    def run(self, args: Dict[str, any]) -> None:
        """
        Args:
          states            Number of states the model population starts with
          population        Number of models to consider in each generation
          iterations        Number of generations to run
          fix_states        Flag for if states should be fixed (False)
          reversible        Flag for if reactions should be reversible (True)
          one_active_state  Flag for if models should have only one active state (True)
          n_processors      Number of processors to parallelise model evaluation
          cache_folder      Path to cache folder where data may be cached
          output_file       Name of file to output model data
        """
        _args = self.default_args

        for arg_name, arg_value in args.items():
            if arg_name in _args:
                _args[arg_name] = arg_value

        if self.verbose:
            print(f"[{self.name}] parameters used:")
            print("\n".join(f"\t{name}: {value}" for (name, value) in _args.items()))

        data, _, _, _ = get_tf_data(cache_folder=_args["cache_folder"])

        mutations = [
            MutationOperator.edit_edge,
            MutationOperator.add_edge,
            MutationOperator.flip_tf,
            MutationOperator.add_noise,
        ]

        if bool(_args["fixed_states"]):
            crossover = CrossoverOperator.subgraph_swap
            # Penalise models with many states (regardless of edge count - a TODO)
            _MAX_MI, arb_k = 2, 6
            scale_fitness = lambda model, mi: max(
                mi - _MAX_MI / (1 + arb_k * exp(arb_k - model.num_states)), 0
            )
        else:
            crossover = CrossoverOperator.one_point_triangular_row_swap
            scale_fitness = lambda _, mi: mi

        select = SelectionOperator.roulette_wheel
        runner = GeneticRunner(data, mutations, crossover, select, scale_fitness)

        mg_params = {
            "reversible": bool(_args["reversible"]),
            "one_active_state": bool(_args["one_active_state"]),
        }

        models = runner.run(
            states=int(_args["states"]),
            population=int(_args["population"]),
            iterations=int(_args["iterations"]),
            n_processors=int(_args["n_processors"]),
            model_generator_params=mg_params,
            verbose=True,
            debug=True,
        )

        with open(_args["output_file"], "wb") as f:
            pickle.dump(models, f)
            print(f"Cached best models at {f}.")
