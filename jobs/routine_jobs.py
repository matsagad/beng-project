from typing import Dict
from evolution.genetic.runner import GeneticRunner
from evolution.genetic.operators.mutation import MutationOperator
from evolution.genetic.operators.selection import SelectionOperator
from evolution.genetic.operators.crossover import CrossoverOperator
from evolution.genetic.penalty import ModelPenalty
from utils.process import get_tf_data
from jobs.job import Job
import pickle


class GeneticAlgorithmJob(Job):
    def __init__(self, verbose: bool):
        self.default_args = {
            "states": 2,
            "population": 10,
            "iterations": 10,
            "elite_ratio": 0.2,
            "fixed_states": "False",
            "no_penalty": "False",
            "penalty_coeff": 8.0,
            "reversed_penalty": "False",
            "target_states": -1,
            "p_edge": 0.5,
            "reversible": "True",
            "one_active_state": "True",
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
          elite_ratio       Percentage of population that are kept as elites
          fixed_states      Flag for if states should be fixed (False)
          no_penalty        Flag for if models should be penalised
          penalty_coeff     Parameter for penalising models
          reversed_penalty  Flag for if smaller models should be penalised
          target_states     Target number of states for models (-1)
          p_edge            Probability of edge connections at init population (0.5)
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

        if _args["one_active_state"] == "False":
            mutations.append(MutationOperator.flip_activity)
            mutations.append(MutationOperator.add_activity_noise)

        if _args["fixed_states"] == "False":
            crossover = CrossoverOperator.subgraph_swap
            penalty_coeff = float(_args["penalty_coeff"])
            if _args["no_penalty"] == "True":
                scale_fitness = lambda _, mi: mi
            elif _args["reversed_penalty"] == "True":
                scale_fitness = ModelPenalty.reversed_state_penalty(m=penalty_coeff)
            elif int(_args["target_states"]) < 2:
                scale_fitness = ModelPenalty.state_penalty(m=penalty_coeff)
            else:
                scale_fitness = ModelPenalty.balanced_state_penalty(
                    target_state=int(_args["target_states"]),
                    m=float(_args["penalty_coeff"]),
                )
        else:
            crossover = CrossoverOperator.one_point_triangular_row_swap
            scale_fitness = lambda _, mi: mi

        select = SelectionOperator.roulette_wheel
        runner = GeneticRunner(data, mutations, crossover, select, scale_fitness)

        mg_params = {
            "reversible": _args["reversible"] == "True",
            "one_active_state": _args["one_active_state"] == "True",
            "p_edge": float(_args["p_edge"]),
        }

        models, stats = runner.run(
            states=int(_args["states"]),
            population=int(_args["population"]),
            iterations=int(_args["iterations"]),
            elite_ratio=float(_args["elite_ratio"]),
            n_processors=int(_args["n_processors"]),
            model_generator_params=mg_params,
            verbose=True,
            debug=True,
        )

        with open(_args["output_file"], "wb") as f:
            pickle.dump(models, f)
            print(f"Cached best models at {f}.")
            pickle.dump(stats, "stats_" + f)
            print(f"Cached GA runner stats at stats_{f}.")
