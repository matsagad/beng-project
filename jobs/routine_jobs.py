from evolution.genetic.runner import GeneticRunner
from evolution.genetic.operators.mutation import MutationOperator
from evolution.genetic.operators.selection import SelectionOperator
from evolution.genetic.operators.crossover import CrossoverOperator
from evolution.genetic.penalty import ModelPenalty
from evolution.novelty.runner import NoveltySearchRunner
from jobs.job import Job
from optimisation.particle_swarm import ParticleSwarm
from typing import Dict
from utils.process import get_tf_data
import pickle
import os
import sys
import signal


class GeneticAlgorithmJob(Job):
    def __init__(self, verbose: bool):
        self.default_args = {
            # Initial population
            "states": 2,
            "population": 10,
            "iterations": 10,
            "elite_ratio": 0.2,
            "p_edge": 0.5,
            # Penalty functions
            "target_states": -1,
            "penalty__active": "True",
            "penalty__coeff": 8.0,
            "penalty__reversed": "False",
            # Operators
            "fixed_states": "False",
            "selection": "tournament",
            "selection__replacement": "True",
            "selection__tournament_size": 3,
            # Constraints
            "reversible": "True",
            "one_active_state": "True",
            # I/O and hardware
            "n_processors": 1,
            "cache_folder": "cache",
            "initial_population": "",
            "output_file": "models.dat",
        }
        self.name = "Genetic Algorithm"

        # For data recovery
        self.runner = None
        self.cached_results = False
        signal.signal(signal.SIGTERM, self.on_interrupted)

        super().__init__(verbose)

    @Job.timed
    def run(self, args: Dict[str, any]) -> None:
        """
        Args:
          # Initial population
          states            Number of states the model population starts with
          population        Number of models to consider in each generation
          iterations        Number of generations to run
          elite_ratio       Percentage of population that are kept as elites
          p_edge            Probability of edge connections at init population (0.5)

          # Penalty functions
          target_states     Target number of states for models (-1)
          penalty__active   Flag for if models should be penalised
          penalty__coeff    Hyperparameter for penalising models
          penalty__reversed Flag for if smaller models should be penalised

          # Genetic operators
          fixed_states               Flag for if states should be fixed (False)
          selection                  Selection operator (tournament, roulette)
          selection__replacement     Flag for if selection should be done with replacement
          selection__tournament_size Number of models sampled if tournament selection operator is chosen

          # Constraints
          reversible        Flag for if reactions should be reversible (True)
          one_active_state  Flag for if models should have only one active state (True)

          # I/O and hardware
          n_processors          Number of processors to parallelise model evaluation
          cache_folder          Path to cache folder where data may be cached
          initial_population    Name of file containing models to initialise the run
          output_file           Name of file to output model data
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
            penalty_coeff = float(_args["penalty__coeff"])
            if _args["penalty__active"] == "False":
                scale_fitness = lambda _, mi: mi
            elif _args["penalty__reversed"] == "True":
                scale_fitness = ModelPenalty.reversed_state_penalty(m=penalty_coeff)
            elif int(_args["target_states"]) < 2:
                scale_fitness = ModelPenalty.state_penalty(m=penalty_coeff)
            else:
                scale_fitness = ModelPenalty.balanced_state_penalty(
                    target_state=int(_args["target_states"]),
                    m=penalty_coeff,
                )
        else:
            crossover = CrossoverOperator.one_point_triangular_row_swap
            scale_fitness = lambda _, mi: mi

        with_replacement = _args["selection__replacement"] == "True"
        if _args["selection"] == "tournament":
            select = lambda *args, **kwargs: SelectionOperator.tournament(
                *args,
                **kwargs,
                k=int(_args["selection__tournament_size"]),
                replace=with_replacement,
            )
        else:
            select = lambda *args, **kwargs: SelectionOperator.roulette_wheel(
                *args, **kwargs, replace=with_replacement
            )

        initial_population = []
        if _args["initial_population"]:
            try:
                with open(_args["initial_population"], "rb") as f:
                    initial_population = pickle.load(f)
            except:
                print(f"No such file: {_args['initial_population']} found.")
                print("Randomly initialising the population instead.")

        self.runner = GeneticRunner(data, mutations, crossover, select, scale_fitness)

        mg_params = {
            "reversible": _args["reversible"] == "True",
            "one_active_state": _args["one_active_state"] == "True",
            "p_edge": float(_args["p_edge"]),
        }

        models, stats = self.runner.run(
            states=int(_args["states"]),
            population=int(_args["population"]),
            iterations=int(_args["iterations"]),
            elite_ratio=float(_args["elite_ratio"]),
            n_processors=int(_args["n_processors"]),
            model_generator_params=mg_params,
            initial_population=initial_population,
            verbose=True,
            debug=True,
        )

        with open(_args["output_file"], "wb") as f:
            pickle.dump(models, f)
            print(f"Cached best models at {f}.")

        with open("stats_" + _args["output_file"], "wb") as f:
            pickle.dump(stats, f)
            print(f"Cached GA runner stats at {f}.")

    def on_interrupted(self, *args, **kwargs) -> None:
        if self.cached_results:
            sys.exit()

        print("Interrupted. Caching currently found results.")

        models, stats = self.runner.sorted_models, self.runner.runner_stats

        with open(self.default_args["output_file"], "wb") as f:
            pickle.dump(models, f)
            print(f"Cached best models at {f}.")

        with open("stats_" + self.default_args["output_file"], "wb") as f:
            pickle.dump(stats, f)
            print(f"Cached GA runner stats at {f}.")

        self.cached_results = True
        sys.exit()


class ParticleSwarmWeightOptimisationJob(Job):
    def __init__(self, verbose: bool):
        self.default_args = {
            "n_particles": 10,
            "n_processes": 10,
            "iters": 10,
            "set_curr_as_init": "False",
            "model_file": "",
            "cache_folder": "cache",
            "output_file": "pso_model_weights.dat",
        }
        self.name = "Particle Swarm Weight Optimisation"
        super().__init__(verbose)

    @Job.timed
    def run(self, args: Dict[str, any]) -> None:
        """
        Args:
          n_particles           Number of particles for the simulation
          n_processes           Number of processors to use
          iters                 Number of iterations of the simulation
          set_curr_as_init      Flag for if particles should be initialised with input model's weights
          model_file            Path to pickledƒ file containing model
          cache_folder          Path to cache folder where data may be cached
          output_file           Name of file to output data
        """
        _args = self.default_args

        for arg_name, arg_value in args.items():
            if arg_name in _args:
                _args[arg_name] = arg_value

        if self.verbose:
            print(f"[{self.name}] parameters used:")
            print("\n".join(f"\t{name}: {value}" for (name, value) in _args.items()))

        if not _args["model_file"]:
            print("Model file not specified.")
            return
        if not os.path.isfile(_args["model_file"]):
            print(f"Model file {_args['model_file']} not found.")
            return

        with open(_args["model_file"], "rb") as f:
            model = pickle.load(f)

        data, _, _, _ = get_tf_data(cache_folder=_args["cache_folder"])

        pso = ParticleSwarm()
        cost, pos = pso.optimise(
            data,
            model,
            n_particles=int(_args["n_particles"]),
            n_processes=int(_args["n_processes"]),
            iters=int(_args["iters"]),
            start_at_pos=_args["set_curr_as_init"] == "True",
        )

        with open(_args["output_file"], "wb") as f:
            pickle.dump((cost, pos), f)
            print(f"Cached found weights and their cost at {f}.")

    def on_interrupted(self, *args, **kwargs) -> None:
        return super().on_interrupted()


class NoveltySearchJob(Job):
    def __init__(self, verbose: bool):
        self.default_args = {
            # Initial population
            "states": 2,
            "population": 100,
            "iterations": 100,
            "elite_ratio": 0.1,
            "p_edge": 0.5,
            # Novelty specifics
            "linear_metric": "True",
            "novelty_threshold": -1,
            "archival_rate_threshold": 4,
            "archival_stagnation_threshold": 3,
            "max_archival_rate": -1,
            "n_neighbors": 15,
            # Penalty functions
            "target_states": -1,
            "penalty__active": "True",
            "penalty__coeff": 8.0,
            "penalty__reversed": "False",
            # Operators
            "fixed_states": "False",
            "selection": "tournament",
            "selection__replacement": "True",
            "selection__tournament_size": 3,
            # Constraints
            "reversible": "True",
            "one_active_state": "True",
            # I/O and hardware
            "n_processors": 1,
            "cache_folder": "cache",
            "initial_population": "",
            "output_file": "models.dat",
        }
        self.name = "Novelty Search with Local Competition"

        # For data recovery
        self.runner = None
        self.cached_results = False
        signal.signal(signal.SIGTERM, self.on_interrupted)

        super().__init__(verbose)

    @Job.timed
    def run(self, args: Dict[str, any]) -> None:
        """
        Args:
          # Initial population
          states            Number of states the model population starts with
          population        Number of models to consider in each generation
          iterations        Number of generations to run
          elite_ratio       Percentage of population that are kept as elites
          p_edge            Probability of edge connections at init population (0.5)

          # Novelty specifics
          linear_metric                 Flag for if distance metric is based on trajectories
          novelty_threshold             The minimum threshold to be met for a model to be archived
          archival_rate_threshold       The minimum number of models archived in a single iteration
                                         for which the novelty threshold is increased
          archival_stagnation_threshold The minimum number of iterations without archives
                                         before the novelty threshold is decreased

          # Penalty functions
          target_states     Target number of states for models (-1)
          penalty__active   Flag for if models should be penalised
          penalty__coeff    Hyperparameter for penalising models
          penalty__reversed Flag for if smaller models should be penalised

          # Genetic operators
          fixed_states               Flag for if states should be fixed (False)
          selection                  Selection operator (tournament, roulette)
          selection__replacement     Flag for if selection should be done with replacement
          selection__tournament_size Number of models sampled if tournament selection operator is chosen

          # Constraints
          reversible        Flag for if reactions should be reversible (True)
          one_active_state  Flag for if models should have only one active state (True)

          # I/O and hardware
          n_processors          Number of processors to parallelise model evaluation
          cache_folder          Path to cache folder where data may be cached
          initial_population    Name of file containing models to initialise the run
          output_file           Name of file to output model data
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
            penalty_coeff = float(_args["penalty__coeff"])
            if _args["penalty__active"] == "False":
                scale_fitness = lambda _, mi: mi
            elif _args["penalty__reversed"] == "True":
                scale_fitness = ModelPenalty.reversed_state_penalty(m=penalty_coeff)
            elif int(_args["target_states"]) < 2:
                scale_fitness = ModelPenalty.state_penalty(m=penalty_coeff)
            else:
                scale_fitness = ModelPenalty.balanced_state_penalty(
                    target_state=int(_args["target_states"]),
                    m=penalty_coeff,
                )
        else:
            crossover = CrossoverOperator.one_point_triangular_row_swap
            scale_fitness = lambda _, mi: mi

        with_replacement = _args["selection__replacement"] == "True"
        if _args["selection"] == "tournament":
            select = lambda *args, **kwargs: SelectionOperator.tournament(
                *args,
                **kwargs,
                k=int(_args["selection__tournament_size"]),
                replace=with_replacement,
            )
        else:
            select = lambda *args, **kwargs: SelectionOperator.roulette_wheel(
                *args, **kwargs, replace=with_replacement
            )

        initial_population = []
        if _args["initial_population"]:
            try:
                with open(_args["initial_population"], "rb") as f:
                    initial_population = pickle.load(f)
            except:
                print(f"No such file: {_args['initial_population']} found.")
                print("Randomly initialising the population instead.")

        self.runner = NoveltySearchRunner(
            data, mutations, crossover, select, scale_fitness
        )

        mg_params = {
            "reversible": _args["reversible"] == "True",
            "one_active_state": _args["one_active_state"] == "True",
            "p_edge": float(_args["p_edge"]),
        }

        archive, models, stats = self.runner.run(
            states=int(_args["states"]),
            population=int(_args["population"]),
            iterations=int(_args["iterations"]),
            elite_ratio=float(_args["elite_ratio"]),
            linear_metric=_args["linear_metric"] == "True",
            novelty_threshold=float(_args["novelty_threshold"]),
            archival_rate_threshold=int(_args["archival_rate_threshold"]),
            archival_stagnation_threshold=int(_args["archival_stagnation_threshold"]),
            max_archival_rate=int(_args["max_archival_rate"]),
            n_neighbors=int(_args["n_neighbors"]),
            n_processors=int(_args["n_processors"]),
            model_generator_params=mg_params,
            initial_population=initial_population,
            verbose=True,
            debug=True,
        )

        with open(_args["output_file"], "wb") as f:
            pickle.dump(models, f)
            print(f"Cached final population models at {f}.")

        with open("archive" + _args["output_file"], "wb") as f:
            pickle.dump(archive, f)
            print(f"Cached novelty archive at {f}.")

        with open("stats_" + _args["output_file"], "wb") as f:
            pickle.dump(stats, f)
            print(f"Cached NSLC runner stats at {f}.")

    def on_interrupted(self, *args, **kwargs) -> None:
        if self.cached_results:
            sys.exit()

        print("Interrupted. Caching currently found results.")

        archive, models, stats = self.runner.sorted_models, self.runner.runner_stats

        with open(self.default_args["output_file"], "wb") as f:
            pickle.dump(models, f)
            print(f"Cached final population models at {f}.")

        with open("archive_" + self.default_args["output_file"], "wb") as f:
            pickle.dump(archive, f)
            print(f"Cached novelty archive at {f}.")

        with open("stats_" + self.default_args["output_file"], "wb") as f:
            pickle.dump(stats, f)
            print(f"Cached NSLC runner stats at {f}.")

        self.cached_results = True
        sys.exit()
