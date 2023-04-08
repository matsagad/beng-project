# Jobs

Jobs can be run via the shell command:

```bash
python3 run_job.py {job_name} {job_arg=job_value}*
```

Common to all jobs is the `max_runtime` argument which is set to `7200` seconds (two hours) by default. When this is exceeded, the job is interrupted and the `on_interrupted` method is called. To account for module imports and such, please allow a max runtime of at least 1-2 minutes earlier than required. 

## Supported Jobs

### Genetic Algorithm Simulation

Runs a genetic algorithm on a randomly generated initial population of models. At the final generation, it saves the models in order of fitness to an output file.

| Arguments | Description | Default |
|---|---|:---:|
| `states`           | Number of states the model population starts with | `2` |
| `population`       | Number of models to consider in each generation | `10` |
| `iterations`       | Number of generations to run | `10` |
| `elite_ratio`      | Percentage of population that are kept as elites | `0.2` |
| `p_edge`           | Probability of edge connections at init population | `0.5` |
| `target_states`    | Target number of states for models | `-1` (inactive) |
| `penalty__active`    | Flag for if models should be penalised | `True` |
| `penalty__coeff`    | Hyperparameter for penalising models | `8.0` |
| `penalty__reversed` | Flag for if smaller models should be penalised | `False` |
| `fixed_states`     | Flag for if states should be fixed | `False` |
| `selection` | Selection operator (tournament, roulette) | `tournament` |
| `selection__replacement` | Flag for if selection should be done with replacement | `True` |
| `selection__tournament_size` | Number of models sampled if tournament selection operator is chosen | `3` |
| `reversible`       | Flag for if reactions should be reversible | `True` |
| `one_active_state` | Flag for if models should have only one active state | `True` |
| `n_processors`     | Number of processors to parallelise model evaluation | `1` |
| `cache_folder`     | Path to cache folder where data may be cached | `cache/` |
| `initial_population` | Name of file containing models to initialise the run | n/a |
| `output_file`      | Name of file to output model data | `models.dat` |

Example:

```bash
python3 run_job.py genetic_algorithm cache_folder=$CACHE_FOLDER n_processors=100 states=4 population=200 iterations=100 one_active_state=True fixed_states=False
```

### Particle Swarm Weight Optimisation

Given a model (from a pickled file), this runs a particle swarm optimisation to find the weights/coefficients of existing rate functions that yield the greatest MI.

| Arguments | Description | Default |
|---|---|:---:|
|  `n_particles` |  Number of particles for the simulation | `10` |
| `n_processes` |  Number of processors to use | `10` |
| `iters`        |  Number of iterations of the simulation | `10` |
| `set_curr_as_init` |  Flag for if particles should be initialised with input model's weights | `False` |
| `model_file`   |  Path to pickled file containing model | n/a |
| `cache_folder` |  Path to cache folder where data may be cached | `cache/` |
| `output_file`  |  Name of file to output data | `pso_model_weights.dat` |

Example:

```bash
python3 run_job.py pso_weight_optimisation cache_folder=$CACHE_FOLDER n_particles 8 n_processes=8 iters=100 set_curr_as_init=True
```
