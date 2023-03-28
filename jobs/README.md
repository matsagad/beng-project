# Jobs

Jobs can be run via the shell command:

```bash
python3 run_job.py {job_name} {job_arg=job_value}*
```

## Supported Jobs

### Genetic Algorithm Simulation

Runs a genetic algorithm on a randomly generated initial population of models. At the final generation, it saves the models in order of fitness to an output file.

|Arguments| Description|
|---|---|
| `states`           | Number of states the model population starts with |
| `population`       | Number of models to consider in each generation |
| `iterations`       | Number of generations to run |
| `elite_ratio`      | Percentage of population that are kept as elites (0.2) |
| `p_edge`           | Probability of edge connections at init population (0.5) |
| `target_states`    | Target number of states for models (-1) |
| `penalty__active`    | Flag for if models should be penalised |
| `penalty__coeff`    | Hyperparameter for penalising models |
| `penalty__reversed` | Flag for if smaller models should be penalised|
| `fixed_states`     | Flag for if states should be fixed (False) |
| `selection` | Selection operator (tournament, roulette)
| `selection__replacement` | Flag for if selection should be done with replacement |
| `selection__tournament_size` | Number of models sampled if tournament selection operator is chosen |
| `reversible`       | Flag for if reactions should be reversible (True) |
| `one_active_state` | Flag for if models should have only one active state (True) |
| `n_processors`     | Number of processors to parallelise model evaluation |
| `cache_folder`     | Path to cache folder where data may be cached |
| `output_file`      | Name of file to output model data |

Example:

```bash
python3 run_job.py genetic_algorithm cache_folder=$CACHE_FOLDER n_processors=100 states=4 population=200 iterations=100 one_active_state=True fixed_states=False
```
