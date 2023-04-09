# BEng Project

A repository for code used in my WIP undergraduate thesis.

**Table of Contents**

1. [Structure](#structure)
2. [Installation](#installation)
   1. [Dependencies](#dependencies)
   2. [Data](#data)
3. [Simulations, Visualisations, and Benchmarking](#simulations-visualisations-and-benchmarking)
4. [Running Jobs](#running-jobs)
5. [Running Tests](#running-tests)

## Structure

The project's structure is given as follows:

```bash
├── archive                       # Archive of my previous attempt
├── cache                         # Cached JSONs, numpy arrays, and images
├── data                          # Experimental data
├── /evolution
│   └── /genetic
│       ├── /operators             
│       │   ├── crossover.py      # Crossover operators for promoter models
│       │   ├── mutation.py       # Mutation operators for promoter models 
│       │   └── selection.py      # Selection operators for promoter models
│       ├── penalty.py            # A collection of penalty functions
│       └── runner.py             # Genetic algorithm runner
├── /examples
│   ├── benchmarking.py           # Benchmarking of simulations
│   ├── genetic_analysis.py       # Analysis of past GA runs
│   ├── mi_trends.py              # Trends in MI as methodology is tweaked
│   ├── optimisation.py           # Model rates optimisation
│   ├── tutorial.py               # Tutorial examples to get started
│   └── visualisation.py          # Visualisation of models and trajectories
├── /jobs
│   ├── job.py                    # Abstract class for jobs/tasks to be run
│   └── routine_jobs.py           # Routine jobs to run
├── main.py                       # Main file
├── /mi_estimation
│   ├── decoding.py               # Decoding-based MI estimator
│   └── estimator.py              # Abstract class for MI estimators
├── /models
│   ├── generator.py              # Random model generator
│   ├── model.py                  # Promoter model class
│   ├── preset.py                 # Set of familiar models
│   └── /rates
│       └── function.py           # Rate functions taking exogenous input
├── /optimisation
│   └── grid_search.py            # Grid search on simple models
│   └── particle_swarm.py         # Particle swarm on general models
├── /pipeline
│   ├── one_step_decoding.py      # One-step + Decoding pipeline
│   └── pipeline.py               # Pipeline class
├── requirements.txt              # Package dependencies list
├── run_job.py                    # Entrypoint for running jobs
├── run_test.py                   # Entrypoint for running tests
├── /ssa
│   ├── one_step.py               # One-step Master equation simulator
│   └── simulator.py              # Abstract class for trajectory simulators
├── /tests
│   ├── genetic_operator.py       # Tests genetic operators behave as expected
│   ├── mi_estimation.py          # Tests MI estimation is robust
│   └── test.py                   # Class for running and tallying tests
└── /utils
    ├── data.py                   # Class for inheritting access to data
    └── process.py                # Processing of experimental data
```

## Installation

### Dependencies

Set up a virtual environment, e.g.

```bash
virtualenv venv
source venv/bin/activate
```

Install packages with pip.

```bash
pip3 install -r requirements.txt
```

As of writing, at least Python 3.10 is required to accommodate certain package versions.

### Data

Populate the `/data` folder with the data expected (see `/data/README.md`). This is necessary for the bulk of simulations to work.

## Simulations, Visualisations, and Benchmarking

Examples on accessing the library classes and functions are all documented in the `/examples` folder. The `main.py` file contains commented-out snippets taken from there which can be run by uncommenting and importing as appropriate. More information on these snippets can be found within their function declarations.

## Running Jobs

The `/jobs` folder contains a high-level abstraction for running common tasks in parallel on job-scheduling platforms (e.g. for an HPC cluster). Jobs are run through `run_job.py`. More information on the expected command-line arguments can be found in `/jobs/README.md`.

## Running Tests
The `/tests` folder contains tests pertaining to parts of the project methodology. Run `python3 run_test.py` to run all tests.