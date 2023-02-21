# BEng Project

A repository for code used in my WIP undergraduate thesis.

**Table of Contents**

1. [Structure](#structure)
2. [Usage](#usage)
   1. [Installing Dependencies](#installing-dependencies)
   2. [Simulations, Visualisations, and Benchmarking](#simulations-visualisations-and-benchmarking)

## Structure

The project's structure is given as follows:

```bash
├── archive                       # Archive of my previous attempt
├── cache                         # Cached JSONs, numpy arrays, and images
├── data                          # Experimental data
├── evolution
│   └── genetic
│       └── operator.py           # Genetic operators for promoter models 
│       └── runner.py             # Evolutionary runner
├── main.py                       # Main file
├── mi_estimation
│   ├── decoding.py               # Decoding-based MI estimator
│   └── estimator.py              # Abstract class for MI estimators
├── models
│   ├── generator.py              # Random model generator
│   ├── model.py                  # Promoter model class
│   ├── preset.py                 # Set of familiar models
│   └── rates
│       └── function.py           # Rate functions taking exogenous input
├── optimisation
│   └── grid_search.py            # Grid search on simple models
│   └── particle_swarm.py         # Particle swarm on general models
├── pipeline
│   ├── one_step_decoding.py      # One-step + Decoding pipeline
│   └── pipeline.py               # Pipeline class
├── requirements.txt              # Package dependencies list
├── ssa
│   ├── one_step.py               # One-step Master equation simulator
│   └── simulator.py              # Abstract class for trajectory simulators
└── utils
    └── process.py                # Processing of experimental data
```

## Usage

### Installing Dependencies

Set up a virtual environment, e.g.

```bash
virtualenv venv
source venv/bin/activate
```

Install packages with pip.

```bash
pip3 install -r requirements.txt
```

### Simulations, Visualisations, and Benchmarking

The `main.py` file contains examples which can be run by specifying the function under `main`. In the future, a dedicated `/examples` folder with proper documentation will be added.
