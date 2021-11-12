# hijax

### WARNING - UNDER CONSTRUCTION
### This repo could change drastically in the near future!

High level abstractions for Jax. Specifically, hijax provides:
- CLI for initialising new machine-learning projects
- A framework for training neural nets using Jax and Haiku
- Dataset management
- State management (i.e checkpoint saving)
- Plotting to Weights and Biases
- Other modules and utilities?

## Requirements

- Python >= 3.8
- Jax (install depending on your OS)
- PyTorch CPU install (used for data loading)
- (recommended) Linux

## Installation

TODO

## QuickStart

hijax enforces a particular directory layout, and as such provides scripts for initialising
a new project or for converting an existing one.

`TODO`
### Existing Project

### New Project

## Experiment Framework

Experiments in hijax are organised around three core concepts:

### Models

- A Haiku module object (inherits from `hk.Module`)
- Should not contain optimisation related code.
- Must be included `models/__init__.py` to be dynamically loaded by the worker.

### Workers

- Responsible for training and evaluating models
- Must implement `.train` and `.evaluate`
- Should inherit from the `AbstractWorker` class, which will:
    1. provide utility methods for saving and loading experiment checkpoints.
    2. enforce a common training interface
- Must be included in `workers/__init__.py` to be dynamically loaded by the experiment runner.

### Datasets

- Must implement `__len__` and `__getitem__`
- Will be wrapped with the PyTorch `DataLoader` class for batching and collation
- Must be included in `datasets/__init__.py` to be dynamically loaded by the experiment runner.

Note that there is not a strict 1:1 relationship between workers and models, models and datasets etc.
It is up to the developer to ensure a compatible API between these various components.

### Examples

TODO

For more information on how Hydra works refer to the [Hydra docs](https://hydra.cc/docs/intro).