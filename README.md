# hijax

An experiment framework for [jax](https://github.com/google/jax) and [haiku](https://github.com/deepmind).
Provides an opinionated structure for running machine learning experiments, including
the following features:
- Config management using [Hydra](https://hydra.cc/docs/intro)
- Model checkpointing and loading
- Plotting to [Weights and Biases](wandb.ai/)
- Data loading using the PyTorch `DataLoader`

## Example Project

For a complete example, see https://github.com/angusturner/neural_chess

## Requirements

- Python >= 3.8
- [Jax](https://github.com/google/jax). Note: this is not handled by hijax, since it depends 
on your specific platform.

## Installation

Clone the repository, and then run `pip install hijax`.

## Structure of a Hijax Project

Hijax is opinionated about the structure of a project. While additional folders and files can be added, it is highly
advised to at least have the following skeleton:
```text
<module name>
├── config
│   ├── dataset
│       └── ...
│   ├── loader
│       └── ...
│   ├── model
│       └── ...
│   ├── worker
│       └── ...
│   └── config.yaml
├── <module name>
│   ├── datasets
│       └── __init__.py
│       └── ...
│   ├── models
│       └── __init__.py
│       └── ...
│   └── workers
│       └── __init__.py
│       └── ...
└── ...other scripts, notebooks, data etc.
```

Where `<your module name>` is the name of your project (i.e. `neural_chess` in the example project).

## Core Concepts

Experiments in hijax are organised around three core concepts:

### Models

The concept of a model in hijax is quite flexible. A model can be any python callable that maps a set of inputs to a set
of outputs. For example, this could be some neural net code implemented in Jax or Haiku. 

For each model we define, we also need to export a corresponding factory function in
`model.__init__.py`. This function takes the model settings/hyper-parameters (passed down from the yaml config),
and returns the initialised model as a python callable.

Functions registered in the `__init__.py` can be referenced by name in the config, allowing
us to dynamically load the appropriate model code for our experiment. See [`build_policy_net`](https://github.com/angusturner/neural_chess/blob/master/neural_chess/models/policy_net.py)
for an example.

### Workers

The `Worker` abstraction in hijax provides a common interface for running experiments, saving and loading 
checkpoints and plotting to WandB. This is where optimisation code belongs, as well as any extra methods
that are required to support model evaluation or inference.

A worker must implement the following required methods:
- `__init__`: takes the model and data loaders as arguments, along with any user-defined args/kwargs.
- `train`: iterate over a training loader, and perform optimisation
- `eval`: iterate over a validation loader, and return evaluation metrics
- `get_state_dict()` : return a dictionary of the current state of the worker (e.g. model and optimiser state)
- `load_state_dict(state_dict)` : load a state dict into the worker

As with models, each worker should be exported in `worker.__init__.py` to allow it to be referenced
in the config.


### Datasets

Datasets in hijax leverage the excellent PyTorch `DataLoader` class. A `Dataset` can therefore be any
class that is compatible with the PyTorch `DataLoader`, meaning:

- It must implement the `__len__` method, which returns the number of samples in the dataset.
- It must implement the `__getitem__` method, which returns a single sample from the dataset.

Once again, we must declare each dataset in `datasets.__init__.py` to allow it to be referenced in the config.

### Coupling between each component

There is not a strict 1:1 relationship between workers and models, models, datasets. It is up to the 
developer to ensure a compatible API between these various components. For example, a supervised image 
classification task may require a single worker with generic optimisation logic, that can support
many datasets and models.

## Configuring an Experiment

TODO

For more information on how Hydra works refer to the [Hydra docs](https://hydra.cc/docs/intro).

## Tying it all Together


### A simple training script

```python
import hydra
from omegaconf import DictConfig
from hijax.setup import setup_worker


@hydra.main(config_path="config", config_name="config")
def train(cfg: DictConfig) -> None:
    # get the experiment name
    name = cfg.get("name", False)
    if not name:
        raise Exception("Must specify experiment name on CLI. e.g. `python train.py name=vae ...`")

    # setup the worker
    overwrite = cfg.get("overwrite", False)
    reset_metrics = cfg.get("reset_metrics", False)
    worker, cfg = setup_worker(
        name=name,
        cfg=cfg,
        overwrite=overwrite,
        reset_metrics=reset_metrics,
        module="neural_chess",  # the module containing the model, worker and datasets
        with_wandb=True,
        with_loaders=True,
    )

    # train
    worker.run(nb_epoch=cfg["nb_epoch"])


if __name__ == "__main__":
    train()
```