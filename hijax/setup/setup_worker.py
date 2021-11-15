import os

import wandb
from wandb.wandb_run import Run

from omegaconf import DictConfig
from typing import Optional, Union, Callable, Any

from hijax import Worker
from hijax.setup import Loaders, setup_loaders
from hijax.setup.setup_config import setup_config
from hijax.setup.utils import import_pkg


def setup_worker(
    name: str,
    module: str,
    cfg: Optional[Union[DictConfig, dict]] = None,
    exp_dir: Optional[str] = None,
    with_wandb: bool = True,
    with_loaders: bool = True,
    overwrite: bool = False,
    reset_metrics: bool = False,
    checkpoint_id: str = "best",
) -> (Worker, DictConfig):
    """
    Create or load an experiment.
    :param name: unique experiment name for saving/resuming
    :param module: name of the current python module. should export `workers`, `models` and `datasets`
    :param cfg: experiment config. can be `None` if loading an existing experiment,
        otherwise a `dict` or `DictConfig` containing the configuration is expected.
    :param exp_dir: directory to store experiment runs. can be set as environment variable `EXPERIMENT_DIR`
    :param with_wandb: include an instance of wandb ? (required for training)
    :param with_loaders: load the dataset and supply to the worker ?
    :param overwrite: overwrite any existing configuration, if an experiment with this name already exists
    :param reset_metrics: if `True` the `lowest_loss` and `summary_stats` are reset
    :param checkpoint_id: for inference / resuming, which checkpoint should we load?
    :return:
    """
    # get experiment directory
    if exp_dir is None:
        exp_dir = os.environ.get("EXPERIMENT_DIR", False)
        if not exp_dir:
            raise Exception(
                "No experiment directory defined. Set environment variable `EXPERIMENT_DIR` or "
                "pass as kwarg `setup_worker(..., exp_dir=/somewhere/to/save/checkpoints)"
            )

    # setup the configuration and experiment directory
    cfg = setup_config(name, cfg, exp_dir, overwrite)

    # load the available workers and models
    workers = import_pkg(module, "workers")
    models = import_pkg(module, "models")

    # setup the model
    model_constructor: Callable[..., Any] = getattr(models, cfg["model"].pop("__constructor__"))
    model: Any = model_constructor(**cfg["model"])

    # setup the loaders
    loaders: Optional[Loaders] = None
    if with_loaders:
        loaders = setup_loaders(data_opts=cfg["dataset"], loader_opts=cfg["loader"], module=module)

    # setup visualisation
    run: Optional[Run] = None
    if with_wandb:
        assert "wandb" in cfg, "Wandb is enabled but no wandb configuration is provided."
        cfg_ = cfg
        if type(cfg) == DictConfig:
            cfg_ = dict(cfg)
        run = wandb.init(name=name, config=cfg_, id=cfg["run_id"], resume="allow", **cfg["wandb"])

    # initialise the worker (any subclass of Worker)
    run_dir = "{}/{}".format(exp_dir, name)
    worker_class: Callable[..., Worker] = getattr(workers, cfg["worker"].pop("__constructor__"))
    worker: Worker = worker_class(
        model=model,
        loaders=loaders,
        checkpoint_id=checkpoint_id,
        exp_name=name,
        run_dir=run_dir,
        wandb=run,
        reset_metrics=reset_metrics,
        **cfg["worker"],
    )

    return worker, cfg
