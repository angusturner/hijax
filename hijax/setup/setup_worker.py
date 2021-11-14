import os
import wandb

from omegaconf import DictConfig
from typing import Optional, Union, Callable, Any

from hijax import AbstractWorker
from hijax.setup.setup_config import setup_config
from hijax.setup.utils import import_pkg


def setup_worker(
    name: str,
    module: str,
    cfg: Optional[Union[DictConfig, dict]] = None,
    exp_dir: Optional[str] = None,
    include_wandb: bool = True,
    overwrite: bool = False,
    reset_metrics: bool = False,
    checkpoint_id: str = "best",
) -> (AbstractWorker, DictConfig):
    """
    Create or load an experiment.
    :param name: unique experiment name for saving/resuming
    :param module: a python module containing `workers`, `models` and `datasets`
    :param cfg: experiment config. can be `None` for existing experiment.
        otherwise a `dict` or `DictConfig` containing the configuration.
    :param exp_dir: directory to store experiment runs
    :param include_wandb: include an instance of wandb ? (required for training)
    :param overwrite: overwrite existing experiment
    :param reset_metrics: if `True` the `lowest_loss` and `summary_stats` are reset
    :param checkpoint_id: for inference / resuming, which checkpoint should we load?
    :return:
    """
    # load the available workers and models
    workers = import_pkg(module, "workers")
    models = import_pkg(module, "models")

    # get experiment directory
    if exp_dir is None:
        exp_dir = os.environ.get("EXPERIMENT_DIR", False)
        if not exp_dir:
            raise Exception(
                "No experiment directory defined. Set environment variable `EXPERIMENT_DIR` or "
                "pass as kwarg `setup_worker(..., exp_dir=?)"
            )

    # setup the directory
    cfg = setup_config(name, cfg, exp_dir, overwrite)

    # setup the model
    model_constructor: Callable[..., Any] = getattr(models, cfg["model"].pop("__constructor__"))
    model: Any = model_constructor(**cfg["model"])

    # setup visualisation
    run = None
    if include_wandb and "wandb" in cfg:
        cfg_ = cfg
        if type(cfg) == DictConfig:
            cfg_ = dict(cfg)
        run = wandb.init(name=name, config=cfg_, id=cfg["run_id"], resume="allow", **cfg["wandb"])

    # initialise the worker
    run_dir = "{}/{}".format(exp_dir, name)
    worker_class: Callable[..., AbstractWorker] = getattr(workers, cfg["worker"].pop("__constructor__"))
    worker: AbstractWorker = worker_class(
        model=model,
        checkpoint_id=checkpoint_id,
        exp_name=name,
        run_dir=run_dir,
        wandb=run,
        reset_metrics=reset_metrics,
        **cfg["worker"],
    )

    return worker, cfg
