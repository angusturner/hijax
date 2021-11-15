import torch
import os
import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any
from pathlib import Path

from wandb.wandb_run import Run

from torch.utils.data import DataLoader

from hijax import Loaders
from hijax.utils import recursive_set


class Worker(ABC):
    def __init__(
        self,
        exp_name: str,
        run_dir: str,
        loaders: Optional[Loaders] = None,
        wandb: Optional[Run] = None,
        epoch_save_freq: int = 50,
        log_interval: int = 50,
        reset_metrics: bool = False,
        upload_checkpoints: bool = False,
        *_args,
        **kwargs,
    ):
        """
        All workers should inherit from this class. It will:
        1. provide common utilities for saving, loading and device management.
        2. enforce a common interface between workers. i.e) must implement `.train()` and `evaluate.()`
        3. provide utilities for logging to WandB.

        :param exp_name: name of experiment
        :param run_dir: directory to save configurations, weights, artifacts for this experiment.
        :param wandb: the session that is returned from wandb.init(...).
        :param loaders: a Loaders object that contains the train and test sets.
        :param epoch_save_freq: how often to save intermediate checkpoints
        :param log_interval: how many gradient updates before logging to wandb
        :param reset_metrics: whether to discard metrics saved with the existing checkpoint.
            (i.e. for fine-tuning)
        :param upload_checkpoints: whether to upload checkpoints to wandb.
        :param args:
        :param kwargs:
        """

        self.exp_name = exp_name
        self.wandb = wandb
        self.loaders = loaders
        self.run_dir = run_dir
        self.reset_metrics = reset_metrics
        self.upload_checkpoints = upload_checkpoints

        # keep a record of all stateful objects, to include in the experiment checkpoints
        # see: `register_state()
        self.state_dict = {}

        # track loss and summary stats
        self.lowest_loss = float("inf")
        self.summary_stats = {}

        # epoch related stuff
        self.epoch_counter = 0
        self.epoch_save_freq = epoch_save_freq
        self.log_interval = log_interval

        # cache values until its time to plot them
        self._counters = {"train": 0, "test": 0}
        self._metric_cache = {"train": {}, "test": {}}

    @abstractmethod
    def train(self, loader: Optional[DataLoader] = None) -> Any:
        pass

    @abstractmethod
    def evaluate(self, loader: Optional[DataLoader] = None) -> (float, Dict):
        pass

    def run(self, nb_epoch: int):
        """
        Run an experiment for the specified number of epoch.
        :param nb_epoch:
        :return:
        """
        assert self.loaders is not None, "No loaders initialised! Ensure `with_loaders=True` in call to setup_worker()"
        for epoch in range(nb_epoch):
            # reset numpy random seed
            np.random.seed(0)

            # train one epoch and evaluate
            self.train()
            self.epoch_counter += 1
            loss_score, summary_stats = self.evaluate()

            # track individual metrics at an epoch level
            assert type(summary_stats) == dict, "`worker.evaluate()` expects return type : (float, Dict)"
            for k, v in summary_stats.items():
                self.summary_stats[k] = v

            # save `best`
            if loss_score < self.lowest_loss:
                print("New lowest test loss {}".format(loss_score))
                self.save(checkpoint_id="best")
                self.lowest_loss = loss_score
                if self.wandb is not None:
                    self.wandb.summary["lowest_loss"] = loss_score
                    for k, v in self.summary_stats.items():
                        self.wandb.summary[f"{k}: Lowest Test Set (Avg)"] = v

            # save every `x` epoch
            if self.epoch_counter % self.epoch_save_freq == 0:
                self.save(checkpoint_id="{}".format(self.epoch_counter))

            # overwrite latest weights
            self.save(checkpoint_id="latest")

    def register_state(self, obj: dict, name: str):
        """
        Register an object to be included in the experiment checkpoint. Any pickle-able dictionary is valid.
        :param obj: any dictionary
        :param name: unique name for what is being registered
        """
        assert name not in self.state_dict, "Duplicate key in state list for '{}'".format(name)
        self.state_dict[name] = obj

    def save(self, checkpoint_id: str = "latest"):
        """
        Save state of all registered objects to a checkpoint.
        :param checkpoint_id:
        :return:
        """
        # track additional top-level state, related to the evaluation metrics and nb. of completed epochs
        state_dict = {
            "lowest_loss": self.lowest_loss,
            "summary_stats": self.summary_stats,
            "epoch_counter": self.epoch_counter,
        }
        for key, val in self.state_dict.items():
            state_dict[key] = val

        # save checkpoint
        checkpoint_path = os.path.join(self.run_dir, f"checkpoint_{checkpoint_id}.pt")
        print("Saving checkpoint {}".format(checkpoint_path))
        torch.save(state_dict, checkpoint_path)

        # upload and overwrite the checkpoints to wandb if `upload_checkpoints` enabled in worker config
        if self.wandb is not None and self.upload_checkpoints and checkpoint_id in {"latest", "best"}:
            self.wandb.save(glob_str=checkpoint_path, base_path=str(Path(self.run_dir).parent), policy="live")

    def load(self, checkpoint_id: str = "best", skip_keys: Optional[List[str]] = None):
        """
        Load state from existing checkpoint.
        :param checkpoint_id:
        :param skip_keys: skip loading the state of these keys. For example, we may want to discard certain bits of
            state (e.g. the optimiser state) for fine-tuning.
        :return:
        """
        skip_keys = set(skip_keys or [])
        checkpoint_path = os.path.join(self.run_dir, f"checkpoint_{checkpoint_id}.pt")

        if not os.path.exists(checkpoint_path):
            print("Checkpoint not found {}".format(checkpoint_path))
            return

        state_dict = torch.load(checkpoint_path)
        for key, src in state_dict.items():
            if key in skip_keys:
                print(f"Skipping loading of {key}")
                continue
            # skip over important top-level state (handled separately, below)
            if key in {"lowest_loss", "summary_stats", "epoch_counter"}:
                continue
            # handle key mismatches
            if key not in self.state_dict:
                print(
                    f"Warning! {key} was found in the checkpoint but not in the stateful objects for the current"
                    f"run!"
                )
                continue

            # copy the leaf-values of the loaded dict structure into the state dict
            sink = self.state_dict[key]
            recursive_set(src, sink)

        if not self.reset_metrics:
            self.summary_stats = state_dict.get("summary_stats", {})
            self.lowest_loss = state_dict.get("lowest_loss", float("inf"))
        self.epoch_counter = state_dict.get("epoch_counter", 0)

    @staticmethod
    def unwrap_value(v):
        if torch.is_tensor(v):
            return v.item()
        elif type(v) == np.ndarray:
            return v.item()
        else:
            return v

    def _plot_loss(self, metrics: dict, train=True):
        """
        Plot metrics to weights and biases.

        Keeps a moving average of values and pushes them every `log_interval`.

        :param metrics: dictionary of things to track
        :param train: ?
        """
        subset = "train" if train else "test"

        for k, v in metrics.items():
            full_key = "{} {}".format(k, subset)
            if full_key not in self._metric_cache[subset]:
                self._metric_cache[subset][full_key] = []
            v_raw = Worker.unwrap_value(v)
            self._metric_cache[subset][full_key].append(v_raw)

        # increment the counter
        self._counters[subset] += 1

        # empty cache and plot!
        if self._counters[subset] % self.log_interval == 0:
            avg = {}
            for k, v in self._metric_cache[subset].items():
                avg[k] = np.mean(v)
                self._metric_cache[subset][k] = []

            self.wandb.log(avg)
            self._counters[subset] = 0
