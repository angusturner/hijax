from functools import partial

import numpy as np
from typing import Callable

from torch.utils.data import DataLoader, Dataset

from hijax.loaders import Loaders
from hijax.setup.utils import import_pkg


def worker_init_fn(worker_id):
    """
    Ensures that shuffle order and data augmentation is varied
    between the threads, when num_workers > 1.
    Note: possibly fixed in PyTorch 1.9
    :param worker_id:
    :return:
    """
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def setup_loaders(data_opts: dict, loader_opts: dict, module: str) -> Loaders:
    """
    Create PyTorch data loaders for training and validation.
    :param data_opts: `dataset` config sub-dictionary
    :param loader_opts: `loader` config sub-dictionary
    :param module: string name of the module to import `datasets` from
    :return:
    """
    # load available datasets from specified module
    datasets = import_pkg(module, "datasets")

    # get dataset options
    both = data_opts.get("both", {})
    train = data_opts.get("train", {})
    test = data_opts.get("test", {})

    # initialise datasets
    constructor_name: str = data_opts.pop("__constructor__", None)
    if constructor_name is None:
        raise ValueError("`data_opts` must specify a valid `__constructor__` key.")
    dataset_constructor: Callable[..., Dataset] = getattr(datasets, constructor_name)
    train_dataset: Dataset = dataset_constructor(**{**both, **train})
    test_dataset: Dataset = dataset_constructor(**{**both, **test})

    # loader settings
    both = loader_opts.get("both", {})
    train = loader_opts.get("train", {})
    test = loader_opts.get("test", {})

    # account for custom `collate_fn`
    collate_fn = None
    if hasattr(train_dataset, "get_collate_fn"):
        collate_fn = train_dataset.get_collate_fn()

    # account for custom batch sampler
    train_loader_opts = {**both, **train}
    test_loader_opts = {**both, **test}
    if hasattr(train_dataset, "get_batch_sampler"):
        train_loader_opts["batch_sampler"] = train_dataset.get_batch_sampler()
        test_loader_opts["batch_sampler"] = test_dataset.get_batch_sampler()

        # remove mutually exclusive args
        for opts in (train_loader_opts, test_loader_opts):
            opts.pop("batch_size", None)
            opts.pop("shuffle", None)
            opts.pop("sampler", None)
            opts.pop("drop_last", None)

    # initialise loaders
    DataLoader_ = partial(DataLoader, collate_fn=collate_fn, worker_init_fn=worker_init_fn)
    train_loader = DataLoader_(dataset=train_dataset, **train_loader_opts)
    test_loader = DataLoader_(dataset=test_dataset, **test_loader_opts)

    return Loaders(train_loader, test_loader)
