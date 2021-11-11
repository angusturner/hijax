import numpy as np
from typing import Tuple

from torch.utils.data import DataLoader, Dataset
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


def setup_loaders(dataset_class: str, data_opts: dict, loader_opts: dict, module: str) -> Tuple[DataLoader, DataLoader]:
    """
    Create PyTorch data loaders for training and validation.
    :param dataset_class: name of dataset class. any class exported in <module>/data/__init__.py
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
    dataset_class = getattr(datasets, dataset_class)
    train_dataset: Dataset = dataset_class(**{**both, **train})
    test_dataset: Dataset = dataset_class(**{**both, **test})

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
    train_loader = DataLoader(
        dataset=train_dataset, collate_fn=collate_fn, worker_init_fn=worker_init_fn, **train_loader_opts
    )
    test_loader = DataLoader(
        dataset=test_dataset, collate_fn=collate_fn, worker_init_fn=worker_init_fn, **test_loader_opts
    )

    return train_loader, test_loader