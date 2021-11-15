from torch.utils.data import DataLoader
from typing import NamedTuple


class Loaders(NamedTuple):
    """
    Container for the data loaders
    """

    train: DataLoader
    test: DataLoader
