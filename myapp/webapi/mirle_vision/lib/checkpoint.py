from dataclasses import dataclass

import torch
from torch.optim.optimizer import Optimizer

from .model import Model


@dataclass
class Checkpoint:
    epoch: int
    model: Model
    optimizer: Optimizer

    @staticmethod
    def save(checkpoint: 'Checkpoint', path_to_checkpoint: str):
        raise NotImplementedError

    @staticmethod
    def load(path_to_checkpoint: str, device: torch.device) -> 'Checkpoint':
        raise NotImplementedError
