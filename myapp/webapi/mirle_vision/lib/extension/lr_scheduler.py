from bisect import bisect_right
from typing import List, Optional

from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer


class WarmUpMultiStepLR(_LRScheduler):

    def __init__(self, optimizer: Optimizer, milestones: List[int], gamma: float = 0.1,
                 factor: float = 0.3333, num_iters: int = 500, last_epoch: int = 0):
        self.milestones = milestones
        self.gamma = gamma
        self.factor = factor
        self.num_iters = num_iters
        self.last_warm_iter = 0
        super().__init__(optimizer, last_epoch - 1)

    def get_lr(self) -> List[float]:
        if (not self.milestones or self.last_epoch < self.milestones[0]) and self.last_warm_iter < self.num_iters:
            alpha = self.last_warm_iter / self.num_iters
            factor = (1 - self.factor) * alpha + self.factor
        else:
            factor = 1

        decay_power = bisect_right(self.milestones, self.last_epoch)
        return [base_lr * factor * self.gamma ** decay_power
                for base_lr in self.base_lrs]

    def warm_step(self):
        self.last_warm_iter += 1
