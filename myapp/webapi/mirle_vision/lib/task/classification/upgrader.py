import glob
import os
from typing import Tuple, List

import torch


class Upgrader:

    def __init__(self, path_to_checkpoints_dir: str,
                 from_version_tuple: Tuple[int, ...], to_version_tuple: Tuple[int, ...]):
        super().__init__()
        self.path_to_checkpoints_dir = path_to_checkpoints_dir
        self.from_version_tuple = from_version_tuple
        self.to_version_tuple = to_version_tuple

    def upgrade(self):
        pass
    #     self._recursively_upgrade(versions_to_upgrade=[(1, 1, 2)])
    #
    # def _recursively_upgrade(self, versions_to_upgrade: List[Tuple[int, ...]]):
    #     version_to_upgrade = versions_to_upgrade.pop(0)
    #
    #     if self.from_version_tuple < version_to_upgrade <= self.to_version_tuple:
    #         if version_to_upgrade == (1, 1, 2):
    #             self._upgrade_to_1_1_2()
    #
    #     if len(versions_to_upgrade) > 0:
    #         self._recursively_upgrade(versions_to_upgrade)
    #
    # def _upgrade_to_1_1_2(self):
    #     path_to_checkpoint_list = sorted(glob.glob(os.path.join(self.path_to_checkpoints_dir, '*', 'checkpoint.pth')))
    #     for path_to_checkpoint in path_to_checkpoint_list:
    #         checkpoint = torch.load(path_to_checkpoint)
    #         needs_freeze_bn = checkpoint['algorithm_params'].pop('batch_size_per_device') < 16
    #         checkpoint['algorithm_params']['needs_freeze_bn'] = needs_freeze_bn
    #         torch.save(checkpoint, path_to_checkpoint)
