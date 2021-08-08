import os
from typing import Optional, List

from torch import nn


class Util:

    @staticmethod
    def setup_visible_devices(visible_devices: Optional[List[int]]):
        # NOTE: This should be called before reaching any CUDA functions (e.g.: `torch.cuda.is_available()`)
        if visible_devices is None:
            os.environ.pop('CUDA_VISIBLE_DEVICES', None)
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(it) for it in visible_devices])

    @staticmethod
    def freeze_bn_modules(module: nn.Module):
        # NOTE: It's crucial to freeze batch normalization modules for few batches training, which can be done by following processes
        #         (1) change mode to `eval`
        #         (2) disable gradient
        bn_modules = nn.ModuleList([it for it in module.modules() if isinstance(it, nn.BatchNorm2d)])
        for bn_module in bn_modules:
            bn_module.eval()
            for parameter in bn_module.parameters():
                parameter.requires_grad = False
