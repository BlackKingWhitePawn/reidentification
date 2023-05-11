from collections.abc import Callable
from copy import deepcopy

import torch
import torch.nn.functional as F
from torch import nn


class SiameseTransfered(nn.Module):
    def __init__(
        self,
            base_model: nn.Module,
            replace_last_layer_fn: Callable[[nn.Module], None] = None,
            freeze_grad_fn: Callable[[nn.Module], None] = None,
            name: str = 'siamese_transfered'
    ) -> None:
        """Сиамская сеть с предобученной моделью в качестве twin модели
        ### Parameters:
        - base_model: Module - предобуенная модель
        - freeze_grad_fn: Callable[[Module], None] - функция, определяющая, как замораживаются градиенты внутренней модели 
        - replace_last_layer_fn: Callable[[nn.Module], None] - функция, определяющая, как заменяется выходной слой
        - name: str - название модели
        """
        super(SiameseTransfered, self).__init__()
        self.base_model = deepcopy(base_model)
        self.name = name

        if (freeze_grad_fn is not None):
            freeze_grad_fn(self.base_model)

        if (replace_last_layer_fn):
            replace_last_layer_fn(self.base_model)

    def forward(self, x1, x2):
        out1 = self.base_model(x1)
        out2 = self.base_model(x2)

        return torch.sigmoid((F.pairwise_distance(
            out1, out2, keepdim=True)))
