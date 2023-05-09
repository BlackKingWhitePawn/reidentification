from torch import nn
from copy import deepcopy
from collections.abc import Callable


class SiameseTransfered(nn.Module):
    def __init__(
        self,
            base_model: nn.Module,
            replace_last_layer_fn: Callable[[nn.Module], None],
            freeze_grad_fn: Callable[[nn.Module], None] = None,
    ) -> None:
        """Сиамская сеть с предобученной моделью в качестве twin модели
        ### Parameters:
        - base_model: Module - предобуенная модель
        - freeze_grad_fn: Callable[[Module], None] - функция, определяющая, как замораживаются градиенты внутренней модели 
        - replace_last_layer_fn: Callable[[nn.Module], None] - функция, определяющая, как заменяется выходной слой
        """
        super(SiameseTransfered).__init__()
        self.base_model = deepcopy(base_model)
        if (freeze_grad_fn is not None):
            freeze_grad_fn(self.base_model)

        replace_last_layer_fn(self.base_model)

    def forward(self, x):
        return self.base_model(x)
