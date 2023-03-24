from __future__ import annotations
from abc import ABC

import torch
from src.pinn import PINN

from src.params import Params


class Loss(ABC):
    """Callable class that defines loss function
    
    Attributes
    ----------
    x
        Quadrature points
    eps
        PDE constant
    n_test_func
        Number of test functions

    Examples
    --------
        >>> loss = LossAD(x, 1.0, 30)
        >>> loss(pinn)                      # <-- You can do this
    """

    def __call__(self, pinn: PINN) -> torch.Tensor:
        raise NotImplementedError()
