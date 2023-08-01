from abc import ABC

import torch

from src.pinn import PINN


class NNError(ABC):
    """Parent abstract class for error calculation"""

    def error(self, pinn: PINN) -> float:
        """
        Calculates error (😮) in the norm, so:
        |pinn - u| / |u|
        """
        raise NotImplementedError()

