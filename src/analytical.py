from __future__ import annotations

from abc import ABC
import numpy as np

import torch

from src.params import Params


class Analytical(ABC):
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplemented()
    
    def dx(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplemented()

class AnalyticalAD(Analytical):
    def __init__(self, eps: float):
        self.eps = eps

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        eps = torch.tensor(self.eps)
        y_ana = 2*(1-torch.exp((x-1)/eps))/(1-torch.exp(-2/eps)) + x -1
        return y_ana
    
    def dx(self, x: torch.Tensor) -> torch.Tensor:
        eps = torch.tensor(self.eps)
        y_dx = (-2) / (1-torch.exp(-2/eps)) * 1/eps * torch.exp((x-1)/eps) + 1
        return y_dx

    @classmethod
    def from_params(cls, params: Params) -> AnalyticalAD:
        return cls(params.eps)
    

class AnalyticalDelta(Analytical):
    def __init__(self, eps: float, Xd: float):
        self.eps = eps
        self.Xd = Xd

    # TODO - check
    def __call__(self, x: torch.Tensor):
        Xd = self.Xd
        x_left = (x <= Xd)
        x_right = (x > Xd)
        return  x_left * self.left(x) + x_right * self.right(x)

    def left(self, x:torch.Tensor):
        return 1/(2*self.eps)*(1-self.Xd)*(x+1)

    def right(self, x: torch.Tensor):
        return 1/(2*self.eps)*(self.Xd+1)*(1-x)

    def dx(self, x):
        Xd = self.Xd
        eps = self.eps
        x_left = (x <= Xd)
        x_right = (x > Xd)
        return x_left * self.left_dx(x) + x_right * self.right_dx(x)

    def left_dx(self, x: torch.Tensor):
        return x*0 + (1/(2*self.eps) * (1 - self.Xd))

    def right_dx(self, x: torch.Tensor):
        return x*0 - (1/(2*self.eps) * (self.Xd + 1))

    @classmethod
    def from_params(cls, params: Params) -> AnalyticalDelta:
        return cls(params.eps, params.Xd)


def analytical_from_params(params: Params) -> Analytical:
    if params.equation == "ad":
        return AnalyticalAD(params.eps)
    if params.equation == "delta":
        return AnalyticalDelta(params.eps, params.Xd)
    raise ValueError(f"Wrong equation in params: {params.equation}")
