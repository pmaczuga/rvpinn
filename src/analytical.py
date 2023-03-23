from __future__ import annotations

from abc import ABC

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
    def __init__(self, eps:float, Xd: float):
        self.eps = eps
        self.Xd = Xd

    def __call__(self, x: torch.Tensor):
        x_left = (x <= self.Xd)
        x_right = (x > self.Xd)
        y_ana_np1 = 1/self.eps*(1-self.Xd)*(x+1) #(1-np.exp((xnp-1)/eps))/(1-np.exp(-1/eps)) + xnp - 1#(np.exp(1/eps)-np.exp((xnp)/eps))/(np.exp(1/eps) -1)
        y_ana_np2 = 1/self.eps*(self.Xd+1)*(1-x)
        return  x_left * y_ana_np1 + x_right * y_ana_np2

    def dx(self, x):
        x_left = (x <= self.Xd)
        x_right = (x > self.Xd)
        return x_left * 1/self.eps * (1 - self.Xd) - x_right * 1/self.eps * (self.Xd + 1)

    @classmethod
    def from_params(cls, params: Params) -> AnalyticalDelta:
        return cls(params.eps, params.Xd)


def analytical_from_params(params: Params) -> Analytical:
    if params.equation == "ad":
        return AnalyticalAD(params.eps)
    if params.equation == "delta":
        return AnalyticalDelta(params.eps, params.Xd)
    raise ValueError(f"Wrong equation in params: {params.equation}")
