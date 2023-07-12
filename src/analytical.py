from __future__ import annotations

from abc import ABC

import torch

from src.params import Params


class Analytical(ABC):
    """Abstract class that represent analytical solution to PDE"""
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplemented()
    
    def dx(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplemented()

class AnalyticalAD(Analytical):
    """Analytical solution to Advection-Diffusion equation
    
    Instances of this class can be called as functions"""
    
    def __init__(self, eps: float):
        self.eps = eps

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Solution for given x's"""
        eps = torch.tensor(self.eps)
        y_ana = 2*(1-torch.exp((x-1)/eps))/(1-torch.exp(-2/eps)) + x -1
        return y_ana
    
    def dx(self, x: torch.Tensor) -> torch.Tensor:
        """Derivative of the solution"""
        eps = torch.tensor(self.eps)
        y_dx = (-2) / (1-torch.exp(-2/eps)) * 1/eps * torch.exp((x-1)/eps) + 1
        return y_dx

    @classmethod
    def from_params(cls, params: Params) -> AnalyticalAD:
        return cls(params.eps)
    
class AnalyticalSmooth(Analytical):
    """Analytical solution to Diffusion equation
    
    Instances of this class can be called as functions"""
    
    def __init__(self):
        pass

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Solution for given x's"""
        return torch.sin(torch.pi * (x+1))
    
    def dx(self, x: torch.Tensor) -> torch.Tensor:
        """Derivative of the solution"""
        return torch.pi * torch.cos(torch.pi * (x+1))

    @classmethod
    def from_params(cls, params: Params) -> AnalyticalSmooth:
        return cls()
    

class AnalyticalDelta(Analytical):
    """Analytical solution to PDE: $ eps * u'' = delta_x0 $

    Instances of this class can be called as functions.
    """
    
    def __init__(self, eps: float, Xd: float):
        self.eps = eps
        self.Xd = Xd

    def __call__(self, x: torch.Tensor):
        """Solution for given x's - basically a pyramid, with summit at x = x0"""
        Xd = self.Xd
        x_left = (x <= Xd)
        x_right = (x > Xd)
        return  x_left * self.left(x) + x_right * self.right(x)

    def left(self, x:torch.Tensor):
        """Linear function for x < x0"""
        return 1/(2*self.eps)*(1-self.Xd)*(x+1)

    def right(self, x: torch.Tensor):
        """Linear function for x > x0"""
        return 1/(2*self.eps)*(self.Xd+1)*(1-x)

    def dx(self, x):
        """Derivative of the solution - two constant functions"""
        Xd = self.Xd
        eps = self.eps
        x_left = (x <= Xd)
        x_right = (x > Xd)
        return x_left * self.left_dx(x) + x_right * self.right_dx(x)

    def left_dx(self, x: torch.Tensor):
        """Constant function for x < x0"""
        return x*0 + (1/(2*self.eps) * (1 - self.Xd))

    def right_dx(self, x: torch.Tensor):
        """Constant function for x > x0"""
        return x*0 - (1/(2*self.eps) * (self.Xd + 1))

    @classmethod
    def from_params(cls, params: Params) -> AnalyticalDelta:
        return cls(params.eps, params.Xd)


def analytical_from_params(params: Params) -> Analytical:
    """Return instance of proper AnalyticalSolution class based on Params"""
    if params.equation == "ad":
        return AnalyticalAD(params.eps)
    if params.equation == "delta":
        return AnalyticalDelta(params.eps, params.Xd)
    if params.equation == "smooth":
        return AnalyticalSmooth()
    raise ValueError(f"Wrong equation in params: {params.equation}")
