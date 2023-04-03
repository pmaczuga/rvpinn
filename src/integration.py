from abc import ABC
from typing import Callable, Tuple
import torch

from scipy import integrate

TorchFunction = Callable[[torch.Tensor], torch.Tensor]
# IntegrationRule = Callable[[TorchFunction, torch.Tensor], torch.Tensor]

def midpoints(x: torch.Tensor):
    return (x[0:-1]+x[1:])/2

def dx_between_points(x: torch.Tensor):
    return (x[1:] - x[0:-1])

def add_halves_inbetween(a: torch.Tensor):
    b = (a[0:-1]+a[1:])/2
    c = torch.empty(a.shape[0] + b.shape[0], a.shape[1]).to(a.device)
    c[0::2] = a
    c[1::2] = b
    return c

class IntegrationRule(ABC):
    def __call__(self, f: TorchFunction, x: torch.Tensor) -> torch.Tensor:
        x, dx = self.prepare_x_dx(x)
        return self.int_using_x_dx(f(x), x, dx)

    def prepare_x_dx(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError()

    def int_using_x_dx(self, y: torch.Tensor, x: torch.Tensor, dx: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

class MidpointInt(IntegrationRule):
    def prepare_x_dx(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return midpoints(x), dx_between_points(x)
    
    def int_using_x_dx(self, y: torch.Tensor, x: torch.Tensor, dx: torch.Tensor) -> torch.Tensor:
        return (dx * y).sum()

class TrapzInt(IntegrationRule):
    def prepare_x_dx(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return x, dx_between_points(x)
    
    def int_using_x_dx(self, y: torch.Tensor, x: torch.Tensor, dx: torch.Tensor) -> torch.Tensor:
        avg_bases = (y[1:] + y[0:-1]) / 2.0
        return (dx * avg_bases).sum()

class SimpsonInt(IntegrationRule):
    """
    Delegated to scipy.integrate, so it will not work in loss function
    """
    
    def prepare_x_dx(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return x, dx_between_points(x)
    
    def int_using_x_dx(self, y: torch.Tensor, x: torch.Tensor, dx: torch.Tensor) -> torch.Tensor:
        x = x.detach().flatten()
        y = y.detach().flatten()
        res = integrate.simpson(y, x=x)
        return torch.tensor(res)

def midpoint_int(f: TorchFunction, x: torch.Tensor) -> torch.Tensor:
    m_points = midpoints(x)
    dxs = dx_between_points(x)
    val = f(m_points)
    return (dxs * val).sum()

def trapz_int(f: TorchFunction, x: torch.Tensor) -> torch.Tensor:
    dxs = dx_between_points(x)
    y = f(x)
    darea = (y[1:] + y[0:-1]) / 2.0
    return (dxs * darea).sum()

# Does NOT work on loss function - it does x.detach()
# def simpson_int(f: TorchFunction, x: torch.Tensor) -> torch.Tensor:
#     x = x.detach().flatten()
#     y = f(x)
#     res = integrate.simpson(y, x=x)
#     return torch.tensor(res)


def simpson_int(f: TorchFunction, x: torch.Tensor) -> torch.Tensor:

    h = x[1:] - x[0:-1]
    print(f"h.shape = {h.shape}")

    x = add_halves_inbetween(x)
    y = f(x)
    print(f"x.shape = {x.shape}")

    weights = torch.empty_like(x)
    weights[1:-1:2] = 4 * h
    weights[2:-1:2] = h[0:-1] + h[1:] 
    weights[0] = h[0]
    weights[-1] = h[-1]

    weights = 1/3 * weights

    return torch.sum(y * weights)

def get_int_rule(name: str) -> IntegrationRule:
    if name == 'midpoint':
        return MidpointInt()
    if name == 'trapz':
        return TrapzInt()
    if name == 'simpson':
        return SimpsonInt()
    raise ValueError(f'Integration rule "{name}" is not supported')
