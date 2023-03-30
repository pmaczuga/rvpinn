from typing import Callable
import torch

def midpoints(x: torch.Tensor):
    return (x[0:-1]+x[1:])/2

def dx_between_points(x: torch.Tensor):
    return (x[1:] - x[0:-1])

def midpoint_int(f: Callable, x: torch.Tensor) -> torch.Tensor:
    m_points = midpoints(x)
    dxs = dx_between_points(x)
    val = f(m_points)
    return (dxs * val).sum()
