from __future__ import annotations
from abc import ABC
import math

from typing import List
import numpy as np
import torch

def precompute_base(base_fun: BaseFun, x: torch.Tensor, n_test_func: int) -> PrecomputedBase:
    return PrecomputedBase(base_fun, x, n_test_func)

class PrecomputedBase():
    def __init__(self, base_fun: BaseFun, x: torch.Tensor, n_test_func: int):
        print("Here!")
        self.base_fun = base_fun
        self.n_test_func = n_test_func
        self.val = [base_fun(x, n) for n in range(1, n_test_func + 1)]
        self.dx = [base_fun.dx(x, n) for n in range(1, n_test_func + 1)]

    def get(self, n: int) -> torch.Tensor:
        return self.val[n-1]

    def get_dx(self, n: int) -> torch.Tensor:
        return self.dx[n-1]


class BaseFun(ABC):
    def __call__(self, x: torch.Tensor, i: int) -> torch.Tensor:
        raise NotImplementedError()

    def dx(self, x: torch.Tensor, i: int) -> torch.Tensor:
        raise NotImplementedError()
    
    def divider(self, n: int) -> float:
        raise NotImplementedError()
    

class SinBase(BaseFun):
    def __call__(self, x: torch.Tensor, n: int) -> torch.Tensor:
        return torch.sin(n*math.pi*(x+1)/2)
    
    def dx(self, x: torch.Tensor, n: int) -> torch.Tensor:
        return n * math.pi/2 * torch.cos(n*math.pi*(x+1)/2)
    
    def divider(self, n: int) -> float:
        return (n*math.pi)**2 / 4.0
