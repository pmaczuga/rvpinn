from __future__ import annotations
from abc import ABC
import math

from typing import List
import numpy as np
import torch

from src.params import Params

def precompute_base(base_fun: BaseFun, x: torch.Tensor, n_test_func: int) -> PrecomputedBase:
    return PrecomputedBase(base_fun, x, n_test_func)

class PrecomputedBase():
    def __init__(self, base_fun: BaseFun, x: torch.Tensor, n_test_func: int):
        self.base_fun = base_fun
        self.n_test_func = n_test_func
        self._vals = [base_fun(x, n) for n in range(1, n_test_func + 1)]
        self._dxs = [base_fun.dx(x, n) for n in range(1, n_test_func + 1)]

    def get(self, n: int) -> torch.Tensor:
        return self._vals[n-1]

    def get_dx(self, n: int) -> torch.Tensor:
        return self._dxs[n-1]


class BaseFun(ABC):
    def __call__(self, x: torch.Tensor, n: int) -> torch.Tensor:
        raise NotImplementedError()

    def dx(self, x: torch.Tensor, n: int) -> torch.Tensor:
        raise NotImplementedError()
    
    def divider(self, n: int) -> float:
        raise NotImplementedError()

    @classmethod
    def from_params(cls, params: Params) -> BaseFun:
        if params.test_func == "sin":
            return SinBase()
        if params.test_func == "poly":
            return PolyBase()
        raise ValueError(f"Not supported test_func in params: {params.test_func}")


class SinBase(BaseFun):
    def __call__(self, x: torch.Tensor, n: int) -> torch.Tensor:
        return torch.sin(n*math.pi*(x+1)/2)
    
    def dx(self, x: torch.Tensor, n: int) -> torch.Tensor:
        return n * math.pi/2 * torch.cos(n*math.pi*(x+1)/2)
    
    def divider(self, n: int) -> float:
        return (n*math.pi)**2 / 4.0

# TODO - Implement
# Only thing you need to make this work is to implement 3 methods below
# See SinBase for reference
# divider is what we divide loss in the end (=1 if orthonormal)
# However when implemented you probably want to change:
# - comment in params.ini
# - help message for parser in drvpinn.py
class PolyBase(BaseFun):
    def __call__(self, x: torch.Tensor, n: int) -> torch.Tensor:
        raise NotImplementedError("PolyBase is not implemented")
    
    def dx(self, x: torch.Tensor, n: int) -> torch.Tensor:
        raise NotImplementedError("PolyBase is not implemented")
    
    def divider(self, n: int) -> float:
        raise NotImplementedError("PolyBase is not implemented")
