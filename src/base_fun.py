from __future__ import annotations
from abc import ABC
import math
from typing import List

import torch

from src.params import Params

def precompute_base(base_fun: BaseFun, x: torch.Tensor, eps: float, n_test_func: int) -> PrecomputedBase:
    return PrecomputedBase(base_fun, x, eps, n_test_func)

class PrecomputedBase():
    def __init__(self, base_fun: BaseFun, x: torch.Tensor, eps: float, n_test_func: int):
        self.base_fun = base_fun
        self.n_test_func = n_test_func
        self._vals = [base_fun(x, n) for n in range(1, n_test_func + 1)]
        self._dxs = [base_fun.dx(x, n) for n in range(1, n_test_func + 1)]
        self._matrix = self.base_fun.calculate_matrix(eps, n_test_func)

    def get(self, n: int) -> torch.Tensor:
        return self._vals[n-1]

    def get_dx(self, n: int) -> torch.Tensor:
        return self._dxs[n-1]

    def get_matrix(self) -> torch.Tensor:
        return self._matrix

class BaseFun(ABC):
    def __call__(self, x: torch.Tensor, n: int) -> torch.Tensor:
        raise NotImplementedError()

    def dx(self, x: torch.Tensor, n: int) -> torch.Tensor:
        raise NotImplementedError()
    
    def divider(self, n: int) -> float:
        raise NotImplementedError()
    
    def calculate_matrix(self, eps: float, n_test_func: int) -> torch.Tensor:
        raise NotImplementedError()

    @classmethod
    def from_params(cls, params: Params) -> BaseFun:
        if params.test_func == "sin":
            return SinBase()
        if params.test_func == "fem":
            return FemBase(params.n_test_func)
        if params.test_func == "mixed":
            N1 = int(math.ceil(params.n_test_func / 2))
            N2 = params.n_test_func = N1
            return MixedBase(N1, N2)
        if params.test_func.startswith("mixed"):
            import re
            N2 = int(re.findall(r'\d+', params.test_func)[0])
            return MixedBase(params.n_test_func - N2, N2)
        raise ValueError(f"Not supported test_func in params: {params.test_func}")


class SinBase(BaseFun):
    def __call__(self, x: torch.Tensor, n: int) -> torch.Tensor:
        return torch.sin(n*math.pi*(x+1)/2)
    
    def dx(self, x: torch.Tensor, n: int) -> torch.Tensor:
        return n * math.pi/2 * torch.cos(n*math.pi*(x+1)/2)
    
    def divider(self, n: int) -> float:
        return (n*math.pi)**2 / 4.0
    
    def calculate_matrix(self, eps: float, n_test_func: int) -> torch.Tensor:
        matrix = torch.zeros(n_test_func, n_test_func)
        for i in range(n_test_func):
            n = i+1
            matrix[i,i] = eps * (n*math.pi)**2 / 4.0
        return torch.inverse(matrix)

class FemBase(BaseFun):
    def __init__(self, N: int, log=True):
        self.N = N

    def __call__(self, x: torch.Tensor, n: int) -> torch.Tensor:
        left_x0 = x <= self.tip_x(n-1)
        left_x = torch.logical_and(x > self.tip_x(n-1), x <= self.tip_x(n))
        right_x = torch.logical_and(x > self.tip_x(n), x <= self.tip_x(n+1))
        right_x0 = x > self.tip_x(n+1)
        left =  self.left(x, n)
        right = self.right(x, n)
        return left_x0 * 0.0 + left_x * left + right_x * right + right_x0 * 0.0
    
    def left(self, x: torch.Tensor, n: int) -> torch.Tensor:
        return x * 1.0 / self.delta_x() + 1.0 / self.delta_x() - n + 1
    
    def right(self, x: torch.Tensor, n: int) -> torch.Tensor:
        return -x * 1.0 / self.delta_x() - 1.0 / self.delta_x() + n + 1

    def dx(self, x: torch.Tensor, n: int) -> torch.Tensor:
        left_x0 = x <= self.tip_x(n-1)
        left_x = torch.logical_and(x > self.tip_x(n-1), x <= self.tip_x(n))
        right_x = torch.logical_and(x > self.tip_x(n), x <= self.tip_x(n+1))
        right_x0 = x > self.tip_x(n+1)
        left = self.dx_left(x, n)
        right = self.dx_right(x, n)
        return left_x0 * 0.0 + left_x * left + right_x * right + right_x0 * 0.0
    
    def dx_left(self, x: torch.Tensor, n: int) -> torch.Tensor:
        return x * 0.0 + 1.0 / self.delta_x()

    def dx_right(self, x: torch.Tensor, n: int) -> torch.Tensor:
        return x * 0.0 - 1.0 / self.delta_x()

    def calculate_matrix(self, eps: float, n_test_func: int) -> torch.Tensor:
        matrix = torch.zeros(n_test_func, n_test_func)
        for i in range(n_test_func):
            matrix[i, i] = eps * (1.0 / self.delta_x())**2 * self.delta_x() * 2
        for i in range(n_test_func - 1):
            matrix[i, i+1] = -1.0 / self.delta_x() * eps
            matrix[i+1, i] = -1.0 / self.delta_x() * eps
        return torch.inverse(matrix)
    
    def tip_x(self, n) -> float:
        return -1.0 + n*self.delta_x()
    
    def delta_x(self) -> float:
        return 2.0/(self.N+1)


class MixedBase(BaseFun):
    def __init__(self, N1: int, N2: int, log=True):
        self.N1 = N1
        self.N2 = N2
        self.sin = SinBase()
        self.fem = FemBase(N2)

    def __call__(self, x: torch.Tensor, n: int) -> torch.Tensor:
        if n <= self.N1:
            return self.sin(x, n)
        else:
            return self.fem(x, n - self.N1) 
    
    def dx(self, x: torch.Tensor, n: int) -> torch.Tensor:
        if n <= self.N1:
            return self.sin.dx(x, n)
        else:
            return self.fem.dx(x, n - self.N1) 

    def calculate_matrix(self, eps: float, n_test_func: int) -> torch.Tensor:
        delta_x = self.fem.delta_x()
        matrix = torch.zeros(n_test_func, n_test_func)
        # sin
        for i in range(n_test_func):
            n = i+1
            matrix[i,i] = eps * (n*math.pi)**2 / 4.0
        # FEM
        for i in range(self.N1, self.N2):
            matrix[i, i] = eps * (1.0 / delta_x)**2 * delta_x * 2
        for i in range(self.N1, self.N2 - 1):
            matrix[i, i+1] = -1.0 / delta_x * eps
            matrix[i+1, i] = -1.0 / delta_x * eps
        # There should also be non zero values across sin and fem
        # But this approximation should do
        return torch.inverse(matrix)
    