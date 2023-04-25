from __future__ import annotations
from abc import ABC
import math
from sympy import *

from typing import List
import numpy as np
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
        if params.test_func == "poly":
            return PolyBase(params.n_test_func)
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
        left0 = x <= self._tip_x(n-1)
        left = torch.logical_and(x > self._tip_x(n-1), x <= self._tip_x(n))
        right = torch.logical_and(x > self._tip_x(n), x <= self._tip_x(n+1))
        right0 = x > self._tip_x(n+1)
        left_y =   x * 1.0 / self._delta_x() + 1.0 / self._delta_x() - n + 1
        right_y = -x * 1.0 / self._delta_x() - 1.0 / self._delta_x() + n + 1
        return left0 * 0.0 + left * left_y + right * right_y + right0 * 0.0
    
    def dx(self, x: torch.Tensor, n: int) -> torch.Tensor:
        left0 = x <= self._tip_x(n-1)
        left = torch.logical_and(x > self._tip_x(n-1), x <= self._tip_x(n))
        right = torch.logical_and(x > self._tip_x(n), x <= self._tip_x(n+1))
        right0 = x > self._tip_x(n+1)
        return left0 * 0.0 + left / self._delta_x() - right / self._delta_x() + right0 * 0.0
    
    def calculate_matrix(self, eps: float, n_test_func: int) -> torch.Tensor:
        matrix = torch.zeros(n_test_func, n_test_func)
        for i in range(n_test_func):
            matrix[i, i] = eps * (1.0 / self._delta_x())**2 * self._delta_x() * 2
        for i in range(n_test_func - 1):
            matrix[i, i+1] = -1.0 / self._delta_x() * eps
            matrix[i+1, i] = -1.0 / self._delta_x() * eps
        return torch.inverse(matrix)
    
    def _tip_x(self, n) -> float:
        return -1.0 + n*self._delta_x()
    
    def _delta_x(self) -> float:
        return 2.0/(self.N+1)

class PolyBase(BaseFun):
    def __init__(self, N: int, log=True):
        self.log = log
        if log:
            print("Generating polynomial base. Hold tight...")
        base, base_norm = self._gram_schmidt(N)
        self.fs = [self._to_lambda(sym) for sym in base_norm]
        self.dxs = [self._to_lambda(diff(sym)) for sym in base_norm]

    def __call__(self, x: torch.Tensor, n: int) -> torch.Tensor:
        return self.fs[n-1](x)      # n starts at 1
    
    def dx(self, x: torch.Tensor, n: int) -> torch.Tensor:
        return self.dxs[n-1](x)     # n starts at 1
    
    def divider(self, n: int) -> float:
        return 1.0                  # Orthonormal

    def calculate_matrix(self, n_test_func: int) -> torch.Tensor:
        matrix = torch.zeros(n_test_func, n_test_func)
        for i in range(n_test_func):
            matrix[i,i] = 1.0
        return torch.inverse(matrix)

    def _inner_prod(self, u,v,x=symbols('x')):
        return integrate(diff(u,x)*diff(v,x), (x, -1, 1))

    def _proj_u(self, u,v):
        return self._inner_prod(u,v)/(self._inner_prod(u,u))*u

    def _gram_schmidt(self, N: int):
        x = symbols('x')
        basis = []
        basis_norm = []

        basis.append(factor(x**2-1))
        basis_norm.append(expand(basis[0]/sqrt(self._inner_prod(basis[0],basis[0]))))

        for n in range(1,N):

            vk = x*basis[n-1]

            uk = 1*vk

            for j in range(n):

                uk -= self._proj_u(basis[j],vk)
            basis.append(expand(uk))
            basis_norm.append(expand(uk/sqrt(self._inner_prod(uk,uk))))
            if self.log and (n+1)%10 == 0:
                print(f"\t{n+1}/{N}")

        return basis, basis_norm

    def _to_lambda(self, sym):
        proper_sqrt = str(sym).replace("sqrt", "math.sqrt")
        with_lambda = f"lambda x: {proper_sqrt}"
        return eval(with_lambda)
