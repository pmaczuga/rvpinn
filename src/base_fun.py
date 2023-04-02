from __future__ import annotations
from abc import ABC
import math
from sympy import *

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
            return PolyBase(params.n_test_func)
        raise ValueError(f"Not supported test_func in params: {params.test_func}")


class SinBase(BaseFun):
    def __call__(self, x: torch.Tensor, n: int) -> torch.Tensor:
        return torch.sin(n*math.pi*(x+1)/2)
    
    def dx(self, x: torch.Tensor, n: int) -> torch.Tensor:
        return n * math.pi/2 * torch.cos(n*math.pi*(x+1)/2)
    
    def divider(self, n: int) -> float:
        return (n*math.pi)**2 / 4.0


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
