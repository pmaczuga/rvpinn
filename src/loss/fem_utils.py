import torch
import numpy as np

from src.base_fun import FemBase

def prepare_x_for_fem_int(n_test_func: int, n_points_x_fem: int, requires_grad=False) -> torch.Tensor:
    base_fun = FemBase(n_test_func)
    
    x_norm, _ = np.polynomial.legendre.leggauss(n_points_x_fem)
    x_norm = torch.from_numpy(x_norm).float()
    xs = []
    for i in range(0, n_test_func + 1):
        a = base_fun.tip_x(i)
        b = base_fun.tip_x(i+1)
        x = x_norm * (b-a)/2 + (a+b)/2
        xs.append(x)
    x = torch.cat(xs)
    x.requires_grad = requires_grad
    return x

def gauss_weights(n: int) -> torch.Tensor:
    _, w = np.polynomial.legendre.leggauss(n)
    return torch.from_numpy(w).float()
