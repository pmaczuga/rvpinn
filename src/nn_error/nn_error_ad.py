from __future__ import annotations

import math
import numpy as np
import torch
from scipy import integrate

from src.base_fun import BaseFun, PrecomputedBase, SinBase, precompute_base
from src.params import Params
from src.analytical import AnalyticalAD
from src.pinn import PINN, dfdx
from src.nn_error import NNError

class NNErrorAD(NNError):
    def __init__(self, eps: float, n_points_error: int, precomputed_base: PrecomputedBase):
        self.eps = eps
        self.n_points_error = n_points_error
        self.precomputed_base = precomputed_base

    def error(self, pinn: PINN) -> float:
        eps = self.eps
        device = pinn.get_device()
        analytical = AnalyticalAD(eps)
        x = self.prepare_x(self.n_points_error, device)

        val = dfdx(pinn, x, order=1)
        ana = analytical.dx(x)

        up   = (val - ana)**2
        down = ana**2

        up = up.detach().flatten()
        down = down.detach().flatten()
        x = x.detach().flatten()

        up   = integrate.simpson(up, x=x)
        down = integrate.simpson(down, x=x)

        return math.sqrt(up) / math.sqrt(down)

    def norm(self, pinn: PINN) -> float:
        eps = self.eps
        precomputed_base = self.precomputed_base
        base_fun = precomputed_base.base_fun
        device = pinn.get_device()
        x = self.prepare_x(self.n_points_error, device)

        beta = 1

        final_loss = 0.0
    
        val = dfdx(pinn, x, order=1) #this can be precomputed to save time
        interior_loss_trial1 = eps * val
        interior_loss_trial2 = beta * val
        
        for n in range(1, precomputed_base.n_test_func + 1): 
            interior_loss_test1 = precomputed_base.get_dx(n)
            interior_loss_test2 = precomputed_base.get(n)
            inte1 = interior_loss_trial1.mul(interior_loss_test1)
            inte2 = interior_loss_trial2.mul(interior_loss_test2)

            x_int = x.detach().flatten()
            y1 = inte1.detach().flatten()
            y2 = inte2.detach().flatten()
            y3 = interior_loss_test2.detach().flatten()

            val1 = integrate.simpson(y1, x=x_int)
            val2 = integrate.simpson(y2, x=x_int)
            val3 = integrate.simpson(y3, x=x_int)

            interior_loss = val1 + val2 - val3
            # update the final MSE loss 
            divider = base_fun.divider(n)
            final_loss+= 1/(eps * divider)*interior_loss**2 

        return final_loss
    
    @classmethod
    def prepare_x(cls, n_points_error: int, device: torch.device = torch.device("cpu")):
        x = torch.linspace(0.0, 1.0, n_points_error)
        
        # Function that grows from 0 to 1
        # The bigger the p, the slower it grows towards the end
        # When p = 1, then it becomes f(x) = x
        distr = lambda x, p: -(-x+1)**p + 1

        # NOTE: 4 lines below are currently not used
        # When eps = 0.1, p is about 3
        # When eps = 0.01, p is about 6
        # Condition is to make it at least 1
        # p = -np.log2(self.eps) if self.eps < 0.5 else 1

        real_x = distr(x, 2) * 2.0 - 1.0        # x is from 0 to 1, our domain is -1 to 1
        real_x = real_x.reshape(-1, 1).to(device)
        real_x.requires_grad = True

        return real_x
    
    @classmethod
    def from_params(cls, params: Params) -> NNErrorAD:
        x = cls.prepare_x(params.n_points_error)
        base_fun = BaseFun.from_params(params)
        precomputed_base = precompute_base(base_fun, x, params.n_test_func)
        return cls(params.eps, params.n_points_error, precomputed_base)
