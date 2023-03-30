from __future__ import annotations

import math
from typing import Tuple
import numpy as np
import torch
from src.base_fun import BaseFun, PrecomputedBase, SinBase, precompute_base
from src.params import Params
from src.integration import midpoint_int
from src.analytical import AnalyticalDelta
from src.pinn import PINN, dfdx
from src.nn_error import NNError
from scipy import integrate

class NNErrorDelta(NNError):
    def __init__(self, eps: float,  Xd: float, n_points_error: int, precomputed_base: PrecomputedBase):
        self.eps = eps
        self.Xd = Xd
        self.n_points_error = n_points_error
        self.precomputed_base = precomputed_base

    def error(self, pinn: PINN) -> float:
        eps = self.eps
        Xd = self.Xd
        n_points_error = self.n_points_error
        device = pinn.get_device()
        x1, x2 = self.prepare_twin_x(n_points_error, Xd, device)
        analytical = AnalyticalDelta(eps, Xd)

        def up1_f(x):
            val = dfdx(pinn, x, order=1)
            u = analytical.left_dx(x)
            return (u - val)**2
        
        def up2_f(x):
            val = dfdx(pinn, x, order=1)
            u = analytical.right_dx(x)
            return (u-val)**2

        x1 = x1
        x2 = x2

        up1 = midpoint_int(up1_f, x=x1).detach().flatten()
        up2 = midpoint_int(up2_f, x=x2).detach().flatten()

        # It is constant
        u1 = analytical.left_dx(x1)[0].item()
        u2 = analytical.right_dx(x1)[0].item()

        up = up1 + up2
        down = u1**2*(Xd - 1.0) + u2**2*(1.0 - Xd)

        return math.sqrt(up) / math.sqrt(down)

    def norm(self, pinn: PINN) -> float:
        eps = self.eps
        Xd = self.Xd
        precomputed_base = self.precomputed_base
        base_fun = precomputed_base.base_fun
        device = pinn.get_device()
        x = self.prepare_x(self.n_points_error, device)
    
        final_loss = 0.0
        
        val = dfdx(pinn, x, order=1)
        interior_loss_trial1 = eps*val

        for n in range(1, precomputed_base.n_test_func + 1):
            interior_loss_test1 = precomputed_base.get_dx(n)
            interior_loss = interior_loss_trial1.mul(interior_loss_test1)
            interior_loss = interior_loss.detach().flatten()
            interior_loss = integrate.simpson(interior_loss, x=x.detach().flatten())
            interior_loss = interior_loss - base_fun(torch.tensor(Xd), n).item()

            # update the final MSE loss 
            divider = base_fun.divider(n)
            final_loss+= 1/(eps * divider)*interior_loss**2 
    
        return final_loss
    
    def prepare_twin_x(self, n_points_error: int, Xd: float, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        n1 = int(np.floor((Xd + 1.0) / (1.0 - Xd)))
        n2 = n_points_error - n1
        x1 = torch.linspace(-1.0, Xd-1e-3, n1).reshape(-1, 1).to(device)
        x2 = torch.linspace(Xd+1e-3, 1.0, n2).reshape(-1, 1).to(device)
        x1.requires_grad = True
        x2.requires_grad = True
        return x1, x2

    @classmethod
    def from_params(cls, params: Params) -> NNErrorDelta:
        x = cls.prepare_x(params.n_points_error)
        base_fun = BaseFun.from_params(params)
        precomputed_base = precompute_base(base_fun, x, params.n_test_func)
        return cls(params.eps, params.Xd, params.n_points_error, precomputed_base)
