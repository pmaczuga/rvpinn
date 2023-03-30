from __future__ import annotations

import math
from typing import List
import numpy as np
import torch

from src.base_fun import BaseFun, PrecomputedBase, SinBase, precompute_base
from src.loss.loss import Loss
from src.pinn import PINN, dfdx, f
from src.params import Params

class LossDelta(Loss):
    def __init__(
         self, 
         x: torch.Tensor,
         eps: float,
         Xd: float,
         precomputed_base: PrecomputedBase,
    ):
        self.x = x
        self.eps = eps
        self.Xd = Xd
        self.precomputed_base = precomputed_base


    # Allows to call object as function
    def __call__(self, pinn: PINN) -> torch.Tensor:
        x = self.x
        eps = self.eps
        Xd = self.Xd
        precomputed_base = self.precomputed_base
        base_fun = self.precomputed_base.base_fun
        device = pinn.get_device()

        dx=2.0/len(x)
    
        final_loss = torch.tensor(0.0)
        
        val = dfdx(pinn, x, order=1)
        interior_loss_trial1 = eps*val

        for n in range(1, precomputed_base.n_test_func + 1):
            interior_loss_test1 = precomputed_base.get_dx(n)
            interior_loss = interior_loss_trial1.mul(interior_loss_test1)
            interior_loss = torch.trapz(interior_loss.reshape(1,-1).to(device), dx=dx).to(device).sum()
            interior_loss = interior_loss - base_fun(torch.tensor(Xd), n)
            # update the final MSE loss 
            divider = base_fun.divider(n)
            final_loss+= 1/(eps * divider)*interior_loss.pow(2) 

        boundary_xi = x[0].reshape(-1, 1) #first point = 0
        boundary_loss_xi = f(pinn, boundary_xi)


        boundary_xf = x[-1].reshape(-1, 1) #last point = 1
        boundary_loss_xf = f(pinn, boundary_xf)
        
        final_loss+= \
            (1)*boundary_loss_xi.pow(2).mean() + \
            (1)*boundary_loss_xf.pow(2).mean() 
        return final_loss
    
    @classmethod
    def from_params(cls, x: torch.Tensor, params: Params) -> LossDelta:
        base_fun = BaseFun.from_params(params)
        precomputed_base = precompute_base(base_fun, x, params.n_test_func)
        return cls(x, params.eps, params.Xd, precomputed_base)
