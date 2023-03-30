from __future__ import annotations

import math
import torch

from src.base_fun import PrecomputedBase, SinBase, precompute_base
from src.loss.loss import Loss
from src.pinn import PINN, dfdx, f
from src.params import Params

class LossAD(Loss):
    def __init__(
         self, 
         x: torch.Tensor,
         eps: float,
         precomputed_base: PrecomputedBase
    ):
        self.x = x
        self.eps = eps
        self.precomputed_base = precomputed_base


    # Allows to call object as function
    def __call__(self, pinn: PINN) -> torch.Tensor:
        x = self.x
        eps = self.eps
        precomputed_base = self.precomputed_base
        base_fun = precomputed_base.base_fun
        device = pinn.get_device()

        beta = 1
        dx=2.0/len(x)

        final_loss=0.0
    
        val = dfdx(pinn, x, order=1) #this can be precomputed to save time
        interior_loss_trial1 = eps * val
        interior_loss_trial2 = beta * val
        
        for n in range(1, precomputed_base.n_test_func + 1): 
            interior_loss_test1 = precomputed_base.get_dx(n)
            interior_loss_test2 = precomputed_base.get(n)
            inte1 = interior_loss_trial1.mul(interior_loss_test1)
            inte2 = interior_loss_trial2.mul(interior_loss_test2)

            val1 = torch.trapz(inte1.reshape(1,-1).to(device),dx=dx).to(device).sum()
            val2 = torch.trapz(inte2.reshape(1,-1).to(device),dx=dx).to(device).sum()
            val3 = torch.trapz(interior_loss_test2.reshape(1,-1).to(device),dx=dx).to(device).sum()

            interior_loss = val1 + val2 - val3
            # update the final MSE loss 
            divider = base_fun.divider(n)
            final_loss+= 1.0 / (eps + divider) * interior_loss.sum().pow(2) 


        boundary_xf = x[-1].reshape(-1, 1) #last point = 1
        boundary_loss_xf = f(pinn, boundary_xf)
        #boundary_loss_xi = -eps * dfdx(nn_approximator, boundary_xi) + f(nn_approximator, boundary_xi)
        
        boundary_xi = x[0].reshape(-1, 1) #first point = 0
        boundary_loss_xi = f(pinn, boundary_xi)#-1.0
        #boundary_loss_xf = -eps * dfdx(nn_approximator, boundary_xf) + f(nn_approximator, boundary_xf)-1.0
        final_loss+= \
            (1)*boundary_loss_xi.pow(2).mean() + \
            (1)*boundary_loss_xf.pow(2).mean() 
        return final_loss
    
    @classmethod
    def from_params(cls, x: torch.Tensor, params: Params) -> LossAD:
        base_fun = SinBase()
        precomputed_base = precompute_base(base_fun, x, params.n_test_func)
        return cls(x, params.eps, precomputed_base)
