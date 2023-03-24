from __future__ import annotations

import math
import numpy as np
import torch

from src.loss.loss import Loss
from src.pinn import PINN, dfdx, f
from src.params import Params

class LossAD(Loss):
    def __init__(
         self, 
         x: torch.Tensor,
         eps: float,
         n_test_func: int
    ):
        self.x = x
        self.eps = eps
        self.n_test_func = n_test_func


    # Allows to call object as function
    def __call__(self, pinn: PINN) -> torch.Tensor:
        x = self.x
        eps = self.eps
        n_test_func = self.n_test_func
        device = pinn.get_device()

        beta = 1
        dx=2.0/len(x)

        final_loss=0.0
    
        val = dfdx(pinn, x, order=1) #this can be precomputed to save time
        interior_loss_trial1 = eps * val
        interior_loss_trial2 = beta * val
        
        for n in range(1,n_test_func): 
            interior_loss_test1 = n * math.pi/2 * torch.cos(n*math.pi*(x+1)/2)  #we can precompute this terms also  
            interior_loss_test2 = torch.sin(n*math.pi*(x+1)/2)
            inte1 = interior_loss_trial1.mul(interior_loss_test1)
            inte2 = interior_loss_trial2.mul(interior_loss_test2)

            val1 = torch.trapz(inte1.reshape(1,-1).to(device),dx=dx).to(device).sum()
            val2 = torch.trapz(inte2.reshape(1,-1).to(device),dx=dx).to(device).sum()
            val3 = torch.trapz(interior_loss_test2.reshape(1,-1).to(device),dx=dx).to(device).sum()

            interior_loss = val1 + val2 - val3
            # update the final MSE loss 
            final_loss+= 4/(eps*(n*math.pi)**2)*interior_loss.sum().pow(2) 


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
        return cls(x=x, eps = params.eps, n_test_func = params.n_test_func)
