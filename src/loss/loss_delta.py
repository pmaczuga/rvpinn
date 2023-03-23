from __future__ import annotations

import math
import numpy as np
import torch

from src.loss.loss import Loss
from src.pinn import PINN, dfdx, f
from src.params import Params

class LossDelta(Loss):
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

        dx=1.0/len(x)

        Xd = 0.5  #np.sqrt(2)/2
    
        final_loss = 0.0
        
        val = dfdx(pinn, x, order=1)
        interior_loss_trial1 = eps*val

        for n in range(1, n_test_func):
            interior_loss_test1 = n * math.pi/2 * torch.cos(n*math.pi*(x+1)/2)
            interior_loss = interior_loss_trial1.mul(interior_loss_test1)
            interior_loss = torch.trapz(interior_loss.reshape(1,-1).to(device),dx=dx).to(device).sum()-np.sin(n*math.pi*(Xd+1)/2)
            # update the final MSE loss 
            final_loss+= 4/(eps*(n*math.pi)**2)*interior_loss.pow(2) 

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
        return cls(x=x, eps = params.eps, n_test_func = params.n_test_func)
