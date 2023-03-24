import math
import numpy as np
import torch
from scipy import integrate

from src.analytical import AnalyticalAD
from src.pinn import PINN, dfdx
from src.nn_error import NNError

class NNErrorAD(NNError):
    def __init__(self, eps: float, n_points_error: int, n_test_func: int):
        self.eps = eps
        self.n_points_error = n_points_error
        self.n_test_func = n_test_func

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

        return np.sqrt(up) / np.sqrt(down)

    def norm(self, pinn: PINN) -> float:
        eps = self.eps
        n_test_func = self.n_test_func
        device = pinn.get_device()
        x = self.prepare_x(self.n_points_error, device)

        beta = 1

        final_loss = 0.0
    
        val = dfdx(pinn, x, order=1) #this can be precomputed to save time
        interior_loss_trial1 = eps * val
        interior_loss_trial2 = beta * val
        
        for n in range(1,n_test_func): 
            interior_loss_test1 = n * math.pi/2 * torch.cos(n*math.pi*(x+1)/2)  #we can precompute this terms also  
            interior_loss_test2 = torch.sin(n*math.pi*(x+1)/2)
            inte1 = interior_loss_trial1.mul(interior_loss_test1)
            inte2 = interior_loss_trial2.mul(interior_loss_test2)

            x_int = x.flatten().detach()
            y1 = inte1.flatten().detach()
            y2 = inte2.flatten().detach()
            y3 = interior_loss_test2.flatten().detach()

            val1 = integrate.simpson(y1, x=x_int)
            val2 = integrate.simpson(y2, x=x_int)
            val3 = integrate.simpson(y3, x=x_int)

            interior_loss = val1 + val2 - val3
            # update the final MSE loss 
            final_loss+= 4/(eps*(n*math.pi)**2)*interior_loss**2

        return final_loss