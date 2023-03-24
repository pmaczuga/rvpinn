import math
from typing import Tuple
import numpy as np
import torch
from src.analytical import AnalyticalDelta
from src.pinn import PINN, dfdx
from src.nn_error import NNError
from scipy import integrate

class NNErrorDelta(NNError):
    def __init__(self, eps: float,  Xd: float, n_points_error: int, n_test_func: int):
        self.eps = eps
        self.Xd = Xd
        self.n_points_error = n_points_error
        self.n_test_func = n_test_func

    def error(self, pinn: PINN) -> float:
        eps = self.eps
        Xd = self.Xd
        n_points_error = self.n_points_error
        device = pinn.get_device()
        x1, x2 = self.prepare_twin_x(n_points_error, Xd, device)
        analytical = AnalyticalDelta(eps, Xd)
        val1 = dfdx(pinn, x1, order=1)
        val2 = dfdx(pinn, x2, order=1)
        
        u1 = analytical.left_dx(x1)
        u2 = analytical.right_dx(x2)

        up1 = (u1 - val1)**2
        up2 = (u2 - val2)**2
    
        up1 = up1.flatten().detach()
        up2 = up2.flatten().detach()
        x1 = x1.detach().flatten()
        x2 = x2.detach().flatten()

        up1 = integrate.simpson(up1, x=x1)
        up2 = integrate.simpson(up2, x=x2)

        up = up1 + up2
        x = u1[0]
        down = u1[0].item()**2*(Xd - 1.0) + u2[0].item()**2*(1.0 - Xd)

        return np.sqrt(up) / np.sqrt(down)

    def norm(self, pinn: PINN) -> float:
        eps = self.eps
        Xd = self.Xd
        n_test_func = self.n_test_func
        device = pinn.get_device()
        x = self.prepare_x(self.n_points_error, device)
    
        final_loss = 0.0
        
        val = dfdx(pinn, x, order=1)
        interior_loss_trial1 = eps*val

        for n in range(1, n_test_func):
            interior_loss_test1 = n * math.pi/2 * torch.cos(n*math.pi*(x+1)/2)
            interior_loss = interior_loss_trial1.mul(interior_loss_test1)
            interior_loss = interior_loss.detach().flatten()
            interior_loss = integrate.simpson(interior_loss, x=x.detach().flatten())-np.sin(n*math.pi*(Xd+1)/2)
            # update the final MSE loss 
            final_loss+= 4/(eps*(n*math.pi)**2)*interior_loss**2
    
        return final_loss
    
    def prepare_twin_x(self, n_points_error: int, Xd: float, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        n1 = int(np.floor((Xd + 1.0) / (1.0 - Xd)))
        n2 = n_points_error - n1
        x1 = torch.linspace(-1.0, Xd, n1).reshape(-1, 1).to(device)
        x2 = torch.linspace(Xd, 1.0, n2).reshape(-1, 1).to(device)
        x1.requires_grad = True
        x2.requires_grad = True
        return x1, x2
