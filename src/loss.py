import math
import numpy as np
import torch

from src.pinn import dfdx, f

def compute_loss(
    pinn, 
    x: torch.Tensor = None, 
    eps: float = 1.0,
    beta: float = 0.0,
    n_test_func: int = 50,
    device: torch.device = torch.device("cpu")
) -> torch.float:
    """Compute the full loss function as interior loss + boundary loss
    This custom loss function is fully defined with differentiable tensors therefore
    the .backward() method can be applied to it
    """

    dx=1.0/len(x)

    Xd = 0.5  #np.sqrt(2)/2
   
    final_loss=0.0
    
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
