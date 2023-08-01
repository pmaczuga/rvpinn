from __future__ import annotations

import math
from typing import Tuple
import numpy as np
import torch
from src.params import Params
from src.integration import IntegrationRule, get_int_rule
from src.analytical import AnalyticalDelta
from src.pinn import PINN, dfdx
from src.loss.nn_error import NNError

class NNErrorDelta(NNError):
    def __init__(self, 
                 eps: float,  
                 Xd: float, 
                 n_points_error: int, 
                 integration_rule_error: IntegrationRule,
                 ):
        self.eps = eps
        self.Xd = Xd
        self.n_points_error = n_points_error
        self.integration_rule_error = integration_rule_error

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

        up1 = self.integration_rule_error(up1_f, x1).detach().flatten()
        up2 = self.integration_rule_error(up2_f, x2).detach().flatten()

        # It is constant
        u1 = analytical.left_dx(x1)[0].item()
        u2 = analytical.right_dx(x1)[0].item()

        up = up1 + up2
        down = u1**2*(Xd + 1.0) + u2**2*(1.0 - Xd)

        return math.sqrt(up) / math.sqrt(down)
    
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
        integration_rule_error = get_int_rule(params.integration_rule_error)

        return cls(params.eps, 
                   params.Xd, 
                   params.n_points_error, 
                   integration_rule_error,
                   )
