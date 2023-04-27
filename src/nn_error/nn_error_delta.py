from __future__ import annotations

import math
from typing import Tuple
import numpy as np
import torch
from src.base_fun import BaseFun, PrecomputedBase, SinBase, precompute_base
from src.params import Params
from src.integration import IntegrationRule, TorchFunction, get_int_rule, midpoint_int
from src.analytical import AnalyticalDelta
from src.pinn import PINN, dfdx
from src.nn_error import NNError
from scipy import integrate

class NNErrorDelta(NNError):
    def __init__(self, 
                 eps: float,  
                 Xd: float, 
                 n_points_error: int, 
                 precomputed_base: PrecomputedBase,
                 integration_rule_norm: IntegrationRule,
                 integration_rule_error: IntegrationRule,
                 divide_by_test: bool
                 ):
        self.eps = eps
        self.Xd = Xd
        self.n_points_error = n_points_error
        self.precomputed_base = precomputed_base
        self.integration_rule_norm = integration_rule_norm
        self.integration_rule_error = integration_rule_error
        self.divider = precomputed_base.n_test_func if divide_by_test else 1.0

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

    def norm(self, pinn: PINN) -> float:
        eps = self.eps
        Xd = self.Xd
        precomputed_base = self.precomputed_base
        int_rule = self.integration_rule_norm
        base_fun = precomputed_base.base_fun
        device = pinn.get_device()
        x = self.prepare_x(self.n_points_error, device)
        x, dx = int_rule.prepare_x_dx(x)
        
        val = dfdx(pinn, x, order=1)
        interior_loss_trial1 = eps*val

        L = torch.zeros(precomputed_base.n_test_func)

        for n in range(1, precomputed_base.n_test_func + 1):
            interior_loss_test1 = precomputed_base.get_dx(n)
            interior_loss = interior_loss_trial1.mul(interior_loss_test1)
            interior_loss = int_rule.int_using_x_dx(interior_loss, x, dx).sum()
            interior_loss = interior_loss - base_fun(torch.tensor(Xd), n)

            L[n-1] = interior_loss
    
        G = precomputed_base.get_matrix()
        final_loss = torch.matmul(torch.matmul(L.T, G), L)
        final_loss = final_loss / self.divider

        return final_loss.item()
    
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
        integration_rule_norm = get_int_rule(params.integration_rule_norm)
        x = cls.prepare_x(params.n_points_error)
        base_fun = BaseFun.from_params(params)
        x, dx = integration_rule_norm.prepare_x_dx(x)
        precomputed_base = precompute_base(base_fun, x, params.eps, params.n_test_func)

        return cls(params.eps, 
                   params.Xd, 
                   params.n_points_error, 
                   precomputed_base, 
                   integration_rule_norm, 
                   integration_rule_error,
                   params.divide_by_test
                   )
