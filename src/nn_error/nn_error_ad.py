from __future__ import annotations

import math
import numpy as np
import torch
from scipy import integrate

from src.integration import IntegrationRule, get_int_rule
from src.base_fun import BaseFun, PrecomputedBase, SinBase, precompute_base
from src.params import Params
from src.analytical import AnalyticalAD
from src.pinn import PINN, dfdx
from src.nn_error import NNError
from src.utils import right_centered_distribution

class NNErrorAD(NNError):
    def __init__(self, 
                 eps: float, 
                 n_points_error: int, 
                 precomputed_base: PrecomputedBase,
                 integration_rule_norm: IntegrationRule,
                 integration_rule_error: IntegrationRule):
        self.eps = eps
        self.n_points_error = n_points_error
        self.precomputed_base = precomputed_base
        self.integration_rule_norm = integration_rule_norm
        self.integration_rule_error = integration_rule_error

    def error(self, pinn: PINN) -> float:
        eps = self.eps
        int_rule = self.integration_rule_error
        device = pinn.get_device()
        analytical = AnalyticalAD(eps)
        x = self.prepare_x(self.n_points_error, device)
        x, dx = int_rule.prepare_x_dx(x)

        val = dfdx(pinn, x, order=1)
        ana = analytical.dx(x)

        up   = (val - ana)**2
        down = ana**2

        up = up.detach().flatten()
        down = down.detach().flatten()
        x = x.detach().flatten()

        up   = int_rule.int_using_x_dx(up, x, dx)
        down = int_rule.int_using_x_dx(down, x, dx)

        return math.sqrt(up) / math.sqrt(down)

    def norm(self, pinn: PINN) -> float:
        eps = self.eps
        precomputed_base = self.precomputed_base
        int_rule = self.integration_rule_norm
        base_fun = precomputed_base.base_fun
        device = pinn.get_device()
        x = self.prepare_x(self.n_points_error, device)
        x, dx = int_rule.prepare_x_dx(x)

        beta = 1

        final_loss = 0.0
    
        val = dfdx(pinn, x, order=1) #this can be precomputed to save time
        interior_loss_trial1 = eps * val
        interior_loss_trial2 = beta * val
        
        for n in range(1, precomputed_base.n_test_func + 1): 
            interior_loss_test1 = precomputed_base.get_dx(n)
            interior_loss_test2 = precomputed_base.get(n)
            inte1 = interior_loss_trial1.mul(interior_loss_test1)
            inte2 = interior_loss_trial2.mul(interior_loss_test2)

            x_int = x.detach().flatten().double()
            dx_int = dx.detach().flatten().double()
            y1 = inte1.detach().flatten().double()
            y2 = inte2.detach().flatten().double()
            y3 = interior_loss_test2.detach().flatten().double()

            val1 = int_rule.int_using_x_dx(y1, x_int, dx_int).item()
            val2 = int_rule.int_using_x_dx(y2, x_int, dx_int).item()
            val3 = int_rule.int_using_x_dx(y3, x_int, dx_int).item()

            interior_loss = val1 + val2 - val3
            # update the final MSE loss 
            divider = base_fun.divider(n)
            final_loss+= 1/(eps * divider)*interior_loss**2 

        return final_loss
    
    @classmethod
    def prepare_x(cls, n_points_error: int, device: torch.device = torch.device("cpu")):

        # NOTE: 4 lines below are currently not used
        # When eps = 0.1, p is about 3
        # When eps = 0.01, p is about 6
        # Condition is to make it at least 1
        # p = -np.log2(self.eps) if self.eps < 0.5 else 1

        x = right_centered_distribution(-1.0, 1.0, n_points_error, p=1.5)
        x = x.reshape(-1, 1).to(device)
        x.requires_grad = True

        return x
    
    @classmethod
    def from_params(cls, params: Params) -> NNErrorAD:
        integration_rule_error = get_int_rule(params.integration_rule_error)
        integration_rule_norm = get_int_rule(params.integration_rule_norm)
        x = cls.prepare_x(params.n_points_error)
        x, dx = integration_rule_norm.prepare_x_dx(x)
        base_fun = BaseFun.from_params(params)
        precomputed_base = precompute_base(base_fun, x, params.n_test_func)
        return cls(params.eps, 
                   params.n_points_error, 
                   precomputed_base, 
                   integration_rule_norm, 
                   integration_rule_error)
