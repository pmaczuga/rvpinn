from __future__ import annotations

import math
import numpy as np
import torch
from scipy import integrate

from src.integration import IntegrationRule, get_int_rule
from src.base_fun import BaseFun, PrecomputedBase, SinBase, precompute_base
from src.params import Params
from src.analytical import AnalyticalAD, AnalyticalSmooth
from src.pinn import PINN, dfdx
from src.nn_error import NNError
from src.utils import right_centered_distribution

class NNErrorSmooth(NNError):
    def __init__(self, 
                 n_points_error: int, 
                 precomputed_base: PrecomputedBase,
                 integration_rule_norm: IntegrationRule,
                 integration_rule_error: IntegrationRule,
                 divide_by_test: bool,
                 ):
        self.n_points_error = n_points_error
        self.precomputed_base = precomputed_base
        self.integration_rule_norm = integration_rule_norm
        self.integration_rule_error = integration_rule_error
        self.divider = precomputed_base.n_test_func if divide_by_test else 1.0

    def error(self, pinn: PINN) -> float:
        int_rule = self.integration_rule_error
        device = pinn.get_device()
        analytical = AnalyticalSmooth()
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
        precomputed_base = self.precomputed_base
        int_rule = self.integration_rule_norm
        base_fun = precomputed_base.base_fun
        device = pinn.get_device()
        x = self.prepare_x(self.n_points_error, device)
        
        L = torch.zeros(precomputed_base.n_test_func)

        for n in range(1, precomputed_base.n_test_func + 1): 
            n_points = int(np.ceil(torch.numel(x) / precomputed_base.n_test_func))

            left  = base_fun._tip_x(n-1)
            mid   = base_fun._tip_x(n)
            right = base_fun._tip_x(n+1)

            x_left  = torch.linspace(left, mid,  n_points, requires_grad=True).reshape(-1, 1)
            x_right = torch.linspace(mid, right, n_points, requires_grad=True).reshape(-1, 1)

            val_left = dfdx(pinn, x_left, order=1) 
            rhs_left = torch.pi**2 * torch.sin(torch.pi * (x_left+1))
            val_right = dfdx(pinn, x_right, order=1) 
            rhs_right = torch.pi**2 * torch.sin(torch.pi * (x_right+1))

            base_left  = base_fun(x_left,  n)
            base_right = base_fun(x_right, n)
            base_dx_left  = torch.full((n_points,),  1.0 / base_fun._delta_x(), requires_grad=True).reshape(-1, 1)
            base_dx_right = torch.full((n_points,), -1.0 / base_fun._delta_x(), requires_grad=True).reshape(-1, 1)

            inte1_left = val_left.mul(base_dx_left)
            inte1_right = val_right.mul(base_dx_right)
            inte2_left = rhs_left.mul(base_left)
            inte2_right = rhs_right.mul(base_right)

            x_left, dx_left   = int_rule.prepare_x_dx(x_left)
            x_right, dx_right = int_rule.prepare_x_dx(x_right)

            val1_left  = int_rule.int_using_x_dx(inte1_left, x_left, dx_left)
            val1_right = int_rule.int_using_x_dx(inte1_right, x_right, dx_right)
            val2_left  = int_rule.int_using_x_dx(inte2_left, x_left, dx_left)
            val2_right = int_rule.int_using_x_dx(inte2_right, x_right, dx_right)
            
            interior_loss = (val1_left + val1_right) - (val2_left + val2_right)
            L[n-1] = interior_loss.sum()

        G = precomputed_base.get_matrix()
        final_loss = torch.matmul(torch.matmul(L.T, G), L)
        final_loss = final_loss / self.divider

        return final_loss.item()

    
    @classmethod
    def from_params(cls, params: Params) -> NNErrorSmooth:
        integration_rule_error = get_int_rule(params.integration_rule_error)
        integration_rule_norm = get_int_rule(params.integration_rule_norm)
        x = cls.prepare_x(params.n_points_error)
        x, dx = integration_rule_norm.prepare_x_dx(x)
        base_fun = BaseFun.from_params(params)
        precomputed_base = precompute_base(base_fun, x, params.eps, params.n_test_func)
        return cls(params.n_points_error, 
                   precomputed_base, 
                   integration_rule_norm, 
                   integration_rule_error,
                   params.divide_by_test
                   )
