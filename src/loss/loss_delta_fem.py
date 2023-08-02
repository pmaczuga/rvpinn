from __future__ import annotations

import math
from typing import List
import numpy as np
import torch

from src.integration import IntegrationRule, get_int_rule
from src.base_fun import BaseFun, FemBase, PrecomputedBase, SinBase, precompute_base, prepare_x_per_base
from src.loss.loss import Loss
from src.pinn import PINN, dfdx, f
from src.params import Params
from src.utils import prepare_x

class LossDeltaFem(Loss):
    '''
    Special Loss function for "delta" equation when using FEM 
    base functions (pyramids). Fixes problems with poor results
    when integrating over discontinuous function, which is caused
    by base function derivative. 
    '''

    def __init__(
         self, 
         xs: List[torch.Tensor],
         eps: float,
         Xd: float,
         base_fun: FemBase,
         gramm_matrix: torch.Tensor,
         n_test_func: int,
         integration_rule: IntegrationRule,
         divide_by_test: bool
    ):
        self.xs = xs
        self.eps = eps
        self.Xd = Xd
        self.base_fun = base_fun
        self.gramm_matrix = gramm_matrix
        self.n_test_func = n_test_func
        self.integration_rule = integration_rule
        self.divider = n_test_func if divide_by_test else 1.0


    # Allows to call object as function
    def __call__(self, pinn: PINN) -> torch.Tensor:
        return self.pde_loss(pinn) + self.boundary_loss(pinn)
    
    def pde_loss(self, pinn: PINN) -> torch.Tensor:
        eps = self.eps
        Xd = self.Xd
        int_rule = self.integration_rule
        base_fun = self.base_fun
        device = pinn.get_device()
        
        L = torch.zeros(self.n_test_func)

        for n in range(1, self.n_test_func + 1):
            x_left  = self.xs[n-1]
            x_right = self.xs[n]

            n_points_left  = x_left.numel()
            n_points_right = x_right.numel()

            val_left = dfdx(pinn, x_left, order=1) 
            val_right = dfdx(pinn, x_right, order=1) 

            trial1_left  = eps * val_left
            trial1_right = eps * val_right

            base_dx_left  = torch.full((n_points_left,),  1.0 / base_fun.delta_x(), requires_grad=True).reshape(-1, 1)
            base_dx_right = torch.full((n_points_right,), -1.0 / base_fun.delta_x(), requires_grad=True).reshape(-1, 1)

            inte1_left = trial1_left.mul(base_dx_left)
            inte1_right = trial1_right.mul(base_dx_right)          

            x_left, dx_left   = int_rule.prepare_x_dx(x_left)
            x_right, dx_right = int_rule.prepare_x_dx(x_right)

            val1_left  = int_rule.int_using_x_dx(inte1_left, x_left, dx_left)
            val1_right = int_rule.int_using_x_dx(inte1_right, x_right, dx_right)

            L[n-1] = (val1_left + val1_right) - base_fun(torch.tensor(Xd), n)

        G = self.gramm_matrix
        final_loss = torch.matmul(torch.matmul(L.T, G), L)
        final_loss = final_loss / self.divider

        return final_loss

    def boundary_loss(self, pinn: PINN) -> torch.Tensor:
        boundary_xi = self.xs[0][0].reshape(-1, 1) #first point = 0
        boundary_loss_xi = f(pinn, boundary_xi)


        boundary_xf = self.xs[-1][-1].reshape(-1, 1) #last point = 1
        boundary_loss_xf = f(pinn, boundary_xf)

        boundary_loss = \
            (1)*boundary_loss_xi.pow(2).mean() + \
            (1)*boundary_loss_xf.pow(2).mean() 

        return boundary_loss

    @classmethod
    def from_params(cls, params: Params, device: torch.device) -> LossDeltaFem:
        base_fun = FemBase(params.n_test_func)
        gramm_matrix = base_fun.calculate_matrix(params.eps, params.n_test_func)
        integration_rule = get_int_rule(params.integration_rule_loss)
        xs = prepare_x_per_base(base_fun, params.n_test_func, params.n_points_x_fem, device)
        return cls(xs, params.eps, params.Xd, base_fun, gramm_matrix, params.n_test_func, integration_rule, params.divide_by_test)
    