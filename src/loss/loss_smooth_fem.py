from __future__ import annotations
from typing import List

import torch

from src.integration import IntegrationRule, get_int_rule
from src.base_fun import FemBase, prepare_x_per_base
from src.loss.loss import Loss
from src.pinn import PINN, dfdx, f
from src.params import Params

class LossSmoothFem(Loss):
    '''
    Special Loss function for "smooth" equation when using FEM 
    base functions (pyramids). Fixes problems with poor results
    when integrating over discontinuous function, which is caused
    by base function derivative. 
    '''

    def __init__(
         self, 
         xs: List[torch.Tensor],
         base_fun: FemBase,
         gramm_matrix: torch.Tensor,
         n_test_func: int,
         integration_rule: IntegrationRule,
         divide_by_test: bool
    ):
        self.xs = xs
        self.base_fun = base_fun
        self.gramm_matrix = gramm_matrix
        self.n_test_func = n_test_func
        self.integration_rule = integration_rule
        self.divider = n_test_func if divide_by_test else 1.0


    # Allows to call object as function
    def __call__(self, pinn: PINN) -> torch.Tensor:
        return self.pde_loss(pinn) + self.boundary_loss(pinn)
    
    def pde_loss(self, pinn: PINN) -> torch.Tensor:
        xs = self.xs
        int_rule = self.integration_rule
        base_fun = self.base_fun
        n_test_func = self.n_test_func
        device = pinn.get_device()
        
        L = torch.zeros(n_test_func)

        for n in range(1, n_test_func + 1):
            x_left  = xs[n-1]
            x_right = xs[n]

            n_points_left  = x_left.numel()
            n_points_right = x_right.numel()

            val_left = dfdx(pinn, x_left, order=1) 
            rhs_left = torch.pi**2 * torch.sin(torch.pi * (x_left+1))
            val_right = dfdx(pinn, x_right, order=1) 
            rhs_right = torch.pi**2 * torch.sin(torch.pi * (x_right+1))

            base_left  = base_fun(x_left,  n)
            base_right = base_fun(x_right, n)
            base_dx_left  = torch.full((n_points_left,),  1.0 / base_fun.delta_x(), requires_grad=True).reshape(-1, 1)
            base_dx_right = torch.full((n_points_right,), -1.0 / base_fun.delta_x(), requires_grad=True).reshape(-1, 1)

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

        G = self.gramm_matrix
        final_loss = torch.matmul(torch.matmul(L.T, G), L)
        final_loss = final_loss / self.divider

        return final_loss

    def boundary_loss(self, pinn: PINN) -> torch.Tensor:
        boundary_xf = self.xs[-1][-1].reshape(-1, 1) #last point = 1
        boundary_loss_xf = f(pinn, boundary_xf)
        #boundary_loss_xi = -eps * dfdx(nn_approximator, boundary_xi) + f(nn_approximator, boundary_xi)
        
        boundary_xi = self.xs[0][0].reshape(-1, 1) #first point = 0
        boundary_loss_xi = f(pinn, boundary_xi)#-1.0
        #boundary_loss_xf = -eps * dfdx(nn_approximator, boundary_xf) + f(nn_approximator, boundary_xf)-1.0

        boundary_loss = \
            (1)*boundary_loss_xi.pow(2).mean() + \
            (1)*boundary_loss_xf.pow(2).mean() 
        
        return boundary_loss

    @classmethod
    def from_params(cls, params: Params, device: torch.device) -> LossSmoothFem:
        base_fun = FemBase(params.n_test_func)
        gramm_matrix = base_fun.calculate_matrix(params.eps, params.n_test_func)
        integration_rule = get_int_rule(params.integration_rule_loss)
        xs = prepare_x_per_base(base_fun, params.n_test_func, params.n_points_x_fem, device)
        return cls(xs, base_fun, gramm_matrix, params.n_test_func, integration_rule, params.divide_by_test)
