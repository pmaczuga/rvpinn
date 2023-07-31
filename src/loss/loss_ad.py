from __future__ import annotations

import math
import numpy as np
import torch

from src.integration import IntegrationRule, get_int_rule
from src.base_fun import BaseFun, PrecomputedBase, SinBase, precompute_base
from src.loss.loss import Loss
from src.pinn import PINN, dfdx, f
from src.params import Params

class LossAD(Loss):
    def __init__(
         self, 
         x: torch.Tensor,
         eps: float,
         precomputed_base: PrecomputedBase,
         integration_rule: IntegrationRule,
         divide_by_test: bool
    ):
        self.x = x
        self.eps = eps
        self.precomputed_base = precomputed_base
        self.integration_rule = integration_rule
        self.divider = precomputed_base.n_test_func if divide_by_test else 1.0


    # Allows to call object as function
    def __call__(self, pinn: PINN) -> torch.Tensor:
        x = self.x
        eps = self.eps
        precomputed_base = self.precomputed_base
        int_rule = self.integration_rule
        base_fun = precomputed_base.base_fun
        device = pinn.get_device()

        beta = 1
        
        L = torch.zeros(precomputed_base.n_test_func)

        for n in range(1, precomputed_base.n_test_func + 1): 

            n_points = int(np.ceil(torch.numel(x) / precomputed_base.n_test_func))

            left  = base_fun._tip_x(n-1)
            mid   = base_fun._tip_x(n)
            right = base_fun._tip_x(n+1)

            x_left  = torch.linspace(left, mid,  n_points, requires_grad=True).reshape(-1, 1)
            x_right = torch.linspace(mid, right, n_points, requires_grad=True).reshape(-1, 1)

            val_left = dfdx(pinn, x_left, order=1) 
            val_right = dfdx(pinn, x_right, order=1) 

            trial1_left  = eps * val_left
            trial1_right = eps * val_right
            trial2_left  = beta * val_left
            trial2_right = beta * val_right

            base_left  = base_fun(x_left,  n)
            base_right = base_fun(x_right, n)
            base_dx_left  = torch.full((n_points,),  1.0 / base_fun._delta_x(), requires_grad=True).reshape(-1, 1)
            base_dx_right = torch.full((n_points,), -1.0 / base_fun._delta_x(), requires_grad=True).reshape(-1, 1)

            inte1_left  = trial1_left.mul(base_dx_left)
            inte1_right = trial1_right.mul(base_dx_right)
            inte2_left  = trial2_left.mul(base_left)
            inte2_right  = trial2_right.mul(base_right)

            x_left, dx_left   = int_rule.prepare_x_dx(x_left)
            x_right, dx_right = int_rule.prepare_x_dx(x_right)

            val1_left = int_rule.int_using_x_dx(inte1_left, x_left, dx_left)
            val1_right = int_rule.int_using_x_dx(inte1_right, x_right, dx_right)
            val2_left = int_rule.int_using_x_dx(inte2_left, x_left, dx_left)
            val2_right = int_rule.int_using_x_dx(inte2_right, x_right, dx_right)
            val3_left = int_rule.int_using_x_dx(base_left, x_left, dx_left)
            val3_right = int_rule.int_using_x_dx(base_right, x_right, dx_right)
            
            interior_loss = (val1_left + val1_right) + (val2_left + val2_right) - (val3_left + val3_right)
            L[n-1] = interior_loss.sum()

        # print(L)
        G = precomputed_base.get_matrix()
        final_loss = torch.matmul(torch.matmul(L.T, G), L)
        final_loss = final_loss / self.divider

        boundary_xf = x[-1].reshape(-1, 1) #last point = 1
        boundary_loss_xf = f(pinn, boundary_xf)
        #boundary_loss_xi = -eps * dfdx(nn_approximator, boundary_xi) + f(nn_approximator, boundary_xi)
        
        boundary_xi = x[0].reshape(-1, 1) #first point = 0
        boundary_loss_xi = f(pinn, boundary_xi)#-1.0
        #boundary_loss_xf = -eps * dfdx(nn_approximator, boundary_xf) + f(nn_approximator, boundary_xf)-1.0
        final_loss+= \
            (1)*boundary_loss_xi.pow(2).mean() + \
            (1)*boundary_loss_xf.pow(2).mean() 
        return final_loss
    
    @classmethod
    def from_params(cls, x: torch.Tensor, params: Params) -> LossAD:
        base_fun = BaseFun.from_params(params)
        integration_rule = get_int_rule(params.integration_rule_loss)
        base_x, dx = integration_rule.prepare_x_dx(x)
        precomputed_base = precompute_base(base_fun, base_x, params.eps, params.n_test_func)
        return cls(x, params.eps, precomputed_base, integration_rule, params.divide_by_test)
