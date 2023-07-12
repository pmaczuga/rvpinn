from __future__ import annotations

import math
import torch

from src.integration import IntegrationRule, get_int_rule
from src.base_fun import BaseFun, PrecomputedBase, SinBase, precompute_base
from src.loss.loss import Loss
from src.pinn import PINN, dfdx, f
from src.params import Params

class LossSmooth(Loss):
    def __init__(
         self, 
         x: torch.Tensor,
         precomputed_base: PrecomputedBase,
         integration_rule: IntegrationRule,
         divide_by_test: bool
    ):
        self.x = x
        self.precomputed_base = precomputed_base
        self.integration_rule = integration_rule
        self.divider = precomputed_base.n_test_func if divide_by_test else 1.0


    # Allows to call object as function
    def __call__(self, pinn: PINN) -> torch.Tensor:
        x = self.x
        precomputed_base = self.precomputed_base
        int_rule = self.integration_rule
        base_fun = precomputed_base.base_fun
        device = pinn.get_device()
    
        x, dx = int_rule.prepare_x_dx(x)
        val = dfdx(pinn, x, order=1) #this can be precomputed to save time
        rhs = torch.pi**2 * torch.sin(torch.pi * (x+1))
        
        L = torch.zeros(precomputed_base.n_test_func)

        for n in range(1, precomputed_base.n_test_func + 1): 
            interior_loss_test1 = precomputed_base.get_dx(n)
            interior_loss_test2 = precomputed_base.get(n)
            inte1 = val.mul(interior_loss_test1)
            inte2 = rhs.mul(interior_loss_test2)

            val1 = int_rule.int_using_x_dx(inte1, x, dx)
            val2 = int_rule.int_using_x_dx(inte2, x, dx)
            
            interior_loss = val1 - val2
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
    def from_params(cls, x: torch.Tensor, params: Params) -> LossSmooth:
        base_fun = BaseFun.from_params(params)
        integration_rule = get_int_rule(params.integration_rule_loss)
        base_x, dx = integration_rule.prepare_x_dx(x)
        precomputed_base = precompute_base(base_fun, base_x, params.eps, params.n_test_func)
        return cls(x, precomputed_base, integration_rule, params.divide_by_test)
