from __future__ import annotations

import math
from typing import List
import numpy as np
import torch

from src.integration import IntegrationRule, get_int_rule
from src.base_fun import BaseFun, PrecomputedBase, SinBase, precompute_base
from src.loss.loss import Loss
from src.pinn import PINN, dfdx, f
from src.params import Params
from src.utils import prepare_x

class LossDelta(Loss):
    def __init__(
         self, 
         x: torch.Tensor,
         eps: float,
         Xd: float,
         precomputed_base: PrecomputedBase,
         integration_rule: IntegrationRule,
         divide_by_test: bool,
    ):
        self.x = x
        self.eps = eps
        self.Xd = Xd
        self.precomputed_base = precomputed_base
        self.integration_rule = integration_rule
        self.divider = precomputed_base.n_test_func if divide_by_test else 1.0


    # Allows to call object as function
    def __call__(self, pinn: PINN) -> torch.Tensor:
        return self.pde_loss(pinn) + self.boundary_loss(pinn)
    
    def pde_loss(self, pinn: PINN) -> torch.Tensor:
        x = self.x
        eps = self.eps
        Xd = self.Xd
        precomputed_base = self.precomputed_base
        int_rule = self.integration_rule
        base_fun = self.precomputed_base.base_fun
        device = pinn.get_device()

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

        return final_loss

    def boundary_loss(self, pinn: PINN) -> torch.Tensor:
        boundary_xi = self.x[0].reshape(-1, 1) #first point = 0
        boundary_loss_xi = f(pinn, boundary_xi)


        boundary_xf = self.x[-1].reshape(-1, 1) #last point = 1
        boundary_loss_xf = f(pinn, boundary_xf)

        boundary_loss = \
            (1)*boundary_loss_xi.pow(2).mean() + \
            (1)*boundary_loss_xf.pow(2).mean() 

        return boundary_loss

    @classmethod
    def from_params(cls, params: Params, device: torch.device) -> LossDelta:
        x = prepare_x(params.n_points_x, device)
        base_fun = BaseFun.from_params(params)
        integration_rule = get_int_rule(params.integration_rule_loss)
        base_x, dx = integration_rule.prepare_x_dx(x)
        precomputed_base = precompute_base(base_fun, base_x, params.eps, params.n_test_func)
        return cls(x, params.eps, params.Xd, precomputed_base, integration_rule, params.divide_by_test)
