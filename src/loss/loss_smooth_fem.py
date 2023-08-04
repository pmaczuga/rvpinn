from __future__ import annotations
from typing import List
import numpy as np

import torch
from src.loss.fem_utils import gauss_weights, prepare_x_for_fem_int

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
         base_fun: FemBase,
         gramm_matrix: torch.Tensor,
         n_test_func: int,
         n_points_x_fem: int,
         divide_by_test: bool
    ):
        self.base_fun = base_fun
        self.gramm_matrix = gramm_matrix
        self.n_test_func = n_test_func
        self.n_points = n_points_x_fem
        self.divider = n_test_func if divide_by_test else 1.0
        # x's for each interval from gaussian quadrature concatenated together 
        self.x = prepare_x_for_fem_int(n_test_func, n_points_x_fem, requires_grad=True).reshape(-1, 1).to(gramm_matrix.device)
        # weights for gaussian quadrature for SINGLE interval
        self.w = gauss_weights(self.n_points).reshape(-1, 1).to(gramm_matrix.device)
        self.boundary_x = torch.tensor([-1., 1.], requires_grad=True).to(gramm_matrix.device)

    # Allows to call object as function
    def __call__(self, pinn: PINN) -> torch.Tensor:
        return self.pde_loss(pinn) + self.boundary_loss(pinn)
    
    def pde_loss(self, pinn: PINN) -> torch.Tensor:
        base_fun = self.base_fun
        n_test_func = self.n_test_func
        n_points = self.n_points
        device = pinn.get_device()
        
        L = torch.zeros(n_test_func)

        val = dfdx(pinn, self.x, order=1)
        rhs = torch.pi**2 * torch.sin(torch.pi * (self.x+1))

        for n in range(1, n_test_func + 1):
            l = base_fun.tip_x(n-1)
            m = base_fun.tip_x(n)
            r = base_fun.tip_x(n+1)

            l_i = (n-1) * n_points
            m_i = (n) * n_points
            r_i = (n+1) * n_points
            x_left  = self.x[l_i:m_i]
            x_right = self.x[m_i:r_i]

            val_left = val[l_i:m_i]
            rhs_left = rhs[l_i:m_i]
            val_right = val[m_i:r_i]
            rhs_right = rhs[m_i:r_i]

            base_left  = base_fun.left(x_left,  n)
            base_right = base_fun.right(x_right, n)
            base_dx_left  = base_fun.dx_left(x_left, n)
            base_dx_right = base_fun.dx_right(x_right, n)

            inte1_left = val_left.mul(base_dx_left)
            inte1_right = val_right.mul(base_dx_right)
            inte2_left = rhs_left.mul(base_left)
            inte2_right = rhs_right.mul(base_right)

            # Gaussian Integration
            val1_left  = (inte1_left * self.w).sum() * (m-l)/2
            val1_right = (inte1_right * self.w).sum() * (r-m)/2
            val2_left  = (inte2_left * self.w).sum() * (m-l)/2
            val2_right = (inte2_right * self.w).sum() * (r-m)/2
            
            interior_loss = (val1_left + val1_right) - (val2_left + val2_right)

            L[n-1] = interior_loss.sum()

        G = self.gramm_matrix
        final_loss = torch.matmul(torch.matmul(L.T, G), L)
        final_loss = final_loss / self.divider

        return final_loss

    def boundary_loss(self, pinn: PINN) -> torch.Tensor:
        boundary_xf = self.boundary_x[-1].reshape(-1, 1) #last point = 1
        boundary_loss_xf = f(pinn, boundary_xf)
        #boundary_loss_xi = -eps * dfdx(nn_approximator, boundary_xi) + f(nn_approximator, boundary_xi)
        
        boundary_xi = self.boundary_x[0].reshape(-1, 1) #first point = 0
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
        return cls(base_fun, gramm_matrix, params.n_test_func, params.n_points_x_fem, params.divide_by_test)
