from __future__ import annotations
from typing import List
import numpy as np

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
         base_fun: FemBase,
         gramm_matrix: torch.Tensor,
         n_test_func: int,
         n_points_x_fem: int,
         divide_by_test: bool
    ):
        self.base_fun = base_fun
        self.gramm_matrix = gramm_matrix
        self.n_test_func = n_test_func
        self.divider = n_test_func if divide_by_test else 1.0
        x, w = np.polynomial.legendre.leggauss(n_points_x_fem)
        self.x_norm = torch.from_numpy(x).float().reshape(-1, 1).to(gramm_matrix.device)
        self.x_norm.requires_grad = True
        self.w = torch.from_numpy(w).float().reshape(-1,1).to(gramm_matrix.device)
        self.boundary_x = torch.tensor([-1., 1.], requires_grad=True).to(gramm_matrix.device)

        

    # Allows to call object as function
    def __call__(self, pinn: PINN) -> torch.Tensor:
        return self.pde_loss(pinn) + self.boundary_loss(pinn)
    
    def pde_loss(self, pinn: PINN) -> torch.Tensor:
        base_fun = self.base_fun
        n_test_func = self.n_test_func
        device = pinn.get_device()
        
        L = torch.zeros(n_test_func)

        for n in range(1, n_test_func + 1):
            l = base_fun.tip_x(n-1)
            m = base_fun.tip_x(n)
            r = base_fun.tip_x(n+1)

            x_left  = self.x_norm * (m-l)/2 + (m+l)/2
            x_right = self.x_norm * (r-m)/2 + (r+m)/2

            val_left = dfdx(pinn, x_left, order=1) 
            rhs_left = torch.pi**2 * torch.sin(torch.pi * (x_left+1))
            val_right = dfdx(pinn, x_right, order=1) 
            rhs_right = torch.pi**2 * torch.sin(torch.pi * (x_right+1))

            base_left  = base_fun(x_left,  n)
            base_right = base_fun(x_right, n)
            base_dx_left  = 0.0 * x_left  + 1.0 / base_fun.delta_x()
            base_dx_right = 0.0 * x_right - 1.0 / base_fun.delta_x()

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
