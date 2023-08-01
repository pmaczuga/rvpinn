from __future__ import annotations

import math

from src.integration import IntegrationRule, get_int_rule
from src.params import Params
from src.analytical import AnalyticalSmooth
from src.pinn import PINN, dfdx
from src.loss.nn_error import NNError
from src.utils import prepare_x

class NNErrorSmooth(NNError):
    def __init__(self, 
                 n_points_error: int, 
                 integration_rule_error: IntegrationRule,
                 ):
        self.n_points_error = n_points_error
        self.integration_rule_error = integration_rule_error

    def error(self, pinn: PINN) -> float:
        int_rule = self.integration_rule_error
        device = pinn.get_device()
        analytical = AnalyticalSmooth()
        x = prepare_x(self.n_points_error, device)
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

    
    @classmethod
    def from_params(cls, params: Params) -> NNErrorSmooth:
        integration_rule_error = get_int_rule(params.integration_rule_error)
        return cls(params.n_points_error, 
                   integration_rule_error,
                   )
