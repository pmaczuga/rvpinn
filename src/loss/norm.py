import torch
from src.loss.loss_delta_fem import LossDeltaFem
from src.loss.loss_ad_fem import LossADFem
from src.loss.loss_smooth_fem import LossSmoothFem
from src.loss.loss_smooth import LossSmooth
from src.loss.loss_delta import LossDelta
from src.params import Params
from src.integration import get_int_rule
from src.base_fun import BaseFun, FemBase, precompute_base, prepare_x_per_base
from src.loss.loss_ad import LossAD
from src.utils import prepare_x

# Norm is calculated the same way as loss, but we can't use the same instance as for loss
# because we sometimes want to have different:
# - number of integration points
# - integration rule
# So the only thing we need to take care of is to create proper instance of loss
# based on Params
# And so, the code below is almost identical to the one from the one in respective Loss,
# I only get different value from Params

class NormAD(LossAD):
    @classmethod
    def from_params(cls, params: Params, device: torch.device) -> LossAD:
        x = prepare_x(params.n_points_error, device)
        base_fun = BaseFun.from_params(params)
        integration_rule = get_int_rule(params.integration_rule_norm)
        base_x, dx = integration_rule.prepare_x_dx(x)
        precomputed_base = precompute_base(base_fun, base_x, params.eps, params.n_test_func)
        return cls(x, params.eps, precomputed_base, integration_rule, params.divide_by_test)
    
class NormDelta(LossDelta):
    @classmethod
    def from_params(cls, params: Params, device: torch.device) -> LossDelta:
        x = prepare_x(params.n_points_error, device)
        base_fun = BaseFun.from_params(params)
        integration_rule = get_int_rule(params.integration_rule_norm)
        base_x, dx = integration_rule.prepare_x_dx(x)
        precomputed_base = precompute_base(base_fun, base_x, params.eps, params.n_test_func)
        return cls(x, params.eps, params.Xd, precomputed_base, integration_rule, params.divide_by_test)
    
class NormSmooth(LossSmooth):
    @classmethod
    def from_params(cls, params: Params, device: torch.device) -> LossSmooth:
        x = prepare_x(params.n_points_error, device)
        base_fun = BaseFun.from_params(params)
        integration_rule = get_int_rule(params.integration_rule_norm)
        base_x, dx = integration_rule.prepare_x_dx(x)
        precomputed_base = precompute_base(base_fun, base_x, 1.0, params.n_test_func)
        return cls(x, precomputed_base, integration_rule, params.divide_by_test)
    
class NormSmoothFem(LossSmoothFem):
    @classmethod
    def from_params(cls, params: Params, device: torch.device) -> LossSmoothFem:
        base_fun = FemBase(params.n_test_func)
        gramm_matrix = base_fun.calculate_matrix(params.eps, params.n_test_func)
        integration_rule = get_int_rule(params.integration_rule_norm)
        xs = prepare_x_per_base(base_fun, params.n_test_func, params.n_points_error_fem, device)
        return cls(xs, base_fun, gramm_matrix, params.n_test_func, integration_rule, params.divide_by_test)

class NormADFem(LossADFem):
    @classmethod
    def from_params(cls, params: Params, device: torch.device) -> LossADFem:
        base_fun = FemBase(params.n_test_func)
        gramm_matrix = base_fun.calculate_matrix(params.eps, params.n_test_func)
        integration_rule = get_int_rule(params.integration_rule_norm)
        xs = prepare_x_per_base(base_fun, params.n_test_func, params.n_points_error_fem, device)
        return cls(xs, params.eps, base_fun, gramm_matrix, params.n_test_func, integration_rule, params.divide_by_test)

class NormDeltaFem(LossDeltaFem):
    @classmethod
    def from_params(cls, params: Params, device: torch.device) -> LossDeltaFem:
        base_fun = FemBase(params.n_test_func)
        gramm_matrix = base_fun.calculate_matrix(params.eps, params.n_test_func)
        integration_rule = get_int_rule(params.integration_rule_norm)
        xs = prepare_x_per_base(base_fun, params.n_test_func, params.n_points_error_fem, device)
        return cls(xs, params.eps, params.Xd, base_fun, gramm_matrix, params.n_test_func, integration_rule, params.divide_by_test)
