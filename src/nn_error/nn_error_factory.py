from src.params import Params
from src.nn_error import NNError, NNErrorAD, NNErrorDelta

def nn_error_from_params(params: Params) -> NNError:
    if params.equation == "ad":
        return NNErrorAD(params.eps, params.n_points_error, params.n_test_func)
    if params.equation == "delta":
        return NNErrorDelta(params.eps, params.Xd, params.n_points_error, params.n_test_func)
    raise ValueError(f"Wrong equation in params: {params.equation}")