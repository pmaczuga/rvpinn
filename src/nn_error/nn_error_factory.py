from src.nn_error.nn_error_smooth import NNErrorSmooth
from src.params import Params
from src.nn_error import NNError, NNErrorAD, NNErrorDelta

def nn_error_from_params(params: Params) -> NNError:
    if params.equation == "ad":
        return NNErrorAD.from_params(params)
    if params.equation == "delta":
        return NNErrorDelta.from_params(params)
    if params.equation == "smooth":
        return NNErrorSmooth.from_params(params)
    raise ValueError(f"Wrong equation in params: {params.equation}") 
