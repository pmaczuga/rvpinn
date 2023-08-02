import torch
from src.loss.loss_delta_fem import LossDeltaFem
from src.loss.loss_ad_fem import LossADFem
from src.loss.loss_smooth_fem import LossSmoothFem
from src.loss.nn_error.nn_error import NNError
from src.loss.nn_error.nn_error_ad import NNErrorAD
from src.loss.nn_error.nn_error_delta import NNErrorDelta
from src.loss.nn_error.nn_error_smooth import NNErrorSmooth
from src.loss.norm import NormAD, NormADFem, NormDelta, NormDeltaFem, NormSmooth, NormSmoothFem
from src.loss.loss_smooth import LossSmooth
from src.loss.loss import Loss
from src.loss.loss_ad import LossAD
from src.loss.loss_delta import LossDelta
from src.params import Params


def loss_from_params(params: Params, device: torch.device) -> Loss:
    if params.equation == "ad":
        if params.test_func == "fem":
            return LossADFem.from_params(params, device)
        return LossAD.from_params(params, device)
    if params.equation == "delta":
        if params.test_func == "fem":
            return LossDeltaFem.from_params(params, device)
        return LossDelta.from_params(params, device)
    if params.equation == "smooth":
        if params.test_func == "fem":
            return LossSmoothFem.from_params(params, device)
        return LossSmooth.from_params(params, device)
    raise ValueError(f"Wrong equation in Params: {params.equation}")

def norm_from_params(params: Params, device: torch.device) -> Loss:
    if params.equation == "ad":
        if params.test_func == "fem":
            return NormADFem.from_params(params, device)
        return NormAD.from_params(params, device)
    if params.equation == "delta":
        if params.test_func == "fem":
            return NormDeltaFem.from_params(params, device)
        return NormDelta.from_params(params, device)
    if params.equation == "smooth":
        if params.test_func == "fem":
            return NormSmoothFem.from_params(params, device)
        return NormSmooth.from_params(params, device)
    raise ValueError(f"Wrong equation in Params: {params.equation}")

def nn_error_from_params(params: Params) -> NNError:
    if params.equation == "ad":
        return NNErrorAD.from_params(params)
    if params.equation == "delta":
        return NNErrorDelta.from_params(params)
    if params.equation == "smooth":
        return NNErrorSmooth.from_params(params)
    raise ValueError(f"Wrong equation in params: {params.equation}") 
