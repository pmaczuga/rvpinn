import torch
from src.loss.loss import Loss
from src.loss.loss_ad import LossAD
from src.loss.loss_delta import LossDelta
from src.params import Params


def loss_from_params(x: torch.Tensor, params: Params) -> Loss:
    if params.equation == "ad":
        return LossAD.from_params(x, params)
    if params.equation == "delta":
        return LossDelta.from_params(x, params)
    raise ValueError(f"Wrong equation in Params: {params.equation}")
