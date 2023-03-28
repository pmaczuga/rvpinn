import torch
from src.loss.factory import loss_from_params
from src.io_utils import save_result
from src.pinn import PINN
from src.train import train_model
from src.nn_error import nn_error_from_params

from src.utils import *
from src.params import Params

def run(params, device):
    print(f"Running on a device: {device}")

    x_raw = torch.linspace(-1.0, 1.0, steps=params.n_points_x)
    x_raw.requires_grad_()
    x = x_raw.reshape(-1, 1).to(device)

    pinn = PINN(params.layers, params.neurons_per_layer).to(device)

    # train the PINN
    loss_fn = loss_from_params(x, params)
    result = train_model(
        pinn, 
        loss_fn=loss_fn, 
        learning_rate=params.learning_rate, 
        max_epochs=params.epochs, 
        atol = params.atol, 
        rtol = params.rtol, 
        best = params.use_best_pinn,
        nn_error = nn_error_from_params(params) if params.compute_error else None
    )

    loss_vector = result.loss
    error_vector = result.error
    norm_vector = result.norm

    print(f"Loss at the end: {loss_vector[-1]}")
    if error_vector is not None:
        print(f"Error at the end: {error_vector[-1]}")
    if norm_vector is not None:
        print(f"Norm at the end: {norm_vector[-1]}")

    save_result(pinn, result, params)
