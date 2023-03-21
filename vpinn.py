import torch
from functools import partial
from src.io_utils import save_result
from src.loss import compute_loss
from src.pinn import PINN
from src.train import train_model

from src.utils import *
from src.params import Params

def main():
    device = get_device()
    params = Params()

    print(f"Running on a device: {device}")

    x_domain = [-1., 1.0]; n_points_x=params.n_points_x 
    x_raw = torch.linspace(x_domain[0], x_domain[1], steps=n_points_x)
    x_raw.requires_grad_()

    x = x_raw.reshape(-1, 1).to(device)

    pinn = PINN(params.layers, params.neurons_per_layer).to(device) # this is hyperbolic tangent
    # pinn = PINN(5, 25, act=nn.ReLU()).to(device)
    # pinn = PINN(2, 5, act=nn.LeakyReLU()).to(device) # this is LeakyReLU

    # train the PINN
    loss_fn = partial(compute_loss, x=x, n_test_func=params.n_test_func, device=device)
    pinn, loss_vector = train_model(
        pinn, 
        loss_fn=loss_fn, 
        learning_rate=params.learning_rate, 
        max_epochs=params.epochs, 
        atol = params.atol, 
        rtol = params.rtol, 
        device=device, 
        best = params.use_best_pinn
    )

    save_result(pinn, loss_vector, params)

if __name__ == '__main__':
    main()
