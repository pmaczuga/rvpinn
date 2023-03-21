import os
import shutil
from typing import Tuple
import torch

from src.params import Params
from src.pinn import PINN
from src.utils import get_tag_path

def create_tag_dir(tag: str):
    if not os.path.exists("results"):
        os.mkdir("results")
    target_path = get_tag_path(tag)
    if os.path.exists(target_path):
        shutil.rmtree(target_path)
    os.mkdir(target_path)

def save_pinn_by_tag(pinn: PINN, tag: str):
    filename = f"{get_tag_path(tag)}/pinn.pth"
    torch.save(pinn.state_dict(), filename)

def save_loss_by_tag(loss: torch.Tensor, tag: str):
    filename = f"{get_tag_path(tag)}/loss.pt"
    torch.save(loss, filename)

def save_result(pinn: PINN, loss: torch.Tensor, params: Params):
    create_tag_dir(params.tag)
    params.save_by_tag()
    save_pinn_by_tag(pinn, params.tag)
    save_loss_by_tag(loss, params.tag)

def load_pinn(params: Params) -> PINN:
    filename = f"{get_tag_path(params.tag)}/pinn.pth"
    pinn = PINN(params.layers, params.neurons_per_layer)
    pinn.load_state_dict(torch.load(filename))
    return pinn

def load_loss(params: Params):
    filename = f"{get_tag_path(params.tag)}/loss.pt"
    return torch.load(filename)

def load_result(tag: str) -> Tuple[PINN, torch.Tensor, Params]:
    params = Params.load_by_tag(tag)
    return load_pinn(params), load_loss(params), params
