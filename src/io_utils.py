import os
import shutil
from typing import Tuple
import torch
import numpy as np

from src.params import Params
from src.pinn import PINN
from src.utils import TrainResult, get_tag_path

def create_tag_dir(tag: str):
    """Create or  empty directory 'results/{tag}'"""
    if not os.path.exists("results"):
        os.mkdir("results")
    target_path = get_tag_path(tag)
    if os.path.exists(target_path):
        shutil.rmtree(target_path)
    os.mkdir(target_path)

def save_pinn_by_tag(pinn: PINN, tag: str):
    """Save PINNs state dict to directory 'results/{tag}"""
    filename = f"{get_tag_path(tag)}/pinn.pth"
    torch.save(pinn.state_dict(), filename)

def save_loss_by_tag(loss: torch.Tensor, tag: str):
    """Save loss vector to to directory 'results/{tag}"""
    filename = f"{get_tag_path(tag)}/loss.pt"
    torch.save(loss, filename)

def save_error_by_tag(error: torch.Tensor, tag: str):
    """Save error vector to to directory 'results/{tag}"""
    filename = f"{get_tag_path(tag)}/error.pt"
    torch.save(error, filename)

def save_norm_by_tag(norm: torch.Tensor, tag: str):
    """Save norm vector to to directory 'results/{tag}"""
    filename = f"{get_tag_path(tag)}/norm.pt"
    torch.save(norm, filename)

def save_result(pinn: PINN, result, params: Params):
    """Save result and params to directory 'results/{tag}'
    
    `tag` is extracted from params"""
    create_tag_dir(params.tag)
    params.save_by_tag()
    save_pinn_by_tag(pinn, params.tag)
    save_loss_by_tag(result.loss, params.tag)
    if result.norm is not None:
        save_norm_by_tag(result.loss, params.tag)
    if result.error is not None:
        save_error_by_tag(result.error, params.tag)

def load_pinn(params: Params) -> PINN:
    """Load PINN instance from directory 'results/{tag}'
    
    `tag` is extracted from params"""
    filename = f"{get_tag_path(params.tag)}/pinn.pth"
    pinn = PINN(params.layers, params.neurons_per_layer)
    pinn.load_state_dict(torch.load(filename))
    return pinn

def load_loss(params: Params) -> torch.Tensor:
    """Load loss vector instance from directory 'results/{tag}'
    
    `tag` is extracted from params"""
    filename = f"{get_tag_path(params.tag)}/loss.pt"
    return torch.load(filename)

def load_error(params: Params) -> torch.Tensor | None:
    """Load error vector instance from directory 'results/{tag}'
    
    `tag` is extracted from params
    If file does not exist return None"""
    filename = f"{get_tag_path(params.tag)}/error.pt"
    if os.path.exists(filename):
        return torch.load(filename)
    return None

def load_norm(params: Params) -> torch.Tensor | None:
    """Load norm vector instance from directory 'results/{tag}'
    
    `tag` is extracted from params
    If file does not exist return None"""
    filename = f"{get_tag_path(params.tag)}/norm.pt"
    if os.path.exists(filename):
        return torch.load(filename)
    return None

def load_result(tag: str) -> Tuple[PINN, TrainResult, Params]:
    """Load PINN, result and params from directory results/{tag}"""
    params = Params.load_by_tag(tag)
    loss = load_loss(params)
    error = load_error(params)
    norm = load_norm(params)
    result = TrainResult(loss, error, norm)
    return load_pinn(params), result, params
