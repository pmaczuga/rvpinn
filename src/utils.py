import numpy as np
import torch

def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_tag_path(tag: str) -> str:
    return f"results/{tag}"

class TrainResult:
   def __init__(self, 
                loss: torch.Tensor, 
                error: torch.Tensor | None = None,
                norm: torch.Tensor | None = None):
      self.loss = loss
      self.error = error
      self.norm = norm