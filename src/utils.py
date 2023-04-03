import numpy as np
import torch

def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_tag_path(tag: str) -> str:
    return f"results/{tag}"

def right_centered_distribution(start: float, 
                                end: float, 
                                steps: int, 
                                p: float=1.5,
                                dtype=torch.float32) -> torch.Tensor:
    # Function that grows from 0 to 1
    # The bigger the p, the slower it grows towards the end. 
    # When p = 1, then it becomes f(x) = x
    distr = lambda x, p: -(-x+1)**p + 1

    x = torch.linspace(0.0, 1.0, steps, dtype=dtype)
    x = distr(x, p)
    x = x * (end - start) + start
    return x

class TrainResult:
   def __init__(self, 
                loss: torch.Tensor, 
                error: torch.Tensor | None = None,
                norm: torch.Tensor | None = None):
      self.loss = loss
      self.error = error
      self.norm = norm