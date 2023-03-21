import torch

def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_tag_path(tag: str) -> str:
    return f"results/{tag}"
