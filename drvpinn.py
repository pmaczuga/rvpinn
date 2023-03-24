import argparse
import torch
from src.run import run
from src.loss.factory import loss_from_params
from src.io_utils import save_result
from src.pinn import PINN
from src.train import train_model
from src.nn_error import nn_error_from_params

from src.utils import *
from src.params import Params

parser = argparse.ArgumentParser(
                    prog='DRVPINN',
                    description='Runs the training of DRVPINN')
parser.add_argument('--tag', type=str)
parser.add_argument('--equation', '-e', type=str)


def main():
    args = parser.parse_args()
    device = get_device()
    kwargs = {key: value for key, value in vars(args).items() if value is not None}
    params = Params(**kwargs)

    run(params, device)

if __name__ == '__main__':
    main()
