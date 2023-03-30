import argparse
from src.run import run
from src.pinn import PINN

from src.utils import *
from src.params import Params

parser = argparse.ArgumentParser(
                    prog='DRVPINN',
                    description='Runs the training of DRVPINN')

parser.add_argument('--tag', type=str, 
                    help='Name of the folder for storing training results')

parser.add_argument('--epochs', type=int)
parser.add_argument('--layers', type=int)
parser.add_argument('--neurons_per_layer', type=int)
parser.add_argument('--learning_rate', type=float)
parser.add_argument('--use_best_pinn', type=bool)

parser.add_argument('--equation', '-e', type=str, 
                    help=('Equation to use - '
                          'either "delta" (Dirac delta on RHS) '
                          'or "ad" (Advection-Diffusion)'))
parser.add_argument('--eps', type=float)
parser.add_argument('--Xd', type=float)

parser.add_argument("--compute_error", type=bool,
                    help='Whether to compute error in each iteration will influence performance')
parser.add_argument("--n_points_x", type=int,
                    help='Number of integration nodes')
parser.add_argument("--n_points_error", type=int,
                    help=('Number of integration nodes for computing error. '
                         'Ideally greater than n_points_x'))
parser.add_argument("--n_test_func", type=int, 
                    help='Number of test functions')

parser.add_argument("--atol", type=float)
parser.add_argument("--rtol", type=float)

parser.add_argument("--params", type=str, default="params.ini", 
                    help=('Path to .ini file with parameters. '
                          'Defaults to "params.ini" in current directory'))

def get_params(parser: argparse.ArgumentParser) -> Params:
    args = parser.parse_args()
    kwargs = {key: value for key, value in vars(args).items() if value is not None}
    params = Params(**kwargs)
    return params

def main():
    device = get_device()
    params = get_params(parser)
    run(params, device)

if __name__ == '__main__':
    main()
