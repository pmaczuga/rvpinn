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
parser.add_argument('--neurons-per-layer', type=int)
parser.add_argument('--learning-rate', type=float)
parser.add_argument('--use-best-pinn', type=bool, action=argparse.BooleanOptionalAction)

parser.add_argument('--equation', '-e', type=str, 
                    help=('Equation to use - '
                          'either "delta" (Dirac delta on RHS) '
                          'or "ad" (Advection-Diffusion)'))
parser.add_argument('--test-func', type=str, 
                    help='Type of test functions - either "sin" or "poly"')
parser.add_argument('--eps', type=float)
parser.add_argument('--Xd', type=float)

parser.add_argument("--compute-error", type=bool, action=argparse.BooleanOptionalAction,
                    help='Whether to compute error in each iteration will influence performance')
parser.add_argument("--n-points-x", type=int,
                    help='Number of integration nodes')
parser.add_argument("--n-points-error", type=int,
                    help=('Number of integration nodes for computing error. '
                         'Ideally greater than n_points_x'))
parser.add_argument("--n-test-func", type=int, 
                    help='Number of test functions')

parser.add_argument("--atol", type=float)
parser.add_argument("--rtol", type=float)

parser.add_argument("--params", type=str, default="params.ini", 
                    help=('Path to .ini file with parameters. '
                          'Defaults to "params.ini" in current directory'))


def get_params(args: argparse.Namespace) -> Params:
    kwargs = {key: value for key, value in vars(args).items() if value is not None}
    params = Params(filename=args.params, **kwargs)
    return params

def main():
    args = parser.parse_args()
    params = get_params(args)
    device = get_device()
    run(params, device)

if __name__ == '__main__':
    main()
