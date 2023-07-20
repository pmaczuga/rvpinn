# RVPINN

## Running

For running there are only 3 important files:
- `params.ini` - holds all the parameters, like which equation to run, eps, Xd, and so on...
- `rvpinn.py` - runs the training and saves results
- `plotting.py` - plots the results acquired by running `rvpinn.py`

All those files run just like that, taking all the arguments from `params.ini`

```
$ python3 rvpinn.py
$ python3 plotting.py
```

There is one interesting parameter: `tag`. In practice it is the name of the folder where results
will be saved (and from where results will be read when plotting). The results themselves 
are saved to: `results/{tag}/`. The saved files are:
- `params.ini` - copy of the parameters file used when running the training
- `pinn.pth` - state of the PINN
- `loss.pt` - loss vector, containing loss values for all the epochs

And additionally, if `compute_error = True`:
- `error.pt` - error vector
- `norm.pt` - norm vector

### Command line args

File `rvpinn.py` also takes arguments that allow to override all the parameters. It is sometimes more convenient and mostly useful
for setting `tag` and `equation`.

```
$ python3 rvpinn --tag ad_result --equation ad
```

`--equation` can be shortened to `-e`:
```
# python3 rvpinn --tag ad_result -e ad
```

File `plotting.py` only takes one argument: `--tag`. 
All the other parameters are loaded from params file copied after training.
```
$ python3 plotting.py --tag ad_result
```

## Arguments

You can also do this:
```
$ python3 rvpinn.py --help

usage: RVPINN [-h] [--tag TAG] [--epochs EPOCHS] [--layers LAYERS] [--neurons-per-layer NEURONS_PER_LAYER]
               [--learning-rate LEARNING_RATE] [--use-best-pinn | --no-use-best-pinn] [--equation EQUATION] [--test-func TEST_FUNC]
               [--eps EPS] [--Xd XD] [--compute-error | --no-compute-error] [--n-points-x N_POINTS_X]
               [--n-points-error N_POINTS_ERROR] [--n-test-func N_TEST_FUNC] [--integration-rule-loss INTEGRATION_RULE_LOSS]
               [--integration-rule-norm INTEGRATION_RULE_NORM] [--integration-rule-error INTEGRATION_RULE_ERROR]
               [--divide-by-test DIVIDE_BY_TEST] [--atol ATOL] [--rtol RTOL] [--params PARAMS]

Runs the training of RVPINN

options:
  -h, --help            show this help message and exit
  --tag TAG             Name of the folder for storing training results
  --epochs EPOCHS
  --layers LAYERS
  --neurons-per-layer NEURONS_PER_LAYER
  --learning-rate LEARNING_RATE
  --use-best-pinn, --no-use-best-pinn
  --equation EQUATION, -e EQUATION
                        Equation to use - either "delta" (Dirac delta on RHS) or "ad" (Advection-Diffusion)
  --test-func TEST_FUNC
                        Type of test functions - one of "sin", "poly", "fem" or "mixed". "mixed" takes half of sin and half of fem.
                        Number of fem can be changed like this: "mixed10" - 10 fem, rest is sin.
  --eps EPS
  --Xd XD
  --compute-error, --no-compute-error
                        Whether to compute error in each iteration will influence performance
  --n-points-x N_POINTS_X
                        Number of integration nodes
  --n-points-error N_POINTS_ERROR
                        Number of integration nodes for computing error. Ideally greater than n_points_x
  --n-test-func N_TEST_FUNC
                        Number of test functions
  --integration-rule-loss INTEGRATION_RULE_LOSS
                        Integration rule for loss function. Either "trapz" or "midpoint.
  --integration-rule-norm INTEGRATION_RULE_NORM
                        Integration rule for calculating norm. One of "trapz", "midpoint" or "simpson".
  --integration-rule-error INTEGRATION_RULE_ERROR
                        Integration rule for calculating error. One of "trapz", "midpoint" or "simpson".
  --divide-by-test DIVIDE_BY_TEST
                        Whether to divide loss (and norm) by number of test functions.
  --atol ATOL
  --rtol RTOL
  --params PARAMS       Path to .ini file with parameters. Defaults to "params.ini" in current directory

```

## Interesting files
- `pinn.py` - definition of the neural network
- `train.py` - training of the RVPINN, returns vectors of loss, norm and error to vectors
- `analytical.py` - analytical solutions for both equations
- `loss/loss_ad.py` - loss function for Advection-Diffusion equation
- `loss/loss_delta` - loss function for equation with Dirac delta on RHS
- `nn_error/nn_error_ad.py` - calculation of error and norm for for Advection-Diffusion equation
- `nn_error/nn_error_delta.py` - calculation of error and norm equation with Dirac delta on RHS
- `integration.py` - implemented integration rules