from src.nn_error import nn_error_from_params
from src.nn_error import NNErrorDelta
from src.io_utils import load_result
from src.params import Params

tag_ad = "ad"
tag_delta = "delta"
pinn_ad, result_ad, params_ad = load_result(tag_ad)
pinn_delta, result_delta, params_delta = load_result(tag_delta)

nn_error_ad = nn_error_from_params(params_ad)
nn_error_delta = nn_error_from_params(params_delta)

print(f"Loss of ad: {result_ad.loss[-1]}")
print(f"Norm of ad: {nn_error_ad.norm(pinn_ad)}")
print(f"Error of ad: {nn_error_ad.error(pinn_ad)}")

print(f"Loss of delta: {result_delta.loss[-1]}")
print(f"Norm of delta: {nn_error_delta.norm(pinn_delta)}")
print(f"Error of delta: {nn_error_delta.error(pinn_delta)}")
