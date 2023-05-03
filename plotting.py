import argparse
import math
import os
import matplotlib
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib.pyplot import draw, figure, show
import numpy as np
import torch
from src.analytical import AnalyticalDelta, analytical_from_params

from src.params import Params
from src.io_utils import load_result
from src.pinn import dfdx, f
from src.utils import get_tag_path

matplotlib.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']

vpinn_c = "#0B33B5"
analytical_c = "#D00000"
loss_c = "orange"
norm_c = "#58106a"
error_c = "green"


parser = argparse.ArgumentParser(
                    prog='RVPINN',
                    description='Runs the training of DRVPINN')
parser.add_argument('--tag', type=str)
args = parser.parse_args()

def save_fig(fig: Figure, tag: str, filename: str) -> str:
    directory = f"{get_tag_path(tag)}/figures"
    if not os.path.exists(directory):
        os.mkdir(directory)
    filename = f"{directory}/{filename}"
    fig.savefig(filename, bbox_inches='tight', dpi=200)
    return filename

# Here you can set tag - that is directory inside results where the 
# tag = "tmp"
# Alternatively you can take tag from params.ini in root:
tag = args.tag if args.tag is not None else Params().tag

# Here is trained pinn, loss vector and parameters
pinn, result, params = load_result(tag)
loss_vector = result.loss
error_vector = result.error
norm_vector = result.norm

x_domain = [-1., 1.0]; n_points_x=1000
x_raw = torch.linspace(x_domain[0], x_domain[1], steps=n_points_x)
x_raw.requires_grad_()
x = x_raw.reshape(-1, 1)
x_draw = x.flatten().detach()
y_draw = f(pinn, x).flatten()
#y_ana = (1-torch.exp((x_draw-1)/eps))/(1-np.exp(-1/eps)) + x_draw -1

# y_ana_np1 = 1/params.eps*(1-Xd)*(xnp1+1) #(1-np.exp((xnp-1)/eps))/(1-np.exp(-1/eps)) + xnp - 1#(np.exp(1/eps)-np.exp((xnp)/eps))/(np.exp(1/eps) -1)
# y_ana_np2 = 1/params.eps*(Xd+1)*(1-xnp2)

# x_d = torch.linspace
analytical = analytical_from_params(params)
y_ana_np = analytical(x_draw)

matplotlib.rc('text', usetex=True)

font = {'family' : 'sans-serif', 'size' : 15}
matplotlib.rc('font', **font)

##########################################################################
# Result and exact solution
##########################################################################
fig, ax = plt.subplots()
ax.plot(x_draw.detach().cpu(), y_draw.detach().cpu(),'-', color=vpinn_c, label = 'RVPINNs',linewidth = 2)
ax.plot(x_draw, y_ana_np, '--', color=analytical_c, label="Analytical", linewidth=2)
ax.legend(loc='upper left')
ax.set_xlabel(r" $x$ ", size=19)
ax.set_ylabel(r" $u$ ", size=19)
ax.set_title("NN approximation and exact solution")
save_fig(fig, tag, "result.png")

##########################################################################
# Derivative of result and exact solution
##########################################################################
fig, ax = plt.subplots()
pinn_dx = dfdx(pinn, x, order=1)
exact_dx = analytical.dx(x)
ax.plot(x_draw.detach().cpu(), pinn_dx.detach().cpu(),'-', label = 'RVPINN dx', color=vpinn_c)
ax.plot(x_draw.detach().cpu(), exact_dx.detach().cpu(),'--', linewidth=2, label='Analytical dx', color=analytical_c)
ax.set_title("Derivative of NN approximation and exact solution")
ax.set_xlabel(r" $x$ ", size=19)
ax.set_ylabel(r" $du/dx$ ", size=19)
ax.legend(loc='lower left')#, fontsize='x-large')
save_fig(fig, tag, "result-derivative.png")

##########################################################################
vec = loss_vector
best = math.inf
best_vec = [1.]
pos_vec = [1.]
epochs_vector = np.array(range(1, params.epochs + 1))


for n in range(params.epochs):
  if vec[n]<best and vec[n]>0:
    best_vec.append(vec[n])
    pos_vec.append(n+1)
    best = 1*vec[n]

pos_vec = np.array(pos_vec, dtype=int) - 1
##########################################################################

##########################################################################
# Loss and norm
##########################################################################
fig, ax = plt.subplots()
ax.loglog(pos_vec, loss_vector[pos_vec],'-',linewidth = 2, label="Loss", color=loss_c)
ax.loglog(pos_vec, norm_vector[pos_vec], '--', linewidth=2, label="Norm", color=norm_c)
ax.legend()
ax.set_title("Loss and norm")
ax.set_xlabel(r" Iterations ")
ax.set_ylabel(r" Value ")
save_fig(fig, tag, "loss-and-norm.png")

##########################################################################
# Loss without filter
##########################################################################
fig, ax = plt.subplots()
ax.loglog(epochs_vector, loss_vector,'-',linewidth = 1, label="Loss", color=loss_c)
# ax.legend()
ax.set_title("Loss no filer")
ax.set_xlabel(r" Iterations ")
ax.set_ylabel(r" Loss ")
save_fig(fig, tag, "loss-no-filter.png")

##########################################################################
# Error
##########################################################################
fig, ax = plt.subplots()
ax.loglog(pos_vec, error_vector[pos_vec], '-', linewidth=2, label="Error", color=error_c)
ax.set_xlabel(r" Iterations ")
ax.set_ylabel(r" Error ")
# ax.set_title("Error")
save_fig(fig, tag, "error.png")

##########################################################################
# Error no fiter
##########################################################################
fig, ax = plt.subplots()
ax.loglog(epochs_vector, error_vector, '-', linewidth=1, label="Error", color=error_c)
ax.set_xlabel(r" Iterations ")
ax.set_ylabel(r" Error ")
ax.set_title("Error no filter")
save_fig(fig, tag, "error-no-filter.png")

##########################################################################
# Error to sqrt(loss)
##########################################################################
fig, ax = plt.subplots()
ax.loglog(np.sqrt(loss_vector[pos_vec]), error_vector[pos_vec], color=error_c, label="Error")
ax.loglog(np.sqrt(loss_vector[pos_vec]), np.sqrt(loss_vector[pos_vec]), color=loss_c, label="$y=x$")
ax.set_xlabel(r" $\sqrt{Loss}$ ")
ax.set_ylabel(r" Error ")
ax.legend(loc='upper left')#, fontsize='x-large')
# ax.set_title(r"Error to $\sqrt{Loss}$")
filename = f"{get_tag_path(tag)}/error.pt"
save_fig(fig, tag, "error-to-sqrt-loss.png")

##########################################################################
# Exact solution on right end
##########################################################################
# fig, ax = plt.subplots()
# x_raw = torch.linspace(0.98, x_domain[1], steps=n_points_x)
# x = x_raw.reshape(-1, 1)
# x_draw = x.flatten().detach()
# y_ana_np = analytical(x_draw)
# ax.plot(x_draw, y_ana_np, 'r--', label="Analytical", linewidth=2)

show()
