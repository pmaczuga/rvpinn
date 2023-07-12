import argparse
import math
import os
import matplotlib
import mpltools.annotation as mpl
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib.pyplot import draw, figure, show
import numpy as np
import torch
from src.analytical import AnalyticalAD, AnalyticalDelta, analytical_from_params
from src.integration import MidpointInt, TrapzInt 

from src.params import Params
from src.io_utils import load_result
from src.pinn import dfdx, f
from src.utils import get_tag_path

matplotlib.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']

vpinn_c = "#0B33B5"
analytical_c = "#D00000"
loss_c = "darkorange"
norm_c = "#58106a"
error_c = "darkgreen"


parser = argparse.ArgumentParser(
                    prog='SVPINN',
                    description='Runs the training of SVPINN')
parser.add_argument('--tag', type=str)
args = parser.parse_args()

def save_fig(fig: Figure, tag: str, filename: str) -> str:
    directory = f"{get_tag_path(tag)}/figures"
    if not os.path.exists(directory):
        os.mkdir(directory)
    filename = f"{directory}/{filename}"
    fig.savefig(filename, bbox_inches='tight', dpi=200)
    return filename

def calculate_ana_norm(x, params: Params) -> float:
    analytical = AnalyticalAD(params.eps)
    ana = analytical.dx(x)
    value = ana**2
    int_rule = TrapzInt()
    x, dx = int_rule.prepare_x_dx(x)
    result = int_rule.int_using_x_dx(value, x, dx)
    return np.sqrt(result.item() * params.eps)

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

analytical_norm = calculate_ana_norm(x_raw, params)

# y_ana_np1 = 1/params.eps*(1-Xd)*(xnp1+1) #(1-np.exp((xnp-1)/eps))/(1-np.exp(-1/eps)) + xnp - 1#(np.exp(1/eps)-np.exp((xnp)/eps))/(np.exp(1/eps) -1)
# y_ana_np2 = 1/params.eps*(Xd+1)*(1-xnp2)

# x_d = torch.linspace
analytical = analytical_from_params(params)
y_ana_np = analytical(x_draw)

matplotlib.rc('text', usetex=True)

font = {'family' : 'sans-serif', 'size' : 21}
matplotlib.rc('font', **font)

##########################################################################
# Result and exact solution
##########################################################################
fig, ax = plt.subplots()
ax.plot(x_draw.detach().cpu(), y_draw.detach().cpu(),'-', color=vpinn_c, label = 'SVPINN',linewidth = 2)
ax.plot(x_draw, y_ana_np, '--', color=analytical_c, label="Analytical", linewidth=2)
ax.legend(loc='upper left', labelcolor='linecolor')
ax.set_xlabel(r" $x$ ")
ax.set_ylabel(r" $u$ ")
# ax.set_title("NN approximation and exact solution")
save_fig(fig, tag, "result.png")
save_fig(fig, tag, "result.pdf")

##########################################################################
# Derivative of result and exact solution
##########################################################################
fig, ax = plt.subplots()
pinn_dx = dfdx(pinn, x, order=1)
exact_dx = analytical.dx(x)
ax.plot(x_draw.detach().cpu(), pinn_dx.detach().cpu(),'-', label = 'SVPINN dx', color=vpinn_c)
ax.plot(x_draw.detach().cpu(), exact_dx.detach().cpu(),'--', linewidth=2, label='Analytical dx', color=analytical_c)
ax.set_title("Derivative of NN approximation and exact solution")
ax.set_xlabel(r" $x$ ")
ax.set_ylabel(r" $du/dx$ ")
ax.legend(loc='lower left')#, fontsize='x-large')
save_fig(fig, tag, "result-derivative.png")
save_fig(fig, tag, "result-derivative.pdf")

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
# Loss and error
##########################################################################
fig, ax = plt.subplots()
loss_label = r"$\frac{\sqrt{{\cal L \rm}_r^\phi(u_\theta)}}{\|u\|_U}$"
norm_label = r"$\frac{\|\phi\|_V}{\|u\|_U}$"
error_label = r"$\frac{\|u - u_\theta\|_U}{\|u\|_U}$"
ax.loglog(pos_vec, torch.sqrt(loss_vector[pos_vec]) / analytical_norm,'-',linewidth = 1.5, label=loss_label, color=loss_c)
ax.loglog(pos_vec, torch.sqrt(norm_vector[pos_vec]) / analytical_norm, '-.', linewidth=2, label=norm_label, color=norm_c)
ax.loglog(pos_vec, error_vector[pos_vec], '--', linewidth=1.5, label=error_label, color=error_c)
ax.legend(loc='lower left', labelcolor='linecolor')
ax.set_xlabel(r" Iterations ")
ax.set_ylabel(r" Error (estimates)")
save_fig(fig, tag, "error-and-loss.png")
save_fig(fig, tag, "error-and-loss.pdf")

##########################################################################
# Loss without filter
##########################################################################
fig, ax = plt.subplots()
ax.loglog(epochs_vector, loss_vector,'-',linewidth = 1.5, label="Loss", color=loss_c)
# ax.legend()
ax.set_title("Loss no filer")
ax.set_xlabel(r" Iterations ")
ax.set_ylabel(r" Loss ")
save_fig(fig, tag, "loss-no-filter.png")
save_fig(fig, tag, "loss-no-filter.pdf")

##########################################################################
# Error no fiter
##########################################################################
fig, ax = plt.subplots()
ax.loglog(epochs_vector, error_vector, '-', linewidth=1.5, label="Error", color=error_c)
ax.set_xlabel(r" Iterations ")
ax.set_ylabel(r" Error ")
ax.set_title("Error no filter")
save_fig(fig, tag, "error-no-filter.png")
save_fig(fig, tag, "error-no-filter.pdf")

##########################################################################
# Error to sqrt(loss)
##########################################################################
fig, ax = plt.subplots()
level = pos_vec[int(np.floor(len(pos_vec) * 0.08))]
ax.loglog(np.sqrt(loss_vector[pos_vec]), error_vector[pos_vec], color=error_c, label="Error")
mpl.slope_marker((loss_vector[level]**(1/2), 0.6*error_vector[level]), (1, 1), \
ax=ax, invert=False, poly_kwargs={'facecolor': 'white',
                                    'edgecolor':'black'})
# ax.loglog(np.sqrt(loss_vector[pos_vec]), np.sqrt(loss_vector[pos_vec]), color=loss_c, label="$y=x$")
ax.set_xlabel(r"$\sqrt{Loss}$")
ax.set_ylabel(r"Relative Error ")
# ax.set_title(r"Error to $\sqrt{Loss}$")
filename = f"{get_tag_path(tag)}/error.pt"
save_fig(fig, tag, "error-to-sqrt-loss.png")
save_fig(fig, tag, "error-to-sqrt-loss.pdf")

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
