import argparse
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import draw, figure, show
import numpy as np
import torch
from src.analytical import AnalyticalDelta, analytical_from_params

from src.params import Params
from src.io_utils import load_result
from src.pinn import dfdx, f

matplotlib.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']


parser = argparse.ArgumentParser(
                    prog='DRVPINN',
                    description='Runs the training of DRVPINN')
parser.add_argument('--tag', type=str)
args = parser.parse_args()

# Here you can set tag - that is directory inside results where the 
# tag = "tmp"
# Alternatively you can take tag from params.ini in root:
tag = args.tag if args.tag is not None else Params().tag

# Here is trained pinn, loss vector and parameters
pinn, result, params = load_result(tag)
loss_vector = result.loss
error_vector = result.error
norm_vector = result.norm

x_domain = [-1., 1.0]; n_points_x=params.n_points_x 
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

fig, ax = plt.subplots() #figure(figsize=(7, 7))
ax.plot(x_draw.detach().cpu(), y_draw.detach().cpu(),'b-', label = 'DRVPINNs',linewidth = 2)
# ax.plot(xnp1, y_ana_np1,'r--',linewidth = 2)
# ax.plot(xnp2, y_ana_np2,'r--', label = 'Analytical',linewidth = 2)
ax.plot(x_draw, y_ana_np, 'r--', label="Analytical", linewidth=2)

ax.legend(loc='upper left')#, fontsize='x-large')
#plt.plot(x_draw.detach().cpu(), y_ana.detach().cpu(),'r--')
ax.set_xlabel(r" $x$ ", size = 22)
ax.set_ylabel(r" $u$ ", size = 22)

##########################################################################

figure(figsize=(10, 10))
delta_approx = dfdx(pinn, x, order=1)
plt.plot(x_draw.detach().cpu(), delta_approx.detach().cpu(),'b-')
#plt.semilogy(x_draw.detach().cpu(), abs(y_ana.detach().cpu()-y_draw.detach().cpu()),'b-')

##########################################################################

vec = loss_vector
best = 1
best_vec = [1.]
pos_vec = [1.]



for n in range(1,params.epochs):
  if vec[n]<best and vec[n]>0:
    best_vec.append(vec[n])
    pos_vec.append(n+1)
    best = 1*vec[n]

pos_vec = np.array(pos_vec, dtype=int) - 1

fig, ax = plt.subplots()
ax.loglog(pos_vec, loss_vector[pos_vec],'b-',linewidth = 2, label="Loss")
ax.loglog(pos_vec, norm_vector[pos_vec], 'r--', linewidth=2, label="Norm")
ax.legend()
ax.set_xlabel(r" Iterations ", size = 22)
ax.set_ylabel(r" Loss ", size = 22)

fig, ax = plt.subplots()
ax.loglog(pos_vec, error_vector[pos_vec], 'g-', linewidth=2, label="Error")
ax.set_xlabel(r" Iterations ", size = 22)
ax.set_ylabel(r" Loss ", size = 22)
ax.set_title("Error")

###########################################################

fig, ax = plt.subplots()
ax.loglog(np.sqrt(loss_vector[pos_vec]), error_vector[pos_vec])
ax.loglog(np.sqrt(loss_vector[pos_vec]), np.sqrt(loss_vector[pos_vec]))

# if error_vector is not None:
#   fig, ax = plt.subplots()
#   ax.loglog(error_vector, 'b-', linewidth=2)
#   ax.set_title("Error")
#   ax.set_xlabel(r" Iterations ", size = 22)
#   ax.set_ylabel(r" Error ", size = 22)

# if norm_vector is not None:
#   fig, ax = plt.subplots()
#   ax.loglog(norm_vector, 'b-', linewidth=2)
#   ax.set_title("Norm")
#   ax.set_xlabel(r" Iterations ", size = 22)
#   ax.set_ylabel(r" Norm ", size = 22)

show()
