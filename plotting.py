import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import draw, figure, show
import numpy as np
import torch

from src.params import Params
from src.io_utils import load_result
from src.pinn import dfdx, f

matplotlib.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']

# Here you can set tag - that is directory inside results where the 
# tag = "tmp"
# Alternatively you can take tag from params.ini in root:
tag = Params().tag

# Here is trained pinn, loss vector and parameters
pinn, loss_vector, params = load_result(tag)


x_domain = [-1., 1.0]; n_points_x=params.n_points_x 
x_raw = torch.linspace(x_domain[0], x_domain[1], steps=n_points_x)
x_raw.requires_grad_()
x = x_raw.reshape(-1, 1)
x_draw = x.flatten()
y_draw = f(pinn, x).flatten()
#y_ana = (1-torch.exp((x_draw-1)/eps))/(1-np.exp(-1/eps)) + x_draw -1
Xd = 0.5#np.sqrt(2)/2
xnp1=np.linspace(-1,Xd,100)
xnp2=np.linspace(Xd,1,100)

y_ana_np1 = (1-Xd)*(xnp1+1) #(1-np.exp((xnp-1)/eps))/(1-np.exp(-1/eps)) + xnp - 1#(np.exp(1/eps)-np.exp((xnp)/eps))/(np.exp(1/eps) -1)
y_ana_np2 = (Xd+1)*(1-xnp2)


matplotlib.rc('text', usetex=True)

font = {'family' : 'sans-serif', 'size' : 15}
matplotlib.rc('font', **font)

fig, ax = plt.subplots() #figure(figsize=(7, 7))
ax.plot(x_draw.detach().cpu(), y_draw.detach().cpu(),'b-', label = 'DRVPINNs',linewidth = 2)
ax.plot(xnp1, y_ana_np1,'r--',linewidth = 2)
ax.plot(xnp2, y_ana_np2,'r--', label = 'Analytical',linewidth = 2)
ax.legend(loc='upper left')#, fontsize='x-large')
#plt.plot(x_draw.detach().cpu(), y_ana.detach().cpu(),'r--')
ax.set_xlabel(r" $x$ ", size = 22)
ax.set_ylabel(r" $u$ ", size = 22)

##########################################################################

figure(figsize=(10, 10))
delta_approx = -dfdx(pinn, x, order=2)
plt.plot(x_draw.detach().cpu(), delta_approx.detach().cpu(),'b-')
#plt.semilogy(x_draw.detach().cpu(), abs(y_ana.detach().cpu()-y_draw.detach().cpu()),'b-')

##########################################################################

vec = loss_vector/loss_vector[0]
best = 1
best_vec = [1.]
pos_vec = [1.]



for n in range(1,params.epochs):
  if vec[n]<best and vec[n]>0:
    best_vec.append(vec[n])
    pos_vec.append(n+1)
    best = 1*vec[n]

#print(vec)


fig, ax = plt.subplots()
ax.loglog(pos_vec, best_vec,'b-',linewidth = 2)
ax.set_xlabel(r" Iterations ", size = 22)
ax.set_ylabel(r" Relative loss ", size = 22)
ax.set_ylim([1e-6,2])

show()
