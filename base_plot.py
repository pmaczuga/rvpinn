import torch
from src.base_fun import *
import matplotlib.pyplot as plt
import numpy as np


fun = FemBase(6)
x = torch.linspace(-1, 1, 200)
y1 = fun(x, 1)

plt.plot(x, fun.dx(x, 1))
# plt.plot(x, fun.dx(x, 2))
# plt.plot(x, fun.dx(x, 3))
plt.show()