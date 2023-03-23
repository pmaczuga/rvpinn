import matplotlib.pyplot as plt
import torch

from src.analytical import AnalyticalAD, AnalyticalDelta

eps_ad = 0.01
eps_delta = 0.5
Xd = 0.5

x = torch.linspace(-1, 1, 500)

ad    = AnalyticalAD(eps_ad)
delta = AnalyticalDelta(eps_delta, Xd)

y_ad = ad(x)
y_ad_dx = ad.dx(x)
y_delta = delta(x)
y_delta_dx = delta.dx(x)

fig, ax = plt.subplots()
ax.plot(x, y_ad)
ax.set_title("AD")

fig, ax = plt.subplots()
ax.plot(x, y_ad_dx)
ax.set_title("AD dx")

fig, ax = plt.subplots()
ax.plot(x, y_delta)
ax.set_title("Delta")

fig, ax = plt.subplots()
ax.plot(x, y_delta_dx)
ax.set_title("Delta dx")

plt.show()