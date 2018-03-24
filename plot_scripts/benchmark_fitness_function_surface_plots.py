from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import os

try:
    import numpy as np
except:
    exit()

from deap import benchmarks

# Set the home directory
home_dir = os.path.expanduser('~')

# Set the directory for saving plots
plot_dir = os.path.join(home_dir, 'computing/repositories/MScThesis_SayantanAuddy_2017_NICOOscillatorWalking/plots')


def griewank_arg0(sol):
    return benchmarks.griewank(sol)[0]

# Plot for Griewank

fig = plt.figure(1)
ax = Axes3D(fig, azim = -29, elev = 40)
# ax = Axes3D(fig)
X = np.arange(-50, 50, 0.5)
Y = np.arange(-50, 50, 0.5)
X, Y = np.meshgrid(X, Y)
Z = np.fromiter(map(griewank_arg0, zip(X.flat,Y.flat)), dtype=np.float, count=X.shape[0]*X.shape[1]).reshape(X.shape)

ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.jet, linewidth=0.2)
 
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.savefig(os.path.join(plot_dir, 'griewank.eps'))


# Plot for Rastigrin

def rastrigin_arg0(sol):
    return benchmarks.rastrigin(sol)[0]


fig = plt.figure(2)
ax = Axes3D(fig, azim=-29, elev=50)
X = np.arange(-5, 5, 0.1)
Y = np.arange(-5, 5, 0.1)
X, Y = np.meshgrid(X, Y)
Z = np.fromiter(map(rastrigin_arg0, zip(X.flat, Y.flat)), dtype=np.float, count=X.shape[0] * X.shape[1]).reshape(
    X.shape)

ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.jet, linewidth=0.2)

plt.xlabel("x")
plt.ylabel("y")

plt.savefig(os.path.join(plot_dir, 'rastrigin.eps'))


# Plot for Schwefel

def schwefel_arg0(sol):
    return benchmarks.schwefel(sol)[0]

fig = plt.figure(3)
# ax = Axes3D(fig, azim = -29, elev = 50)
ax = Axes3D(fig)
X = np.arange(-500, 500, 10)
Y = np.arange(-500, 500, 10)
X, Y = np.meshgrid(X, Y)
Z = np.fromiter(map(schwefel_arg0, zip(X.flat,Y.flat)), dtype=np.float, count=X.shape[0]*X.shape[1]).reshape(X.shape)

ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.jet, linewidth=0.2)

plt.xlabel("x")
plt.ylabel("y")

plt.savefig(os.path.join(plot_dir, 'schwefel.eps'))


# Plot for Shekel

A = [[0.5, 0.5], [0.25, 0.25], [0.25, 0.75], [0.75, 0.25], [0.75, 0.75]]
C = [0.002, 0.005, 0.005, 0.005, 0.005]


def shekel_arg0(sol):
    return benchmarks.shekel(sol, A, C)[0]


fig = plt.figure(4)
# ax = Axes3D(fig, azim = -29, elev = 50)
ax = Axes3D(fig)
X = np.arange(0, 1, 0.01)
Y = np.arange(0, 1, 0.01)
X, Y = np.meshgrid(X, Y)
Z = np.fromiter(map(shekel_arg0, zip(X.flat, Y.flat)), dtype=np.float, count=X.shape[0] * X.shape[1]).reshape(X.shape)

ax.plot_surface(X, Y, Z, rstride=1, cstride=1, norm=LogNorm(), cmap=cm.jet, linewidth=0.2)

plt.xlabel("x")
plt.ylabel("y")

plt.savefig(os.path.join(plot_dir, 'shekel.eps'))
