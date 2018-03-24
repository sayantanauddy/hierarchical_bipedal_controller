import numpy as np
import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt

# Constants
a = 2.5
b = 2.5
c = 1.5
tau = 0.25
T = 0.5

dt = 0.01

# Variables
x1 = 0.5
x2 = 0.0
v1 = 0.0
v2 = 0.0
y1 = 0.0
y2 = 0.0

x1_list = list()
x2_list = list()
y1_list = list()
y2_list = list()
t_list = list()

for t in np.arange(0.0, 10.0, dt):

    d_x1_dt = (-x1 + c - a*y2 -b*v1)/tau
    d_v1_dt = (-v1 + y1)/T
    y1 = max([0.0, x1])

    d_x2_dt = (-x2 + c - a*y1 -b*v2)/tau
    d_v2_dt = (-v2 + y2)/T
    y2 = max([0.0, x2])

    x1 += d_x1_dt * dt
    x2 += d_x2_dt * dt
    v1 += d_v1_dt * dt
    v2 += d_v2_dt * dt

    x1_list.append(x1)
    y1_list.append(y1)
    t_list.append(t)

plt.figure(1)
plt.plot(t_list, x1_list, color='red')
plt.plot(t_list, y1_list, color='blue')
plt.xticks(np.arange(0.0, 10.0, 1.0))
plt.yticks(np.arange(-1.0, 1.0, 0.1))
plt.show()

