import numpy as np
import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt

def pacemaker(kf):

    # Constants
    tau = 0.2800
    tau_prime = 0.4977
    beta = 2.5000
    w_0 = 2.2829
    u_e = 0.4111
    m1 = 1.0
    m2 = 1.0
    a = 1.0

    tau *= kf
    tau_prime *= kf

    dt = 0.01

    # Variables
    u1 = 0.0
    u2 = 0.0
    v1 = 0.0
    v2 = 0.0
    y1 = 0.0
    y2 = 0.0
    o = 0.0

    o_list = list()
    t_list = list()

    for t in np.arange(0.0, 5.0, dt):

        d_u1_dt = (-u1 - w_0*y2 -beta*v1 + u_e)/tau
        d_v1_dt = (-v1 + y1)/tau_prime
        y1 = max([0.0, u1])

        d_u2_dt = (-u2 - w_0*y1 -beta*v2 + u_e)/tau
        d_v2_dt = (-v2 + y2)/tau_prime
        y2 = max([0.0, u2])

        u1 += d_u1_dt * dt
        u2 += d_u2_dt * dt
        v1 += d_v1_dt * dt
        v2 += d_v2_dt * dt

        o = -m1*y1 + m2*y2

        o_list.append(o)
        t_list.append(t)

    return t_list, o_list

# Trying out different values of kf
#t_list, o_list_1 = pacemaker(0.1000)
#t_list, o_list_2 = pacemaker(0.2193) # Value in paper
t_list, o_list_2 = pacemaker(0.2000)
t_list, o_list_3 = pacemaker(0.3000)
t_list, o_list_4 = pacemaker(0.4000)

font_size = 18
plt.rcParams.update({'font.size': font_size})


fig, axes = plt.subplots(nrows=3, sharex=True, sharey=True)

#plt.plot(t_list, o_list_1, color='blue', label='kf=0.1', linewidth=2.0)
l0, = axes[0].plot(t_list, o_list_2, color='blue', label=r'$k_f=0.2$', linewidth=3.0)
#axes[0].grid()
axes[0].set_ylim([-0.25,0.25])
axes[0].set_xticks([1.0, 2.0, 3.0, 4.0, 5.0])
axes[0].set_yticks([-0.2, 0.0, 0.2])
for tick in axes[0].xaxis.get_major_ticks():
                tick.label.set_fontsize(22)
for tick in axes[0].yaxis.get_major_ticks():
                tick.label.set_fontsize(22)

l1, = axes[1].plot(t_list, o_list_3, color='red', label=r'$k_f=0.3$', linewidth=3.0)
#axes[1].grid()
axes[1].set_ylim([-0.25,0.25])
axes[1].set_xticks([1.0, 2.0, 3.0, 4.0, 5.0])
axes[1].set_yticks([-0.2, 0.0, 0.2])
for tick in axes[1].xaxis.get_major_ticks():
                tick.label.set_fontsize(22)
for tick in axes[1].yaxis.get_major_ticks():
                tick.label.set_fontsize(22)


l2, = axes[2].plot(t_list, o_list_4, color='green', label=r'$k_f=0.4$', linewidth=3.0)
#axes[2].grid()
axes[2].set_ylim([-0.25,0.25])
axes[2].set_xticks([1.0, 2.0, 3.0, 4.0, 5.0])
axes[2].set_yticks([-0.2, 0.0, 0.2])
for tick in axes[2].xaxis.get_major_ticks():
                tick.label.set_fontsize(22)
for tick in axes[2].yaxis.get_major_ticks():
                tick.label.set_fontsize(22)

fig.text(0.5, 0.20, 'Time (seconds)', ha='center', fontsize=25)
fig.text(0.04, 0.61, 'Oscillator output (radians)', va='center', rotation='vertical', fontsize=25)

fig.subplots_adjust(bottom=0.3, wspace=0.2)
axes[2].legend(handles = [l0,l1,l2] , bbox_to_anchor=(1.0121, 4.0),fancybox=False, shadow=False, ncol=3)

plt.show()

