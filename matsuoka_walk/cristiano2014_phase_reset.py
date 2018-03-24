import numpy as np
import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt

def pacemaker(kf, phase_reset_times):

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

        # Phase reset
        if t in phase_reset_times:
            u1, u2, v1, v2, y1, y2, o = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

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

t_list, o_list_1 = pacemaker(0.2000,phase_reset_times=[1.5,2.5,3.7])
t_list, o_list_2 = pacemaker(0.2000,phase_reset_times=[])

font_size = 18
plt.rcParams.update({'font.size': font_size})
matplotlib.rc('xtick', labelsize=20)
matplotlib.rc('ytick', labelsize=20)

plt.figure(1,figsize=(20,2))
plt.plot(t_list, o_list_1, color='blue', ls='-', label=r'$with\ phase\ reset$', linewidth=3.0)
plt.plot(t_list, o_list_2, color='blue', ls='--', label=r'$without\ phase\ reset$', linewidth=3.0)
plt.plot((1.5,1.5),(-.25,.25), color='black', linewidth=2.0)
plt.plot((2.5,2.5),(-.25,.25), color='black', linewidth=2.0)
plt.plot((3.7,3.7),(-.25,.25), color='black', linewidth=2.0)

plt.xlim(0.0,5.0)
plt.ylim(-0.25,0.25)
plt.xticks([0.0,1.0,1.5,2.0,2.5,3.0,3.7,4.0,5.0])
plt.yticks([-0.2,0.0,0.2])
plt.xlabel('Time (seconds)', fontsize=25)
plt.ylabel('Oscillator output (radians)', fontsize=25)
#plt.grid()
plt.legend(bbox_to_anchor=(1.013, 1), loc='lower right', ncol=2)
plt.show()
