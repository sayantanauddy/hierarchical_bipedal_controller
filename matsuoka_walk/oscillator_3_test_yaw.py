# Script with angle feedback for the saggital hip joints
# This script is executed from the Genetic Algorithm script

import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os
import time

from matsuoka_walk.log import log
from matsuoka_walk.monitor import RobotMonitorThread
from matsuoka_walk.fitness import calc_fitness
from matsuoka_walk.robots import Nico


def oscillator_nw(position_vector, max_time=20.0, fitness_option=6, gain_l=1.0, gain_r=1.0):
    home_dir = os.path.expanduser('~')

    # log('[OSC] Running oscillator_3.oscillator_nw')

    # Extract the elements from the position vector
    kf = position_vector[0]
    GAIN1 = position_vector[1]
    GAIN2 = position_vector[2]
    GAIN3 = position_vector[3]
    GAIN4 = position_vector[4]
    GAIN5 = position_vector[5]
    GAIN6 = position_vector[6]
    BIAS1 = position_vector[7]
    BIAS2 = position_vector[8]
    BIAS3 = position_vector[9]
    BIAS4 = position_vector[10]
    k = position_vector[11]

    # log('[OSC] Printing chromosome')
    # log('[OSC] kf:{0} GAIN1:{1} GAIN2:{2} GAIN3:{3} GAIN4:{4} GAIN5:{5} GAIN6:{6} BIAS1:{7} BIAS2:{8} BIAS3:{9} BIAS4:{10} k:{11}'.
    #    format(kf, GAIN1, GAIN2, GAIN3, GAIN4, GAIN5, GAIN6, BIAS1, BIAS2, BIAS3, BIAS4, k))

    # Try to connect to VREP
    try_counter = 0
    try_max = 5
    robot_handle = None
    while robot_handle is None:
        try:
            # log('[OSC] Trying to create robot handle (attempt: {0} of {1})'.format(try_counter, try_max))
            try_counter += 1
            robot_handle = Nico(sync_sleep_time=0.1,
                                motor_config=os.path.join(home_dir,
                                                          'computing/repositories/hierarchical_bipedal_controller/motor_configs/nico_humanoid_full_v1.json'),
                                vrep=True,
                                vrep_host='127.0.0.1',
                                vrep_port=19997,
                                vrep_scene=os.path.join(home_dir,
                                                        'computing/repositories/hierarchical_bipedal_controller/vrep_scenes/NICO_static_test.ttt')
                                )

        except Exception, e:
            # log('[OSC] Could not connect to VREP')
            # log('[OSC] Error: {0}'.format(e.message))
            time.sleep(1.0)

        if try_counter > try_max:
            # log('[OSC] Unable to create robot handle after {0} tries'.format(try_max))
            exit(1)

            # if robot_handle is not None:
            # log('[OSC] Successfully connected to VREP')

    # Start the monitoring thread
    monitor_thread = RobotMonitorThread(portnum=19998, objname='torso_11_respondable', height_threshold=0.3)
    monitor_thread.start()
    # log('[OSC] Started monitoring thread')

    # Wait 1s for the monitoring thread
    time.sleep(1.0)

    # Note the current position
    start_pos_x = monitor_thread.x
    start_pos_y = monitor_thread.y
    start_pos_z = monitor_thread.z

    # Strange error handler
    if start_pos_y is None:
        start_pos_x = 0.0
    if start_pos_y is None:
        start_pos_y = 0.0
    if start_pos_z is None:
        start_pos_z = 0.0

    # Set up the oscillator constants
    tau = 0.2800
    tau_prime = 0.4977
    beta = 2.5000
    w_0 = 2.2829
    u_e = 0.4111
    m1 = 1.0
    m2 = 1.0
    a = 1.0

    # Modify the time constants based on kf
    tau *= kf
    tau_prime *= kf

    # Step time
    dt = 0.01

    # Variables
    # Oscillator 1 (pacemaker)
    u1_1, u2_1, v1_1, v2_1, y1_1, y2_1, o_1, gain_1, bias_1 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0
    # Oscillator 2
    u1_2, u2_2, v1_2, v2_2, y1_2, y2_2, o_2, gain_2, bias_2 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, GAIN1, BIAS1
    # Oscillator 3
    u1_3, u2_3, v1_3, v2_3, y1_3, y2_3, o_3, gain_3, bias_3 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, GAIN1, BIAS1
    # Oscillator 4
    u1_4, u2_4, v1_4, v2_4, y1_4, y2_4, o_4, gain_4, bias_4 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, GAIN3, BIAS2
    # Oscillator 5
    u1_5, u2_5, v1_5, v2_5, y1_5, y2_5, o_5, gain_5, bias_5 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, GAIN3, BIAS2
    # Oscillator 6
    u1_6, u2_6, v1_6, v2_6, y1_6, y2_6, o_6, gain_6, bias_6 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, GAIN2, BIAS3
    # Oscillator 7
    u1_7, u2_7, v1_7, v2_7, y1_7, y2_7, o_7, gain_7, bias_7 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, GAIN2, BIAS3
    # Oscillator 8
    u1_8, u2_8, v1_8, v2_8, y1_8, y2_8, o_8, gain_8, bias_8 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, GAIN4, 0.0
    # Oscillator 9
    u1_9, u2_9, v1_9, v2_9, y1_9, y2_9, o_9, gain_9, bias_9 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, GAIN5, 0.0
    # Oscillator 10
    u1_10, u2_10, v1_10, v2_10, y1_10, y2_10, o_10, gain_10, bias_10 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, GAIN4, 0.0
    # Oscillator 11
    u1_11, u2_11, v1_11, v2_11, y1_11, y2_11, o_11, gain_11, bias_11 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, GAIN5, 0.0
    # Oscillator 12
    u1_12, u2_12, v1_12, v2_12, y1_12, y2_12, o_12, gain_12, bias_12 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, GAIN6, BIAS4
    # Oscillator 13
    u1_13, u2_13, v1_13, v2_13, y1_13, y2_13, o_13, gain_13, bias_13 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, GAIN6, BIAS4

    # For plots - not needed now
    # o1_list = list()
    # o2_list = list()
    # o3_list = list()
    # o4_list = list()
    # o5_list = list()
    # o6_list = list()
    # o7_list = list()
    # o8_list = list()
    # o9_list = list()
    # o10_list = list()
    # o11_list = list()
    # o12_list = list()
    # o13_list = list()
    # t_list = list()

    def oscillator_next(u1, u2, v1, v2, y1, y2, f1, f2, s1, s2, bias, gain, dt):
        """
        Calculates the state variables in the next time step
        """
        # The extensor neuron
        d_u1_dt = (-u1 - w_0 * y2 - beta * v1 + u_e + f1 + a * s1) / tau
        d_v1_dt = (-v1 + y1) / tau_prime
        y1 = max([0.0, u1])

        # The flexor neuron
        d_u2_dt = (-u2 - w_0 * y1 - beta * v2 + u_e + f2 + a * s2) / tau
        d_v2_dt = (-v2 + y2) / tau_prime
        y2 = max([0.0, u2])

        u1 += d_u1_dt * dt
        u2 += d_u2_dt * dt
        v1 += d_v1_dt * dt
        v2 += d_v2_dt * dt

        o = bias + gain * (-m1 * y1 + m2 * y2)

        return u1, u2, v1, v2, y1, y2, o

    # Set the joints to the initial bias positions - use slow angle setter
    initial_bias_angles = {
        'l_hip_y': bias_2,
        'r_hip_y': bias_3,
        'l_knee_y': bias_4,
        'r_knee_y': bias_5,
        'l_ankle_y': bias_6,
        'r_ankle_y': bias_7,
        'l_shoulder_y': bias_12,
        'r_shoulder_y': bias_13
    }
    robot_handle.set_angles_slow(target_angles=initial_bias_angles, duration=5.0, step=0.01)

    # Sleep for 2 seconds to let any oscillations to die down
    time.sleep(2.0)

    # log('[OSC] Resetting monitoring thread timer')
    # Reset the timer of the monitor
    monitor_thread.reset_timer()

    # New variable for logging up time, since monitor thread is not accurate some times
    up_t = 0.0

    for t in np.arange(0.0, max_time, dt):

        # Increment the up time variable
        up_t = t

        # Calculate the current angles of the l and r saggital hip joints
        feedback_angles = robot_handle.get_angles(['l_hip_y', 'r_hip_y'])

        # Calculate next state of oscillator 1 (pacemaker)
        f1_1, f2_1 = 0.0, 0.0
        s1_1, s2_1 = 0.0, 0.0
        u1_1, u2_1, v1_1, v2_1, y1_1, y2_1, o_1 = oscillator_next(u1=u1_1, u2=u2_1, v1=v1_1, v2=v2_1, y1=y1_1, y2=y2_1,
                                                                  f1=f1_1, f2=f2_1, s1=s1_1, s2=s2_1,
                                                                  bias=bias_1, gain=gain_1,
                                                                  dt=dt)

        # Calculate next state of oscillator 2
        # w_ij -> j=1 (oscillator 1) is master, i=2 (oscillator 2) is slave
        w_21 = 1.0
        f1_2, f2_2 = k * feedback_angles['l_hip_y'], -k * feedback_angles['l_hip_y']
        s1_2, s2_2 = w_21 * u1_1, w_21 * u2_1  # s1_i = w_ij*u1_j, s2_i = w_ij*u2_j
        u1_2, u2_2, v1_2, v2_2, y1_2, y2_2, o_2 = oscillator_next(u1=u1_2, u2=u2_2, v1=v1_2, v2=v2_2, y1=y1_2, y2=y2_2,
                                                                  f1=f1_2, f2=f2_2, s1=s1_2, s2=s2_2,
                                                                  bias=bias_2, gain=gain_l * gain_2,
                                                                  dt=dt)

        # Calculate next state of oscillator 3
        # w_ij -> j=1 (oscillator 1) is master, i=3 (oscillator 3) is slave
        w_31 = -1.0
        f1_3, f2_3 = k * feedback_angles['r_hip_y'], -k * feedback_angles['r_hip_y']
        s1_3, s2_3 = w_31 * u1_1, w_31 * u2_1  # s1_i = w_ij*u1_j, s2_i = w_ij*u2_j
        u1_3, u2_3, v1_3, v2_3, y1_3, y2_3, o_3 = oscillator_next(u1=u1_3, u2=u2_3, v1=v1_3, v2=v2_3, y1=y1_3, y2=y2_3,
                                                                  f1=f1_3, f2=f2_3, s1=s1_3, s2=s2_3,
                                                                  bias=bias_3, gain=gain_r * gain_3,
                                                                  dt=dt)

        # Calculate next state of oscillator 4
        # w_ij -> j=2 (oscillator 2) is master, i=4 (oscillator 4) is slave
        w_42 = -1.0
        f1_4, f2_4 = 0.0, 0.0
        s1_4, s2_4 = w_42 * u1_2, w_42 * u2_2  # s1_i = w_ij*u1_j, s2_i = w_ij*u2_j
        u1_4, u2_4, v1_4, v2_4, y1_4, y2_4, o_4 = oscillator_next(u1=u1_4, u2=u2_4, v1=v1_4, v2=v2_4, y1=y1_4, y2=y2_4,
                                                                  f1=f1_4, f2=f2_4, s1=s1_4, s2=s2_4,
                                                                  bias=bias_4, gain=gain_4,
                                                                  dt=dt)

        # Calculate next state of oscillator 5
        # w_ij -> j=3 (oscillator 3) is master, i=5 (oscillator 5) is slave
        w_53 = -1.0
        f1_5, f2_5 = 0.0, 0.0
        s1_5, s2_5 = w_53 * u1_3, w_53 * u2_3  # s1_i = w_ij*u1_j, s2_i = w_ij*u2_j
        u1_5, u2_5, v1_5, v2_5, y1_5, y2_5, o_5 = oscillator_next(u1=u1_5, u2=u2_5, v1=v1_5, v2=v2_5, y1=y1_5, y2=y2_5,
                                                                  f1=f1_5, f2=f2_5, s1=s1_5, s2=s2_5,
                                                                  bias=bias_5, gain=gain_5,
                                                                  dt=dt)

        # Calculate next state of oscillator 6
        # w_ij -> j=2 (oscillator 2) is master, i=6 (oscillator 6) is slave
        w_62 = -1.0
        f1_6, f2_6 = 0.0, 0.0
        s1_6, s2_6 = w_62 * u1_2, w_62 * u2_2  # s1_i = w_ij*u1_j, s2_i = w_ij*u2_j
        u1_6, u2_6, v1_6, v2_6, y1_6, y2_6, o_6 = oscillator_next(u1=u1_6, u2=u2_6, v1=v1_6, v2=v2_6, y1=y1_6, y2=y2_6,
                                                                  f1=f1_6, f2=f2_6, s1=s1_6, s2=s2_6,
                                                                  bias=bias_6, gain=gain_6,
                                                                  dt=dt)

        # Calculate next state of oscillator 7
        # w_ij -> j=3 (oscillator 3) is master, i=7 (oscillator 7) is slave
        w_73 = -1.0
        f1_7, f2_7 = 0.0, 0.0
        s1_7, s2_7 = w_73 * u1_3, w_73 * u2_3  # s1_i = w_ij*u1_j, s2_i = w_ij*u2_j
        u1_7, u2_7, v1_7, v2_7, y1_7, y2_7, o_7 = oscillator_next(u1=u1_7, u2=u2_7, v1=v1_7, v2=v2_7, y1=y1_7, y2=y2_7,
                                                                  f1=f1_7, f2=f2_7, s1=s1_7, s2=s2_7,
                                                                  bias=bias_7, gain=gain_7,
                                                                  dt=dt)

        # Calculate next state of oscillator 8
        # w_ij -> j=1 (oscillator 1) is master, i=8 (oscillator 8) is slave
        w_81 = 1.0
        f1_8, f2_8 = 0.0, 0.0
        s1_8, s2_8 = w_81 * u1_1, w_81 * u2_1  # s1_i = w_ij*u1_j, s2_i = w_ij*u2_j
        u1_8, u2_8, v1_8, v2_8, y1_8, y2_8, o_8 = oscillator_next(u1=u1_8, u2=u2_8, v1=v1_8, v2=v2_8, y1=y1_8, y2=y2_8,
                                                                  f1=f1_8, f2=f2_8, s1=s1_8, s2=s2_8,
                                                                  bias=bias_8, gain=gain_8,
                                                                  dt=dt)

        # Calculate next state of oscillator 9
        # w_ij -> j=8 (oscillator 8) is master, i=9 (oscillator 9) is slave
        w_98 = -1.0
        f1_9, f2_9 = 0.0, 0.0
        s1_9, s2_9 = w_98 * u1_8, w_98 * u2_8  # s1_i = w_ij*u1_j, s2_i = w_ij*u2_j
        u1_9, u2_9, v1_9, v2_9, y1_9, y2_9, o_9 = oscillator_next(u1=u1_9, u2=u2_9, v1=v1_9, v2=v2_9, y1=y1_9, y2=y2_9,
                                                                  f1=f1_9, f2=f2_9, s1=s1_9, s2=s2_9,
                                                                  bias=bias_9, gain=gain_9,
                                                                  dt=dt)

        # Calculate next state of oscillator 10
        # w_ij -> j=1 (oscillator 1) is master, i=10 (oscillator 10) is slave
        w_101 = 1.0
        f1_10, f2_10 = 0.0, 0.0
        s1_10, s2_10 = w_101 * u1_1, w_101 * u2_1  # s1_i = w_ij*u1_j, s2_i = w_ij*u2_j
        u1_10, u2_10, v1_10, v2_10, y1_10, y2_10, o_10 = oscillator_next(u1=u1_10, u2=u2_10, v1=v1_10, v2=v2_10,
                                                                         y1=y1_10, y2=y2_10,
                                                                         f1=f1_10, f2=f2_10, s1=s1_10, s2=s2_10,
                                                                         bias=bias_10, gain=gain_10,
                                                                         dt=dt)

        # Calculate next state of oscillator 11
        # w_ij -> j=10 (oscillator 10) is master, i=11 (oscillator 11) is slave
        w_1110 = -1.0
        f1_11, f2_11 = 0.0, 0.0
        s1_11, s2_11 = w_1110 * u1_10, w_1110 * u2_10  # s1_i = w_ij*u1_j, s2_i = w_ij*u2_j
        u1_11, u2_11, v1_11, v2_11, y1_11, y2_11, o_11 = oscillator_next(u1=u1_11, u2=u2_11, v1=v1_11, v2=v2_11,
                                                                         y1=y1_11, y2=y2_11,
                                                                         f1=f1_11, f2=f2_11, s1=s1_11, s2=s2_11,
                                                                         bias=bias_11, gain=gain_11,
                                                                         dt=dt)

        # Calculate next state of oscillator 12
        # w_ij -> j=1 (oscillator 1) is master, i=12 (oscillator 12) is slave
        w_121 = -1.0
        f1_12, f2_12 = 0.0, 0.0
        s1_12, s2_12 = w_121 * u1_1, w_121 * u2_1  # s1_i = w_ij*u1_j, s2_i = w_ij*u2_j
        u1_12, u2_12, v1_12, v2_12, y1_12, y2_12, o_12 = oscillator_next(u1=u1_12, u2=u2_12, v1=v1_12, v2=v2_12,
                                                                         y1=y1_12, y2=y2_12,
                                                                         f1=f1_12, f2=f2_12, s1=s1_12, s2=s2_12,
                                                                         bias=bias_12, gain=gain_12,
                                                                         dt=dt)

        # Calculate next state of oscillator 13
        # w_ij -> j=1 (oscillator 1) is master, i=13 (oscillator 13) is slave
        w_131 = 1.0
        f1_13, f2_13 = 0.0, 0.0
        s1_13, s2_13 = w_131 * u1_1, w_131 * u2_1  # s1_i = w_ij*u1_j, s2_i = w_ij*u2_j
        u1_13, u2_13, v1_13, v2_13, y1_13, y2_13, o_13 = oscillator_next(u1=u1_13, u2=u2_13, v1=v1_13, v2=v2_13,
                                                                         y1=y1_13, y2=y2_13,
                                                                         f1=f1_13, f2=f2_13, s1=s1_13, s2=s2_13,
                                                                         bias=bias_13, gain=gain_13,
                                                                         dt=dt)

        # Set the joint positions
        current_angles = {
            'l_hip_y': o_2,
            'r_hip_y': o_3,
            'l_knee_y': o_4,
            'r_knee_y': o_5,
            'l_ankle_y': o_6,
            'r_ankle_y': o_7,
            'l_hip_x': o_8,
            'l_ankle_x': o_9,
            'r_hip_x': o_10,
            'r_ankle_x': o_11,
            'l_shoulder_y': o_12,
            'r_shoulder_y': o_13,
            'l_hip_z': 0.1 if gain_l > gain_r else 0.0,
            'r_hip_z': -0.1 if gain_r > gain_l else 0.0
        }
        robot_handle.set_angles(current_angles)

        time.sleep(dt)

        # Check if the robot has fallen
        if monitor_thread.fallen:
            break

            # For plots - not needed now
            # o1_list.append(o_1)
            # o2_list.append(o_2)
            # o3_list.append(o_3)
            # o4_list.append(o_4)
            # o5_list.append(o_5)
            # o6_list.append(o_6)
            # o7_list.append(o_7)
            # o8_list.append(o_8)
            # o9_list.append(o_9)
            # o10_list.append(o_10)
            # o11_list.append(o_11)
            # o12_list.append(o_12)
            # o13_list.append(o_13)
            # t_list.append(t)

    # log('[OSC] Accurate up time: {0}'.format(up_t))

    # Outside the loop, it means that either the robot has fallen or the max_time has elapsed
    # Find out the end position of the robot
    end_pos_x = monitor_thread.x
    end_pos_y = monitor_thread.y
    end_pos_z = monitor_thread.z

    torso_euler_angles = monitor_thread.vrepio_obj.call_remote_api('simxGetObjectOrientation',
                                                                   monitor_thread.vrepio_obj.get_object_handle(
                                                                       'torso_11_respondable'),
                                                                   -1,  # Orientation needed with respect to world frame
                                                                   streaming=True)

    torso_gamma = torso_euler_angles[2]

    # Find the average height
    avg_z = monitor_thread.avg_z

    # Find the up time
    # up_time = monitor_thread.up_time
    up_time = up_t

    # Calculate the fitness
    if up_time == 0.0:
        fitness = 0.0
        # log('[OSC] up_t==0 so fitness is set to 0.0')
    else:
        fitness = calc_fitness(start_x=start_pos_x, start_y=start_pos_y, start_z=start_pos_z,
                               end_x=end_pos_x, end_y=end_pos_y, end_z=end_pos_z,
                               avg_z=avg_z,
                               up_time=up_time,
                               fitness_option=fitness_option
                               )

        # if not monitor_thread.fallen:
        # log("Robot has not fallen")
        # else:
        # log("Robot has fallen")

    # log('[OSC] Calculated fitness: {0}'.format(fitness))

    # Stop the monitoring thread
    monitor_thread.stop()

    # Close the VREP connection
    robot_handle.cleanup()

    # For plots - not needed now
    # ax1 = plt.subplot(611)
    # plt.plot(t_list, o1_list, color='red', label='o_1')
    # plt.plot(t_list, o2_list, color='green', ls='--', label='o_2')
    # plt.plot(t_list, o3_list, color='green', label='o_3')
    # plt.grid()
    # plt.legend()
    #
    # ax2 = plt.subplot(612, sharex=ax1, sharey=ax1)
    # plt.plot(t_list, o1_list, color='red', label='o_1')
    # plt.plot(t_list, o4_list, color='blue', ls='--', label='o_4')
    # plt.plot(t_list, o5_list, color='blue', label='o_5')
    # plt.grid()
    # plt.legend()
    #
    # ax3 = plt.subplot(613, sharex=ax1, sharey=ax1)
    # plt.plot(t_list, o1_list, color='red', label='o_1')
    # plt.plot(t_list, o6_list, color='black', ls='--', label='o_6')
    # plt.plot(t_list, o7_list, color='black', label='o_7')
    # plt.grid()
    # plt.legend()
    #
    # ax4 = plt.subplot(614, sharex=ax1, sharey=ax1)
    # plt.plot(t_list, o1_list, color='red', label='o_1')
    # plt.plot(t_list, o8_list, color='cyan', ls='--', label='o_8')
    # plt.plot(t_list, o9_list, color='cyan', label='o_9')
    # plt.grid()
    # plt.legend()
    #
    # ax5 = plt.subplot(615, sharex=ax1, sharey=ax1)
    # plt.plot(t_list, o1_list, color='red', label='o_1')
    # plt.plot(t_list, o10_list, color='orange', ls='--', label='o_10')
    # plt.plot(t_list, o11_list, color='orange', label='o_11')
    # plt.grid()
    # plt.legend()
    #
    # ax6 = plt.subplot(616, sharex=ax1, sharey=ax1)
    # plt.plot(t_list, o1_list, color='red', label='o_1')
    # plt.plot(t_list, o12_list, color='brown', ls='--', label='o_12')
    # plt.plot(t_list, o13_list, color='brown', label='o_13')
    # plt.grid()
    # plt.legend()
    #
    # plt.show()

    return (end_pos_x, end_pos_y, torso_gamma)
