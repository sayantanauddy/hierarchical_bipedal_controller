import numpy as np
import os
import time
import gym
from gym import spaces
import pypot
from pypot.vrep.io import VrepIO
from pypot.vrep.io import remote_api
from time import gmtime, strftime
import time

from matsuoka_walk.log import log
from matsuoka_walk.monitor import RobotMonitorThread
from matsuoka_walk.fitness import calc_fitness
from matsuoka_walk.robots import Nico


class MatsuokaAngleFeedbackEnv(gym.Env):

    def gym_init(self):

        log('[ENV] gym_init called')

        # Gym initialization
        # Actions - [gain_factor_l_hip_y, gain_factor_r_hip_y]
        # States - [torso_alpha, torso_beta, torso_gamma, d_torso_alpha, d_torso_beta, d_torso_gamma, torso_x, torso_y, d_torso_x, d_torso_y]

        # Set the max and min gain factor
        # The gain factor is multiplied with the joint gain
        self.gain_factor_max = 1.0
        self.gain_factor_min = 1.0

        # Initialize the gain factors
        self.gain_factor_l_hip_y = 1.0
        self.gain_factor_r_hip_y = 1.0

        # Initialize the action and observation spaces
        obs = np.array([np.inf] * 10)
        self.action_space = spaces.Box(low=np.array([0.75, 0.75]), high=np.array([1.0, 1.0]))
        self.observation_space = spaces.Box(-obs, obs)

        # Variable for the number of steps in the episode
        self.step_counter = 0

    def scale_actions(self, actions):
        """
        Scales the actions to be withing the gain factor limits
        The actions are squashed using a sigmoid activation function but due to addition of noise turing training, their
        values may lie outside the [0,1] range
        0 and below is set to gain_factor_min
        1 and above is set to gain_factor_max
        In-between values are scaled to a value between max and min limits
        :param actions:
        :return: scaled actions
        """

        action_1 = actions[0]
        action_2 = actions[1]
        scaled_action_1 = 0.0
        scaled_action_2 = 0.0

        if action_1 <= 0.0:
            scaled_action_1 = self.gain_factor_min
        elif action_1 >= 1.0:
            scaled_action_1 = self.gain_factor_max
        elif 0.0 < action_1 < 1.0:
            scaled_action_1 = self.gain_factor_min + ((action_1 - 0.0)/(1.0 - 0.0))*(self.gain_factor_max - self.gain_factor_min)

        if action_2 <= 0.0:
            scaled_action_2 = self.gain_factor_min
        elif action_2 >= 1.0:
            scaled_action_2 = self.gain_factor_max
        elif 0.0 < action_2 < 1.0:
            scaled_action_2 = self.gain_factor_min + ((action_2 - 0.0)/(1.0 - 0.0))*(self.gain_factor_max - self.gain_factor_min)

        return [scaled_action_1, scaled_action_2]

    def matsuoka_init(self):

        # Set the home directory
        self.home_dir = os.path.expanduser('~')

        log('[ENV] matsuoka_init called')

        # Max time to walk (in seconds)
        self.max_walk_time = 20

        # Variables from the best chromosome
        position_vector = [0.7461913734531209, 0.8422944031253159, 0.07043758116681641, 0.14236621222553963, 0.48893497409925746, 0.5980055418720059, 0.740811806645801, -0.11618361090424223, 0.492832184960149, -0.2949145038394889, 0.175450703085948, -0.3419733470484183]

        self.kf = position_vector[0]
        self.GAIN1 = position_vector[1]
        self.GAIN2 = position_vector[2]
        self.GAIN3 = position_vector[3]
        self.GAIN4 = position_vector[4]
        self.GAIN5 = position_vector[5]
        self.GAIN6 = position_vector[6]
        self.BIAS1 = position_vector[7]
        self.BIAS2 = position_vector[8]
        self.BIAS3 = position_vector[9]
        self.BIAS4 = position_vector[10]
        self.k = position_vector[11]

        # Create the robot handle
        # Try to connect to VREP
        try_counter = 0
        try_max = 5
        self.robot_handle = None
        while self.robot_handle is None:

            # Close existing connections if any
            pypot.vrep.close_all_connections()

            try:
                log('[ENV] Trying to create robot handle (attempt: {0} of {1})'.format(try_counter, try_max))
                try_counter += 1
                self.robot_handle = Nico(sync_sleep_time=0.1,
                                         motor_config=os.path.join(self.home_dir,
                                                                   'computing/repositories/hierarchical_bipedal_controller/motor_configs/nico_humanoid_full_v1.json'),
                                         vrep=True,
                                         vrep_host='127.0.0.1',
                                         vrep_port=19997,
                                         vrep_scene=os.path.join(self.home_dir,
                                                                 'computing/repositories/hierarchical_bipedal_controller/vrep_scenes/NICO-Simplified-July2017_standing_Foot_sensors_v4_no_graphs.ttt')
                                         )

            except Exception, e:
                log('[ENV] Could not connect to VREP')
                log('[ENV] Error: {0}'.format(e.message))
                time.sleep(1.0)

            if try_counter > try_max:
                log('[ENV] Unable to create robot handle after {0} tries'.format(try_max))
                exit(1)

        if self.robot_handle is not None:
            log('[ENV] Successfully connected to VREP')

        # Start the monitoring thread
        self.monitor_thread = RobotMonitorThread(portnum=19998, objname='torso_11_respondable', height_threshold=0.3)
        self.monitor_thread.start()
        log('[ENV] Started monitoring thread')

        # Wait 1s for the monitoring thread
        time.sleep(1.0)

        # Note the current position
        self.start_pos_x = self.monitor_thread.x
        self.start_pos_y = self.monitor_thread.y
        self.start_pos_z = self.monitor_thread.z

        # Strange error handler
        if self.start_pos_y is None:
            self.start_pos_x = 0.0
        if self.start_pos_y is None:
            self.start_pos_y = 0.0
        if self.start_pos_z is None:
            self.start_pos_z = 0.0

        # Set up the oscillator constants
        self.tau = 0.2800
        self.tau_prime = 0.4977
        self.beta = 2.5000
        self.w_0 = 2.2829
        self.u_e = 0.4111
        self.m1 = 1.0
        self.m2 = 1.0
        self.a = 1.0

        # Modify the time constants based on kf
        self.tau *= self.kf
        self.tau_prime *= self.kf

        # Step times
        self.higher_control_dt = 0.1
        self.lower_control_dt = 0.01

        # Set the oscillator variables
        # Oscillator 1 (pacemaker)
        self.u1_1, self.u2_1, self.v1_1, self.v2_1, self.y1_1, self.y2_1, self.o_1, self.gain_1, self.bias_1 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0
        # Oscillator 2
        self.u1_2, self.u2_2, self.v1_2, self.v2_2, self.y1_2, self.y2_2, self.o_2, self.gain_2, self.bias_2 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, self.GAIN1, self.BIAS1
        # Oscillator 3
        self.u1_3, self.u2_3, self.v1_3, self.v2_3, self.y1_3, self.y2_3, self.o_3, self.gain_3, self.bias_3 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, self.GAIN1, self.BIAS1
        # Oscillator 4
        self.u1_4, self.u2_4, self.v1_4, self.v2_4, self.y1_4, self.y2_4, self.o_4, self.gain_4, self.bias_4 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, self.GAIN3, self.BIAS2
        # Oscillator 5
        self.u1_5, self.u2_5, self.v1_5, self.v2_5, self.y1_5, self.y2_5, self.o_5, self.gain_5, self.bias_5 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, self.GAIN3, self.BIAS2
        # Oscillator 6
        self.u1_6, self.u2_6, self.v1_6, self.v2_6, self.y1_6, self.y2_6, self.o_6, self.gain_6, self.bias_6 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, self.GAIN2, self.BIAS3
        # Oscillator 7
        self.u1_7, self.u2_7, self.v1_7, self.v2_7, self.y1_7, self.y2_7, self.o_7, self.gain_7, self.bias_7 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, self.GAIN2, self.BIAS3
        # Oscillator 8
        self.u1_8, self.u2_8, self.v1_8, self.v2_8, self.y1_8, self.y2_8, self.o_8, self.gain_8, self.bias_8 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, self.GAIN4, 0.0
        # Oscillator 9
        self.u1_9, self.u2_9, self.v1_9, self.v2_9, self.y1_9, self.y2_9, self.o_9, self.gain_9, self.bias_9 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, self.GAIN5, 0.0
        # Oscillator 10
        self.u1_10, self.u2_10, self.v1_10, self.v2_10, self.y1_10, self.y2_10, self.o_10, self.gain_10, self.bias_10 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, self.GAIN4, 0.0
        # Oscillator 11
        self.u1_11, self.u2_11, self.v1_11, self.v2_11, self.y1_11, self.y2_11, self.o_11, self.gain_11, self.bias_11 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, self.GAIN5, 0.0
        # Oscillator 12
        self.u1_12, self.u2_12, self.v1_12, self.v2_12, self.y1_12, self.y2_12, self.o_12, self.gain_12, self.bias_12 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, self.GAIN6, self.BIAS4
        # Oscillator 13
        self.u1_13, self.u2_13, self.v1_13, self.v2_13, self.y1_13, self.y2_13, self.o_13, self.gain_13, self.bias_13 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, self.GAIN6, self.BIAS4

        # Set the joints to the initial bias positions - use slow angle setter
        initial_bias_angles = {
            'l_hip_y': self.bias_2,
            'r_hip_y': self.bias_3,
            'l_knee_y': self.bias_4,
            'r_knee_y': self.bias_5,
            'l_ankle_y': self.bias_6,
            'r_ankle_y': self.bias_7,
            'l_shoulder_y': self.bias_12,
            'r_shoulder_y': self.bias_13
        }
        self.robot_handle.set_angles_slow(target_angles=initial_bias_angles, duration=5.0, step=0.01)

        # Sleep for 2 seconds to let any oscillations to die down
        time.sleep(2.0)

        log('[ENV] Resetting monitoring thread timer')
        # Reset the timer of the monitor
        self.monitor_thread.reset_timer()

        # New variable for logging up time, since monitor thread is not accurate some times
        self.up_t = 0.0

    def __init__(self):
        self.vrep_body_object = 'torso_11_respondable'
        self.matsuoka_init()
        self.gym_init()

    def oscillator_next(self, u1, u2, v1, v2, y1, y2, f1, f2, s1, s2, bias, gain, dt):
        """
        Calculates the state variables in the next time step
        """

        # The extensor neuron
        d_u1_dt = (-u1 - self.w_0*y2 -self.beta*v1 + self.u_e + f1 + self.a*s1)/self.tau
        d_v1_dt = (-v1 + y1)/self.tau_prime
        y1 = max([0.0, u1])

        # The flexor neuron
        d_u2_dt = (-u2 - self.w_0*y1 -self.beta*v2 + self.u_e + f2 + self.a*s2)/self.tau
        d_v2_dt = (-v2 + y2)/self.tau_prime
        y2 = max([0.0, u2])

        u1 += d_u1_dt * dt
        u2 += d_u2_dt * dt
        v1 += d_v1_dt * dt
        v2 += d_v2_dt * dt

        o = bias + gain*(-self.m1*y1 + self.m2*y2)

        return u1, u2, v1, v2, y1, y2, o

    def _self_observe(self):

        # Actions - [gain_factor_l_hip_y, gain_factor_r_hip_y]
        # States - [torso_alpha, torso_beta, torso_gamma, d_torso_alpha, d_torso_beta, d_torso_gamma, torso_x, torso_y, d_torso_x, d_torso_y]

        # Retrieve the torso orientation with respect to the world frame
        self.torso_euler_angles = self.monitor_thread.vrepio_obj.call_remote_api('simxGetObjectOrientation',
                                                                                 self.monitor_thread.vrepio_obj.get_object_handle(self.vrep_body_object),
                                                                                 -1,  # Orientation needed with respect to world frame
                                                                                 streaming=True)

        # Retrieve the torso position
        self.torso_position = self.monitor_thread.vrepio_obj.call_remote_api('simxGetObjectPosition',
                                                                             self.monitor_thread.vrepio_obj.get_object_handle(self.vrep_body_object),
                                                                             -1,  # Orientation needed with respect to world frame
                                                                             streaming=True)

        # Retrieve the torso angular and cartesian velocities
        self.torso_linear_vel, self.torso_angular_vel = self.monitor_thread.vrepio_obj.call_remote_api('simxGetObjectVelocity',
                                                                                                       self.monitor_thread.vrepio_obj.get_object_handle(self.vrep_body_object),
                                                                                                       streaming=True)

        self.torso_alpha = self.torso_euler_angles[0]
        self.torso_beta = self.torso_euler_angles[1]
        self.torso_gamma = self.torso_euler_angles[2]

        self.d_torso_alpha = self.torso_angular_vel[0]
        self.d_torso_beta = self.torso_angular_vel[1]
        self.d_torso_gamma = self.torso_angular_vel[2]

        self.torso_x = self.torso_position[0]
        self.torso_y = self.torso_position[1]
        self.torso_z = self.torso_position[2]

        self.d_torso_x = self.torso_linear_vel[0]
        self.d_torso_y = self.torso_linear_vel[1]
        self.d_torso_z = self.torso_linear_vel[2]

        # Set the observation
        self.observation = np.array([
            self.torso_alpha,
            self.torso_beta,
            self.torso_gamma,
            self.d_torso_alpha,
            self.d_torso_beta,
            self.d_torso_gamma,
            self.torso_x,
            self.torso_y,
            self.torso_z,
            self.d_torso_x,
            self.d_torso_y,
            self.d_torso_z
        ]).astype('float32')

    def _render(self, mode='human', close=False):
        return

    def _step(self, actions):

        # Run 10 iterations of matsuoka loop followed by 1 iteration of checking torso

        # Make observation, take action, calculate reward
        # Scale the actions so that they are within range
        actions = self.scale_actions(actions)

        log('Scaled actions: {}'.format(actions))

        # Set the gain factors
        self.gain_factor_l_hip_y = actions[0]
        self.gain_factor_r_hip_y = actions[1]

        #for t in np.arange(0.0, max_time, dt):

        for i in range(10):

            # Increment the up time variable
            self.up_t += self.lower_control_dt

            # Calculate the current angles of the l and r saggital hip joints
            self.feedback_angles = self.robot_handle.get_angles(['l_hip_y', 'r_hip_y'])

            # Calculate next state of oscillator 1 (pacemaker)
            self.f1_1, self.f2_1 = 0.0, 0.0
            self.s1_1, self.s2_1 = 0.0, 0.0
            self.u1_1, self.u2_1, self.v1_1, self.v2_1, self.y1_1, self.y2_1, self.o_1 = self.oscillator_next(u1=self.u1_1, u2=self.u2_1,
                                                                                                              v1=self.v1_1, v2=self.v2_1,
                                                                                                              y1=self.y1_1, y2=self.y2_1,
                                                                                                              f1=self.f1_1, f2=self.f2_1,
                                                                                                              s1=self.s1_1, s2=self.s2_1,
                                                                                                              bias=self.bias_1, gain=self.gain_1,
                                                                                                              dt=self.lower_control_dt)

            # Calculate next state of oscillator 2
            # w_ij -> j=1 (oscillator 1) is master, i=2 (oscillator 2) is slave
            # Gain factor set by the higher level control is set here
            self.w_21 = 1.0
            self.f1_2, self.f2_2 = self.k * self.feedback_angles['l_hip_y'], -self.k * self.feedback_angles['l_hip_y']
            self.s1_2, self.s2_2 = self.w_21 * self.u1_1, self.w_21 * self.u2_1  # s1_i = w_ij*u1_j, s2_i = w_ij*u2_j
            self.u1_2, self.u2_2, self.v1_2, self.v2_2, self.y1_2, self.y2_2, self.o_2 = self.oscillator_next(u1=self.u1_2, u2=self.u2_2,
                                                                                                              v1=self.v1_2, v2=self.v2_2,
                                                                                                              y1=self.y1_2, y2=self.y2_2,
                                                                                                              f1=self.f1_2, f2=self.f2_2,
                                                                                                              s1=self.s1_2, s2=self.s2_2,
                                                                                                              bias=self.bias_2, gain=self.gain_factor_l_hip_y*self.gain_2,
                                                                                                              dt=self.lower_control_dt)

            # Calculate next state of oscillator 3
            # w_ij -> j=1 (oscillator 1) is master, i=3 (oscillator 3) is slave
            # Gain factor set by the higher level control is set here
            self.w_31 = -1.0
            self.f1_3, self.f2_3 = self.k * self.feedback_angles['r_hip_y'], -self.k * self.feedback_angles['r_hip_y']
            self.s1_3, self.s2_3 = self.w_31 * self.u1_1, self.w_31 * self.u2_1  # s1_i = w_ij*u1_j, s2_i = w_ij*u2_j
            self.u1_3, self.u2_3, self.v1_3, self.v2_3, self.y1_3, self.y2_3, self.o_3 = self.oscillator_next(u1=self.u1_3, u2=self.u2_3,
                                                                                                              v1=self.v1_3, v2=self.v2_3,
                                                                                                              y1=self.y1_3, y2=self.y2_3,
                                                                                                              f1=self.f1_3, f2=self.f2_3,
                                                                                                              s1=self.s1_3, s2=self.s2_3,
                                                                                                              bias=self.bias_3, gain=self.gain_factor_r_hip_y*self.gain_3,
                                                                                                              dt=self.lower_control_dt)

            # Calculate next state of oscillator 4
            # w_ij -> j=2 (oscillator 2) is master, i=4 (oscillator 4) is slave
            self.w_42 = -1.0
            self.f1_4, self.f2_4 = 0.0, 0.0
            self.s1_4, self.s2_4 = self.w_42 * self.u1_2, self.w_42 * self.u2_2  # s1_i = w_ij*u1_j, s2_i = w_ij*u2_j
            self.u1_4, self.u2_4, self.v1_4, self.v2_4, self.y1_4, self.y2_4, self.o_4 = self.oscillator_next(u1=self.u1_4, u2=self.u2_4,
                                                                                                              v1=self.v1_4, v2=self.v2_4,
                                                                                                              y1=self.y1_4, y2=self.y2_4,
                                                                                                              f1=self.f1_4, f2=self.f2_4,
                                                                                                              s1=self.s1_4, s2=self.s2_4,
                                                                                                              bias=self.bias_4, gain=self.gain_4,
                                                                                                              dt=self.lower_control_dt)

            # Calculate next state of oscillator 5
            # w_ij -> j=3 (oscillator 3) is master, i=5 (oscillator 5) is slave
            self.w_53 = -1.0
            self.f1_5, self.f2_5 = 0.0, 0.0
            self.s1_5, self.s2_5 = self.w_53 * self.u1_3, self.w_53 * self.u2_3  # s1_i = w_ij*u1_j, s2_i = w_ij*u2_j
            self.u1_5, self.u2_5, self.v1_5, self.v2_5, self.y1_5, self.y2_5, self.o_5 = self.oscillator_next(u1=self.u1_5, u2=self.u2_5,
                                                                                                              v1=self.v1_5, v2=self.v2_5,
                                                                                                              y1=self.y1_5, y2=self.y2_5,
                                                                                                              f1=self.f1_5, f2=self.f2_5,
                                                                                                              s1=self.s1_5, s2=self.s2_5,
                                                                                                              bias=self.bias_5, gain=self.gain_5,
                                                                                                              dt=self.lower_control_dt)

            # Calculate next state of oscillator 6
            # w_ij -> j=2 (oscillator 2) is master, i=6 (oscillator 6) is slave
            self.w_62 = -1.0
            self.f1_6, self.f2_6 = 0.0, 0.0
            self.s1_6, self.s2_6 = self.w_62 * self.u1_2, self.w_62 * self.u2_2  # s1_i = w_ij*u1_j, s2_i = w_ij*u2_j
            self.u1_6, self.u2_6, self.v1_6, self.v2_6, self.y1_6, self.y2_6, self.o_6 = self.oscillator_next(u1=self.u1_6, u2=self.u2_6,
                                                                                                              v1=self.v1_6, v2=self.v2_6,
                                                                                                              y1=self.y1_6, y2=self.y2_6,
                                                                                                              f1=self.f1_6, f2=self.f2_6,
                                                                                                              s1=self.s1_6, s2=self.s2_6,
                                                                                                              bias=self.bias_6, gain=self.gain_6,
                                                                                                              dt=self.lower_control_dt)

            # Calculate next state of oscillator 7
            # w_ij -> j=3 (oscillator 3) is master, i=7 (oscillator 7) is slave
            self.w_73 = -1.0
            self.f1_7, self.f2_7 = 0.0, 0.0
            self.s1_7, self.s2_7 = self.w_73 * self.u1_3, self.w_73 * self.u2_3  # s1_i = w_ij*u1_j, s2_i = w_ij*u2_j
            self.u1_7, self.u2_7, self.v1_7, self.v2_7, self.y1_7, self.y2_7, self.o_7 = self.oscillator_next(u1=self.u1_7, u2=self.u2_7,
                                                                                                              v1=self.v1_7, v2=self.v2_7,
                                                                                                              y1=self.y1_7, y2=self.y2_7,
                                                                                                              f1=self.f1_7, f2=self.f2_7,
                                                                                                              s1=self.s1_7, s2=self.s2_7,
                                                                                                              bias=self.bias_7, gain=self.gain_7,
                                                                                                              dt=self.lower_control_dt)

            # Calculate next state of oscillator 8
            # w_ij -> j=1 (oscillator 1) is master, i=8 (oscillator 8) is slave
            self.w_81 = 1.0
            self.f1_8, self.f2_8 = 0.0, 0.0
            self.s1_8, self.s2_8 = self.w_81 * self.u1_1, self.w_81 * self.u2_1  # s1_i = w_ij*u1_j, s2_i = w_ij*u2_j
            self.u1_8, self.u2_8, self.v1_8, self.v2_8, self.y1_8, self.y2_8, self.o_8 = self.oscillator_next(u1=self.u1_8, u2=self.u2_8,
                                                                                                              v1=self.v1_8, v2=self.v2_8,
                                                                                                              y1=self.y1_8, y2=self.y2_8,
                                                                                                              f1=self.f1_8, f2=self.f2_8,
                                                                                                              s1=self.s1_8, s2=self.s2_8,
                                                                                                              bias=self.bias_8, gain=self.gain_8,
                                                                                                              dt=self.lower_control_dt)

            # Calculate next state of oscillator 9
            # w_ij -> j=8 (oscillator 8) is master, i=9 (oscillator 9) is slave
            self.w_98 = -1.0
            self.f1_9, self.f2_9 = 0.0, 0.0
            self.s1_9, self.s2_9 = self.w_98 * self.u1_8, self.w_98 * self.u2_8  # s1_i = w_ij*u1_j, s2_i = w_ij*u2_j
            self.u1_9, self.u2_9, self.v1_9, self.v2_9, self.y1_9, self.y2_9, self.o_9 = self.oscillator_next(u1=self.u1_9, u2=self.u2_9,
                                                                                                              v1=self.v1_9, v2=self.v2_9,
                                                                                                              y1=self.y1_9, y2=self.y2_9,
                                                                                                              f1=self.f1_9, f2=self.f2_9,
                                                                                                              s1=self.s1_9, s2=self.s2_9,
                                                                                                              bias=self.bias_9, gain=self.gain_9,
                                                                                                              dt=self.lower_control_dt)

            # Calculate next state of oscillator 10
            # w_ij -> j=1 (oscillator 1) is master, i=10 (oscillator 10) is slave
            self.w_101 = 1.0
            self.f1_10, self.f2_10 = 0.0, 0.0
            self.s1_10, self.s2_10 = self.w_101 * self.u1_1, self.w_101 * self.u2_1  # s1_i = w_ij*u1_j, s2_i = w_ij*u2_j
            self.u1_10, self.u2_10, self.v1_10, self.v2_10, self.y1_10, self.y2_10, self.o_10 = self.oscillator_next(u1=self.u1_10, u2=self.u2_10,
                                                                                                                     v1=self.v1_10, v2=self.v2_10,
                                                                                                                     y1=self.y1_10, y2=self.y2_10,
                                                                                                                     f1=self.f1_10, f2=self.f2_10,
                                                                                                                     s1=self.s1_10, s2=self.s2_10,
                                                                                                                     bias=self.bias_10, gain=self.gain_10,
                                                                                                                     dt=self.lower_control_dt)

            # Calculate next state of oscillator 11
            # w_ij -> j=10 (oscillator 10) is master, i=11 (oscillator 11) is slave
            self.w_1110 = -1.0
            self.f1_11, self.f2_11 = 0.0, 0.0
            self.s1_11, self.s2_11 = self.w_1110 * self.u1_10, self.w_1110 * self.u2_10  # s1_i = w_ij*u1_j, s2_i = w_ij*u2_j
            self.u1_11, self.u2_11, self.v1_11, self.v2_11, self.y1_11, self.y2_11, self.o_11 = self.oscillator_next(u1=self.u1_11, u2=self.u2_11,
                                                                                                                     v1=self.v1_11, v2=self.v2_11,
                                                                                                                     y1=self.y1_11, y2=self.y2_11,
                                                                                                                     f1=self.f1_11, f2=self.f2_11,
                                                                                                                     s1=self.s1_11, s2=self.s2_11,
                                                                                                                     bias=self.bias_11, gain=self.gain_11,
                                                                                                                     dt=self.lower_control_dt)

            # Calculate next state of oscillator 12
            # w_ij -> j=1 (oscillator 1) is master, i=12 (oscillator 12) is slave
            self.w_121 = -1.0
            self.f1_12, self.f2_12 = 0.0, 0.0
            self.s1_12, self.s2_12 = self.w_121 * self.u1_1, self.w_121 * self.u2_1  # s1_i = w_ij*u1_j, s2_i = w_ij*u2_j
            self.u1_12, self.u2_12, self.v1_12, self.v2_12, self.y1_12, self.y2_12, self.o_12 = self.oscillator_next(u1=self.u1_12, u2=self.u2_12,
                                                                                                                     v1=self.v1_12, v2=self.v2_12,
                                                                                                                     y1=self.y1_12, y2=self.y2_12,
                                                                                                                     f1=self.f1_12, f2=self.f2_12,
                                                                                                                     s1=self.s1_12, s2=self.s2_12,
                                                                                                                     bias=self.bias_12, gain=self.gain_12,
                                                                                                                     dt=self.lower_control_dt)

            # Calculate next state of oscillator 13
            # w_ij -> j=1 (oscillator 1) is master, i=13 (oscillator 13) is slave
            self.w_131 = 1.0
            self.f1_13, self.f2_13 = 0.0, 0.0
            self.s1_13, self.s2_13 = self.w_131 * self.u1_1, self.w_131 * self.u2_1  # s1_i = w_ij*u1_j, s2_i = w_ij*u2_j
            self.u1_13, self.u2_13, self.v1_13, self.v2_13, self.y1_13, self.y2_13, self.o_13 = self.oscillator_next(u1=self.u1_13, u2=self.u2_13,
                                                                                                                     v1=self.v1_13, v2=self.v2_13,
                                                                                                                     y1=self.y1_13, y2=self.y2_13,
                                                                                                                     f1=self.f1_13, f2=self.f2_13,
                                                                                                                     s1=self.s1_13, s2=self.s2_13,
                                                                                                                     bias=self.bias_13, gain=self.gain_13,
                                                                                                                     dt=self.lower_control_dt)

            # Set the joint positions
            self.current_angles = {
                'l_hip_y': self.o_2,
                'r_hip_y': self.o_3,
                'l_knee_y': self.o_4,
                'r_knee_y': self.o_5,
                'l_ankle_y': self.o_6,
                'r_ankle_y': self.o_7,
                'l_hip_x': self.o_8,
                'l_ankle_x': self.o_9,
                'r_hip_x': self.o_10,
                'r_ankle_x': self.o_11,
                'l_shoulder_y': self.o_12,
                'r_shoulder_y': self.o_13
            }
            self.robot_handle.set_angles(self.current_angles)

            time.sleep(self.lower_control_dt)

            # Check if the robot has fallen
            if self.monitor_thread.fallen:
                break

        # Make observation
        self._self_observe()

        # Calculate the reward
        # Since the robot starts at x=0,y=0 and faces the x-direction, the reward is calculated as -1.0*abs(y position)
        reward = -1.0 * abs(self.monitor_thread.y)

        # Increment the step counter
        self.step_counter += 1

        # Check if the current state is terminal
        terminal = False
        if self.monitor_thread.fallen or not(self.up_t < self.max_walk_time):
            terminal = True

        # Large negative reward if the robot falls
        if self.monitor_thread.fallen:
            reward -= 100.0

        # No need to put a sleep here, it is taken care of inside the matsuoka loop

        return self.observation, reward, terminal, {}

    def _reset(self):

        log('[ENV] _reset called')

        self.monitor_thread.stop()
        self.matsuoka_init()
        self.gym_init()
        self._self_observe()  # Set the observation
        return self.observation

    def destroy(self):

        log('[ENV] destroy called')

        # Stop the monitoring thread
        self.monitor_thread.stop()

        # Close the VREP connection
        self.robot_handle.cleanup()



