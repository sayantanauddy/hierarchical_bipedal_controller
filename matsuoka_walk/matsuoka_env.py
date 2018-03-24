import numpy as np
import gym
from gym import spaces
import time

from matsuoka_walk.log import log
from matsuoka_walk.oscillator_3_thread import Oscillator3Thread

# Factors which decide the weightage of deviation and forward distance on the reward calculation
# The reward is calculated as: reward = ALPHA*(-abs(deviation)) + BETA*forward_distance + THETA*orientation

ALPHA = 1.0
BETA = 0.3
THETA = 1.0

class MatsuokaEnv(gym.Env):

    def __init__(self):

        log('[ENV] Initilializing environment')

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

        # Initilize the observation variables
        self.observation = None

        # Variable for the number of steps in the episode
        self.step_counter = 0

        # Set the oscillator thread
        # This will initialize the VREP scene and set the robot to the starting position
        self.oscillator_thread = Oscillator3Thread()

        # Start the oscillator thread
        # This will start the matsuoka walk
        self.oscillator_thread.start()

    def _self_observe(self):

        self.observation = self.oscillator_thread.self_observe()

    def _step(self, actions):

        # Set the actions
        self.oscillator_thread.self_action(actions)

        # Time for the actions to take effect
        time.sleep(1.0)

        # Make observation
        self._self_observe()

        # Calculate the reward
        # Since the robot starts at x=0,y=0 and faces the x-direction, the reward is calculated as -1.0*abs(y position)
        objpos = self.oscillator_thread.monitor_thread.objpos
        fallen = self.oscillator_thread.monitor_thread.fallen
        torso_orientation = self.oscillator_thread.monitor_thread.torso_euler_angles
        forward_x = objpos[0]
        deviation = objpos[1]
        torso_gamma = torso_orientation[2]

        reward = ALPHA*(-abs(deviation)) + BETA*forward_x + THETA*(-abs(torso_gamma))

        log('[ENV] Deviation: {0} X-distance: {1} Torso-gamma: {2}'.format(deviation, forward_x, torso_gamma))

        # Increment the step counter
        self.step_counter += 1

        # Large negative reward if the robot falls
        if fallen:
            reward -= 100.0

        return self.observation, reward, self.oscillator_thread.terminal, {}

    def _reset(self):

        log('[ENV] Resetting the environment')

        self.oscillator_thread.monitor_thread.stop()
        self.oscillator_thread.monitor_thread.join()
        # Close the VREP connection
        self.oscillator_thread.robot_handle.cleanup()
        self.oscillator_thread.stop()
        self.oscillator_thread.join()
        self.__init__()
        self._self_observe()  # Set the observation
        return self.observation

    def _close(self):
        log('[ENV] Stopping the environment')
        self.oscillator_thread.monitor_thread.stop()
        self.oscillator_thread.monitor_thread.join()
        # Close the VREP connection
        self.oscillator_thread.robot_handle.cleanup()
        self.oscillator_thread.stop()
        self.oscillator_thread.join()
