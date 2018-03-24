"""
Modified monitor script for evaluating gaits
"""

import time
from threading import Thread
from pypot.vrep.io import VrepIO
import numpy as np
import os

from matsuoka_walk.log import log
from matsuoka_walk.gait_eval_result import GaitEvalResult

# import rospy
# from std_msgs.msg import String
# from bio_walk.ros import force_sensor_callback


class RobotMonitorThread(Thread):
    """
    Class for monitoring objects in VREP simulator.
    This class extends the Thread class and will run as in parallel to the main simulation.
    Only one monitoring thread is enough for monitoring a simulation.
    """

    def __init__(self, portnum, objname, height_threshold, force_threshold=10):
        """
        Initializes the RobotMonitorThread thread

        :param portnum: Port number on which the VREP remote API is listening on the server
        (different than the one used to run the main robot simulation)
        :param objname: Object name which is to be monitored
        :height_threshold: If the object's height is lower than this value, the robot is considered to have falen
        """

        # Call init of super class
        Thread.__init__(self)

        # Setup the log
        home_dir = os.path.expanduser('~')
        log('[MON] Monitor started')

        # Create a VrepIO object using a different port number than the one used to run the main robot simulation
        # Make sure that the VREP remote api on the server is listening on this port number
        # Additional ports for listening can be set up on the server side by editing the file remoteApiConnections.txt
        self.vrepio_obj = VrepIO(vrep_port=portnum)

        # The object to be monitored
        self.objname = objname

        # Threshold for falling
        self.height_threshold = height_threshold

        # Threshold for force sensor
        self.force_threshold = force_threshold

        # The position of the object
        self.objpos = None

        # Initialize the foot sensor forces
        self.l1 = 0.0
        self.l2 = 0.0
        self.l3 = 0.0
        self.l4 = 0.0
        self.l_heel_current = 0.0
        self.l_heel_previous = 0.0
        
        self.r1 = 0.0
        self.r2 = 0.0
        self.r3 = 0.0
        self.r4 = 0.0
        self.r_heel_current = 0.0
        self.r_heel_previous = 0.0

        # Lists to hold the foot positions as ('foot type',x,y) tuples
        self.foot_position = list()

        # Variable to store average foot step
        self.avg_footstep = 0.0

        # Flag for phase reset
        self.phase_reset = False


        # The x,y,z coordinates of the position
        self.x = None
        self.y = None
        self.z = None

        # The starting time
        self.start_time = time.time()
        self.up_time = 0.0

        # The average height of the robot
        self.avg_z = 0.0

        # A list to store the height at each second
        self.z_list = list()

        # A flag which can stop the thread
        self.stop_flag = False

        # Lists to store torso orientations
        self.torso_orientation_alpha = list()
        self.torso_orientation_beta = list()
        self.torso_orientation_gamma = list()

        # Variables to store the torso orientation variance
        self.var_torso_orientation_alpha = 0.0
        self.var_torso_orientation_beta = 0.0
        self.var_torso_orientation_gamma = 0.0

        # A flag to indicate if the robot has fallen
        self.fallen = False

        # Set up the ROS listener
        # rospy.init_node('nico_feet_force_listener', anonymous=True)
        # rospy.Subscriber("/nico_feet_forces", String, force_sensor_callback)

    def reset_timer(self):
        self.start_time = time.time()

    def run(self):
        """
        The monitoring logic is implemented here. The self.objpos is updated at a preset frequency with the latest
        object positions.

        :return: None
        """

        while not self.stop_flag:

            # Update the current position
            self.objpos = self.vrepio_obj.get_object_position(self.objname)
            self.x = self.objpos[0]
            self.y = self.objpos[1]
            self.z = self.objpos[2]

            # Update the time
            self.up_time = time.time() - self.start_time

            # Append the current height to the height list
            self.z_list.append(self.z)

            if self.z < self.height_threshold:
                # Set the flag which indicates that the robot has fallen
                self.fallen = True
                # Calculate the average height
                self.avg_z = sum(self.z_list) / float(len(self.z_list))

            # Update force sensor readings (first index signifies force vector and second index signifies z axis force)
            self.r1 = self.vrepio_obj.call_remote_api('simxReadForceSensor', self.vrepio_obj.get_object_handle('right_sensor_1'), streaming=True)[1][2]
            self.r2 = self.vrepio_obj.call_remote_api('simxReadForceSensor', self.vrepio_obj.get_object_handle('right_sensor_2'), streaming=True)[1][2]
            self.r3 = self.vrepio_obj.call_remote_api('simxReadForceSensor', self.vrepio_obj.get_object_handle('right_sensor_3'), streaming=True)[1][2]
            self.r4 = self.vrepio_obj.call_remote_api('simxReadForceSensor', self.vrepio_obj.get_object_handle('right_sensor_4'), streaming=True)[1][2]

            # Average the forces on the right heel
            self.r_heel_current = (self.r3 + self.r4)/2.0

            self.l1 = self.vrepio_obj.call_remote_api('simxReadForceSensor', self.vrepio_obj.get_object_handle('left_sensor_1'), streaming=True)[1][2]
            self.l2 = self.vrepio_obj.call_remote_api('simxReadForceSensor', self.vrepio_obj.get_object_handle('left_sensor_2'), streaming=True)[1][2]
            self.l3 = self.vrepio_obj.call_remote_api('simxReadForceSensor', self.vrepio_obj.get_object_handle('left_sensor_3'), streaming=True)[1][2]
            self.l4 = self.vrepio_obj.call_remote_api('simxReadForceSensor', self.vrepio_obj.get_object_handle('left_sensor_4'), streaming=True)[1][2]

            # Average the forces on the right heel
            self.l_heel_current = (self.l3 + self.l4) / 2.0

            # Store the foot position if left foot strikes ground
            if self.l_heel_previous < self.force_threshold <= self.l_heel_current:
                # Get the position of the left foot
                left_foot_position = self.vrepio_obj.get_object_position('left_foot_11_respondable')
                left_foot_position_x = left_foot_position[0]
                left_foot_position_y = left_foot_position[1]
                self.foot_position.append(('l', left_foot_position_x, left_foot_position_y))
                print('Left foot prev_z_force: {0} curr_z_force: {1}'.format(self.l_heel_previous, self.l_heel_current))

            # Store the foot position and do phase reset (if needed) if the right foot strikes the ground
            if self.r_heel_previous < self.force_threshold <= self.r_heel_current:
                # Get the position of the left foot
                right_foot_position = self.vrepio_obj.get_object_position('right_foot_11_respondable')
                right_foot_position_x = right_foot_position[0]
                right_foot_position_y = right_foot_position[1]
                self.foot_position.append(('r', right_foot_position_x, right_foot_position_y))

                # Perform phase reset
                self.phase_reset = True
                print('**** Resetting phase ****')
                print('prev_z_force: {0} curr_z_force: {1}'.format(self.r_heel_previous, self.r_heel_current))
            else:
                self.phase_reset = False

            self.l_heel_previous = self.l_heel_current
            self.r_heel_previous = self.r_heel_current

            # Store the current torso orientation
            self.torso_orientation_alpha.append(self.vrepio_obj.get_object_orientation(self.objname)[0])
            self.torso_orientation_beta.append(self.vrepio_obj.get_object_orientation(self.objname)[1])
            self.torso_orientation_gamma.append(self.vrepio_obj.get_object_orientation(self.objname)[2])

            # Sleep
            time.sleep(0.1)


    def calc_var_orientation(self):
        """
        Function to compute the variation in the torso orientations
        :return: None
        """
        self.var_torso_orientation_alpha = np.var(self.torso_orientation_alpha)
        self.var_torso_orientation_beta = np.var(self.torso_orientation_beta)
        self.var_torso_orientation_gamma = np.var(self.torso_orientation_gamma)


    def calc_avg_footstep(self):
        # Inspect the foot position list and filter out only alternating types of positions (l,r,l,r,...)
        # If there are multiple positions of same type (l,l,l,r,l,r,r,r,l,r,l,l,...) consider only the last position of
        # each type (l1, l2, l3, r1, r2, l4, r3, l5, l6, r3, r4 -> l3, r2, l4, r3, l6, r4)

        # A list to store the indices of alternating foot positions
        alternating_indices = list()
        alternating_indices.append(0)

        # Find the indices of alternating foot positions
        for idx in range(len(self.foot_position)):
            foot_pos = self.foot_position[idx]
            last_type = self.foot_position[alternating_indices[-1]][0]
            curr_type = foot_pos[0]
            if last_type == curr_type:
                continue
            else:
                alternating_indices.append(idx)

        # This list contains foot positions of alternating feet
        alternating_foot_positions = [self.foot_position[i] for i in alternating_indices]

        # Calculate the difference between the alternating foot positions (x-axis)
        # Each tuple in the list is of the form ('l/r', x-pos, y-pos)
        foot_step_sizes = [t[1] - s[1] for s, t in zip(alternating_foot_positions, alternating_foot_positions[1:])]

        # Calculate the average foot step length in the x-direction
        self.avg_footstep = np.mean([foot for foot in foot_step_sizes])

    def stop(self):
        """
        Sets the flag which will stop the thread

        :return: None
        """
        # Flag to stop the monitoring thread
        self.stop_flag = True

        # Close the monitoring vrep connection
        self.vrepio_obj.close()

        # Calculate the average foot step size
        self.calc_avg_footstep()

        # Calculate the variance of the torso orientation
        self.calc_var_orientation()

        # Store the results
        GaitEvalResult.fallen = self.fallen
        GaitEvalResult.up_time = self.up_time
        GaitEvalResult.x = self.x
        GaitEvalResult.y = abs(self.y)
        GaitEvalResult.avg_footstep = self.avg_footstep
        GaitEvalResult.var_torso_orientation_alpha = self.var_torso_orientation_alpha
        GaitEvalResult.var_torso_orientation_beta = self.var_torso_orientation_beta
        GaitEvalResult.var_torso_orientation_gamma = self.var_torso_orientation_gamma

        print 'Fall:{0}, ' \
              'Up_time:{1}, ' \
              'X-distance:{2}, ' \
              'Deviation (mag):{3}, ' \
              'Avg. step length:{4}, ' \
              'Torso orientation variance Alpha-Beta-Gamma:{5}|{6}|{7}'.format(self.fallen,
                                                                               self.up_time,
                                                                               self.x,
                                                                               abs(self.y),
                                                                               self.avg_footstep,
                                                                               self.var_torso_orientation_alpha,
                                                                               self.var_torso_orientation_beta,
                                                                               self.var_torso_orientation_gamma)

        log('[MON] Monitor finished')
