"""
Modified monitor script with force sensing capability
"""

import time
from threading import Thread
from pypot.vrep.io import VrepIO

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

        # Initialize the forces
        self.r1 = 0.0
        self.r2 = 0.0
        self.r3 = 0.0
        self.r4 = 0.0
        self.r_heel_current = 0.0
        self.r_heel_previous = 0.0

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

            if self.r_heel_previous < self.force_threshold <= self.r_heel_current:
                self.phase_reset = True
                print('**** Resetting phase ****')
                print('prev_z_force: {0} curr_z_force: {1}'.format(self.r_heel_previous, self.r_heel_current))
            else:
                self.phase_reset = False

            self.r_heel_previous = self.r_heel_current

            # Sleep
            time.sleep(0.1)

    def stop(self):
        """
        Sets the flag which will stop the thread

        :return: None
        """
        # Flag to stop the monitoring thread
        self.stop_flag = True

        # Close the monitoring vrep connection
        self.vrepio_obj.close()


