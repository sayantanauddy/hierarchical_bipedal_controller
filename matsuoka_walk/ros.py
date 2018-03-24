"""
Module for ROS functionality
"""

from bio_walk import Globals, log
from std_msgs.msg import String
import rospy
import numpy as np
import time

# Force threshold for resetting phase
force_threshold = 10.0

global prev_z_force, curr_z_force
prev_z_force = 0.0
curr_z_force = 0.0


def force_sensor_callback(data):

    global prev_z_force, curr_z_force

    # Retrieve the force plot_data as a string
    str_force_data =  data.data

    # Convert into a numpy 2D array (8 rows, 3 columns)
    # Each row for 1 sensor (l1, l2, l3, l4, r1, r2, r3, r4)
    # Each row has x, y and z force values
    arr_force_data = np.fromstring(str_force_data.replace('{','').replace('}',''), sep=',').reshape((8,3))

    # The left foot sensor values (force)
    l1 = arr_force_data[0] # toe
    l2 = arr_force_data[1] # toe
    l3 = arr_force_data[2] # heel
    l4 = arr_force_data[3] # heel

    # The right foot sensor values (force)
    r1 = arr_force_data[4] # toe
    r2 = arr_force_data[5] # toe
    r3 = arr_force_data[6] # heel
    r4 = arr_force_data[7] # heel

    # Average z forces for left foot
    l_toe_avg_z = (l1[2] + l2[2])/2.0
    l_heel_avg_z = (l3[2] + l4[2]) / 2.0
    l_avg_z = (l1[2] + l2[2] + l3[2] + l4[2])/4.0

    # Average z forces for right foot
    r_toe_avg_z = (r1[2] + r2[2]) / 2.0
    r_heel_avg_z = (r3[2] + r4[2]) / 2.0
    r_avg_z = (r1[2] + r2[2] + r3[2] + r4[2]) / 4.0

    # Reset the phase only when the force goes from below the threshold to above threshold
    curr_z_force = r_heel_avg_z

    if prev_z_force < force_threshold <= curr_z_force:
        # When r_heel_avg_z goes above force_threshold reset the phase
        # Sleep for 0.2s for the change to take effect
        Globals.pr_feedback = -1.0
        log('**** Resetting phase ****')
        log('prev_z_force: {0} curr_z_force: {1}'.format(prev_z_force, curr_z_force))
        time.sleep(0.2)
    else:
        # After the reset, change the feedback back to 0.0
        Globals.pr_feedback = 0.0

    prev_z_force = curr_z_force


def test_listener():

    # In ROS, nodes are uniquely named. If two nodes with the same
    # node are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
    rospy.init_node('listener', anonymous=True)

    rospy.Subscriber("/nico_feet_forces", String, force_sensor_callback)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    test_listener()
