"""
Module for wrappers of robot specific classes
"""

from abc import ABCMeta, abstractmethod, abstractproperty
from nicomotion import Motion
import math
import pypot
from pypot.vrep import from_vrep
from pypot.creatures import PoppyHumanoid
import time
import numpy as np
from pypot.utils.stoppablethread import StoppableLoopThread
from pypot.dynamixel.motor import DxlMXMotor


class Robot:
    """
    Abstract class for robot specific functions
    """

    # This class cannot be instantiated but must be inherited by a class that provides implementation of the methods
    # and values for the properties
    __metaclass__ = ABCMeta

    def __init__(self):
        """
        The constructor of the abstract class
        """
        pass

    @abstractproperty
    def sync_sleep_time(self):
        """
        Time to sleep to allow the joints to reach their targets
        """
        pass

    @abstractproperty
    def robot_handle(self):
        """
        Stores the handle to the robot
        This handle is used to invoke methods on the robot
        """
        pass

    @abstractproperty
    def interpolation(self):
        """
        Flag to indicate if intermediate joint angles should be interpolated
        """
        pass

    @abstractproperty
    def fraction_max_speed(self):
        """
        Fraction of the maximum motor speed to use
        """
        pass

    @abstractproperty
    def wait(self):
        """
        Flag to indicate whether the control should wait for each angle to reach its target
        """
        pass

    @abstractmethod
    def set_angles(self, joint_angles, duration=None, joint_velocities=None):
        """
        Sets the joints to the specified angles

        :type joint_angles: dict
        :param joint_angles: Dictionary of joint_names: angles (in radians)
        :type duration: float
        :param duration: Time to reach the angular targets (in seconds)
        :type joint_velocities: dict
        :param joint_velocities: dict of joint angles and velocities
        :return: None
        """
        pass

    @abstractmethod
    def get_angles(self, joint_names):
        """
        Gets the angles of the specified joints and returns a dict of joint_names: angles (in radians)

        :type joint_names: list(str)
        :param joint_names: List of joint names
        :rtype: dict
        """
        pass


class Nico(Robot):
    """
    This class encapsulates the methods and properties for interacting with the nao robot
    It extends the abstract class 'Robot'
    """

    sync_sleep_time = None
    robot_handle = None
    interpolation = None
    fraction_max_speed = None
    wait = None

    def __init__(self, sync_sleep_time, interpolation=False, fraction_max_speed=0.01, wait=False,
                 motor_config='config.json', vrep=False, vrep_host='127.0.0.1', vrep_port=19997, vrep_scene=None):

        """
        The constructor of the class. Class properties should be set here
        The robot handle should be created here
        Any other initializations such as setting angles to particular values should also be taken care of here

        :type sync_sleep_time: float
        :param sync_sleep_time: Time to sleep to allow the joints to reach their targets (in seconds)
        :type interpolation: bool
        :param interpolation: Flag to indicate if intermediate joint angles should be interpolated
        :type fraction_max_speed: float
        :param fraction_max_speed: Fraction of the maximum motor speed to use
        :type wait: bool
        :param wait: Flag to indicate whether the control should wait for each angle to reach its target
        :type motor_config: str
        :param motor_config: json configuration file
        :type vrep: bool
        :param vrep: Flag to indicate if VREP is to be used
        :type vrep_host: str
        :param vrep_host: IP address of VREP server
        :type vrep_port: int
        :param vrep_port: Port of VREP server
        :type vrep_scene: str
        :param vrep_scene: VREP scene to load
        """

        super(Nico, self).__init__()

        # Set the properties
        self.sync_sleep_time = sync_sleep_time
        self.interpolation = interpolation
        self.fraction_max_speed = fraction_max_speed
        self.wait = wait
        self.motor_config = motor_config
        self.vrep = vrep
        self.vrep_host = vrep_host
        self.vrep_port = vrep_port
        self.vrep_scene = vrep_scene


        # Create the robot handle
        self.robot_handle = Motion.Motion(self.motor_config, self.vrep, self.vrep_host, self.vrep_port, self.vrep_scene)

        # List of all joint names
        self.all_joint_names = self.robot_handle.getJointNames()

        # Initialize the joints
        # for joint_name in self.all_joint_names:
        #   self.set_angles({joint_name:0.0})

        # Sleep for a few seconds to allow the changes to take effect
        time.sleep(3)

    def set_angles_slow(self, target_angles, duration, step=0.01):
        """
        Sets the angles over the specified duration using linear interpolation

        :param target_angles:
        :param duration:
        :param step:
        :return:
        """

        # Retrieve the start angles
        start_angles = self.get_angles(joint_names=target_angles.keys())

        # Calculate the slope for each joint
        angle_slopes = dict()
        for joint_name in target_angles.keys():
            start = start_angles[joint_name]
            end = target_angles[joint_name]
            angle_slopes[joint_name] = (end - start)/duration

        # t starts from 0.0 and goes till duration
        for t in np.arange(0.0, duration+0.01, step):
            current_angles = dict()
            # Calculate the value of each joint angle at time t
            for joint_name in target_angles.keys():
                current_angles[joint_name] = start_angles[joint_name] + angle_slopes[joint_name]*t

            # Set the current angles
            self.set_angles(current_angles)

            # Sleep for the step time
            time.sleep(step)

    def set_angles(self, joint_angles, duration=None, joint_velocities=None):
        """
        Sets the joints to the specified angles (after converting radians to degrees since the poppy robot uses degrees)

        :type joint_angles: dict
        :param joint_angles: Dictionary of joint_names: angles (in radians)
        :type duration: float
        :param duration: Time to reach the angular targets (in seconds)
        :type joint_velocities: dict
        :param joint_velocities: dict of joint angles and velocities
        :return: None
        """

        l_knee_max = 90.0
        l_knee_min = 0.0
        r_kee_max = 90.0
        r_knee_min = 0.0

        for joint_name in joint_angles.keys():

            # Convert the angle to degrees
            target_angle_degrees = math.degrees(joint_angles[joint_name])

            if joint_name == 'l_knee_y':
                if target_angle_degrees >= l_knee_max:
                    target_angle_degrees = l_knee_max
                elif target_angle_degrees <= l_knee_min:
                    target_angle_degrees = l_knee_min
                else:
                    target_angle_degrees = target_angle_degrees

            if joint_name == 'r_knee_y':
                if target_angle_degrees >= r_kee_max:
                    target_angle_degrees = r_kee_max
                elif target_angle_degrees <= r_knee_min:
                    target_angle_degrees = r_knee_min
                else:
                    target_angle_degrees = target_angle_degrees

            self.robot_handle.setAngle(joint_name, target_angle_degrees, self.fraction_max_speed)

        # Sleep to allow the motors to reach their targets
        if duration is not None:
            time.sleep(self.sync_sleep_time)

    def get_angles(self, joint_names=None):
        """
        Gets the angles of the specified joints and returns a dict of joint_names: angles (in radians)
        If joint_names=None then the values of all joints are returned

        :type joint_names: list(str)
        :param joint_names: List of joint names
        :rtype: dict
        """

        # Create the dict to be returned
        joint_angles = dict()

        # If joint names are not provided, get values of all joints
        if joint_names is None:
            # Call the nicomotion api function to get list of joint names
            joint_names = self.all_joint_names

        motors = self.robot_handle._robot.motors

        # If no joint names are specified, return angles of all joints in raidans
        # Else return only the angles (in radians) of the specified joints
        for m in motors:
            if joint_names is None:
                joint_angles[str(m.name)] = math.radians(m.present_position)
            else:
                if m.name in joint_names:
                    joint_angles[str(m.name)] = math.radians(m.present_position)

        return joint_angles

    def cleanup(self):
        """
        Cleans up the current connection to the robot
        :return: None
        """
        self.robot_handle.cleanup()


class Poppy(Robot):
    """
    This class encapsulates the methods and properties for interacting with the poppy robot
    It extends the abstract class 'Robot'
    """

    sync_sleep_time = None
    robot_handle = None
    interpolation = None
    fraction_max_speed = None
    wait = None

    def __init__(self, sync_sleep_time, interpolation=False, fraction_max_speed=0.01, wait=False,
                 motor_config=None, vrep=False, vrep_host='127.0.0.1', vrep_port=19997, vrep_scene=None):
        """
        The constructor of the class. Class properties should be set here
        The robot handle should be created here
        Any other initializations such as setting angles to particular values should also be taken care of here

        :type sync_sleep_time: float
        :param sync_sleep_time: Time to sleep to allow the joints to reach their targets (in seconds)
        :type interpolation: bool
        :param interpolation: Flag to indicate if intermediate joint angles should be interpolated
        :type fractionMaxSpeed: float
        :param fractionMaxSpeed: Fraction of the maximum motor speed to use
        :type wait: bool
        :param wait: Flag to indicate whether the control should wait for each angle to reach its target
        """

        super(Poppy, self).__init__()

        # Set the properties
        self.sync_sleep_time = sync_sleep_time
        self.interpolation = interpolation
        self.fraction_max_speed = fraction_max_speed
        self.wait = wait

        self._maximumSpeed = 1.0

        # Close existing vrep connections if any
        pypot.vrep.close_all_connections()

        # Create a new poppy robot and set the robot handle
        self.robot_handle = PoppyHumanoid(simulator='vrep',
                                          config=motor_config,
                                          host=vrep_host,
                                          port=vrep_port,
                                          scene=vrep_scene)

        # Sync the robot joints
        self.robot_handle.start_sync()

        # Perform required joint initializations
        # Move arms to pi/2
        # self.robot_handle.l_shoulder_y.goal_position = -90
        # self.robot_handle.r_shoulder_y.goal_position = -90

        # Sleep for a few seconds to allow the changes to take effect
        time.sleep(3)

    def set_angles(self, joint_angles, duration=None, joint_velocities=None):
        """
        Sets the joints to the specified angles (after converting radians to degrees since the poppy robot uses degrees)

        :type joint_angles: dict
        :param joint_angles: Dictionary of joint_names: angles (in radians)
        :type duration: float
        :param duration: Time to reach the angular targets (in seconds)
        :type joint_velocities: dict
        :param joint_velocities: dict of joint angles and velocities
        :return: None
        """

        for joint_name in joint_angles.keys():

            # Convert the angle to degrees
            target_angle_degrees = math.degrees(joint_angles[joint_name])

            try:
                # Determine the right joint and set the joint angle
                for motor in self.robot_handle.motors:
                    if motor.name == joint_name:
                        motor.compliant = False
                        motor.goal_speed = 1000.0 * min(self.fraction_max_speed, self._maximumSpeed)
                        motor.goal_position = target_angle_degrees
                        break
            except Exception as e:  # Catch all exceptions
                print e.message
                raise RuntimeError('Could not set joint angle')

        # Sleep to allow the motors to reach their targets
        if not duration:
            time.sleep(self.sync_sleep_time)

    def get_angles(self, joint_names=None):
        """
        Gets the angles of the specified joints and returns a dict of joint_names: angles (in radians)
        If joint_names=None then the values of all joints are returned

        :type joint_names: list
        :param joint_names: List of joint name strings
        :rtype: dict
        :returns: dict of joint names and angles
        """

        # Create the dict to be returned
        joint_angles = dict()

        # Retrieve the list of DxlMXMotor objects
        motors = self.robot_handle.motors

        # If no joint names are specified, return angles of all joints in raidans
        # Else return only the angles (in radians) of the specified joints
        for m in motors:
            if joint_names is None:
                joint_angles[str(m.name)] = math.radians(m.present_position)
            else:
                if m.name in joint_names:
                    joint_angles[str(m.name)] = math.radians(m.present_position)

        return joint_angles

    def cleanup(self):
        """
        Cleans up the current connection to the robot
        :return: None
        """
        # TODO check if it works
        self.robot_handle.close()

