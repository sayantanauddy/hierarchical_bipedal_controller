#!/usr/bin/env bash
# This script should be run inside the container

# Set the VREP_ROOT
export VREP_ROOT=$HOME/computing/simulators/V-REP_PRO_EDU_V3_4_0_Linux

# Set BIO_WALK_ROOT
export BIO_WALK_ROOT=$HOME/computing/repositories/MScThesis_SayantanAuddy_2017_NICOOscillatorWalking

# Copy VREP remote API script and start script to VREP_ROOT and set permissions
cp -f $BIO_WALK_ROOT/vrep_scripts/remoteApiConnections.txt $VREP_ROOT/remoteApiConnections.txt
cp -f $BIO_WALK_ROOT/vrep_scripts/start_vrep_headless.sh $VREP_ROOT/start_vrep_headless.sh
chmod a+x $VREP_ROOT/start_vrep_headless.sh

source /opt/ros/indigo/setup.bash
mkdir -p $HOME/catkin_ws/src
cd $HOME/catkin_ws/
catkin build
source $HOME/catkin_ws/devel/setup.bash

# Copy VREP ROS packages to catkin_ws
cp -r $VREP_ROOT/programming/ros_packages/vrep_skeleton_msg_and_srv ~/catkin_ws/src/
cp -r $VREP_ROOT/programming/ros_packages/vrep_plugin_skeleton ~/catkin_ws/src/
cp -r $VREP_ROOT/programming/ros_packages/v_repExtRosInterface ~/catkin_ws/src/
cp -r $VREP_ROOT/programming/ros_packages/ros_bubble_rob2 ~/catkin_ws/src/

#Inside catkin_ws
cd $HOME/catkin_ws/
catkin build
source /opt/ros/indigo/setup.bash
source $HOME/catkin_ws/devel/setup.bash

# Copy the built libraries to VREP_ROOT
cp $HOME/catkin_ws/devel/lib/*.so $VREP_ROOT

# Not needed since log folder is mounted from host system
# Create the log folder
# mkdir -p $HOME/.bio_walk/logs








