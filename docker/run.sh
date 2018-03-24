#!/usr/bin/env bash

# Check args
if [ "$#" -ne 1 ]; then
  echo "usage: sudo ./run.sh sayantanauddy/bio_walk:latest"
  return 1
fi

# Get this script's path
pushd `dirname $0` > /dev/null
SCRIPTPATH=`pwd`
popd > /dev/null

set -e

# Run the container with shared X11
docker run\
  --net=host\
  -e SHELL\
  -e DISPLAY\
  -e DOCKER=1\
  -v "$HOME/computing/repositories/MScThesis_SayantanAuddy_2017_NICOOscillatorWalking:/root/computing/repositories/MScThesis_SayantanAuddy_2017_NICOOscillatorWalking:rw"\
  -v "$HOME/computing/simulators/V-REP_PRO_EDU_V3_4_0_Linux:/root/computing/simulators/V-REP_PRO_EDU_V3_4_0_Linux:rw"\
  -v "$HOME/.bio_walk/logs:/root/.bio_walk/logs:rw"\
  -v "/tmp/.X11-unix:/tmp/.X11-unix:rw"\
  -p 80:80\
  -p 19997:19997\
  -p 19998:19998\
  -it $1 $SHELL

