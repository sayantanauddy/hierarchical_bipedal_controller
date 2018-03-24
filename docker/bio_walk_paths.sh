#!/usr/bin/env bash

# Set the VREP_ROOT
export VREP_ROOT=$HOME/computing/simulators/V-REP_PRO_EDU_V3_4_0_Linux
# Set BIO_WALK_ROOT
export BIO_WALK_ROOT=$HOME/computing/repositories/MScThesis_SayantanAuddy_2017_NICOOscillatorWalking
export MATSUOKA_WALK_ROOT=$HOME/computing/repositories/MScThesis_SayantanAuddy_2017_NICOOscillatorWalking

# Set the PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$BIO_WALK_ROOT/nicomotion/:$BIO_WALK_ROOT/
