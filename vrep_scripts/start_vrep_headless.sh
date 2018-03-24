#!/usr/bin/env bash

LC_NUMERIC=en_US.UTF-8 xvfb-run --auto-servernum --server-num=1 ./vrep.sh -h

