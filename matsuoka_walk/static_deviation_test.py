import os
import numpy as np

from matsuoka_walk import Logger, log
from matsuoka_walk.oscillator_3_test_yaw import oscillator_nw as oscillator_3_test_yaw

# Set the home directory
home_dir = os.path.expanduser('~')

# Set the logging variables
# This also creates a new log file
Logger(log_dir=os.path.join(home_dir, '.bio_walk/logs/'), log_flag=True)

LOWEST_POSSIBLE_GAIN = 0.4

log('[STATIC TEST] LOWEST_POSSIBLE_GAIN: {}'.format(LOWEST_POSSIBLE_GAIN))

wtmpc23_run3_best30 = [0.3178385532762875, 0.3777451259604342, 0.023411599863716586, 0.013217696615302215, 0.4566963469455763, 0.20194162123716233, 0.3309010463046798, -0.05187677829896087, 0.09633745660574622, -0.11559976203529859, 0.4814311312157089, 1.5364038978521224]
asus_run1_bestall = [0.7461913734531209, 0.8422944031253159, 0.07043758116681641, 0.14236621222553963, 0.48893497409925746, 0.5980055418720059, 0.740811806645801, -0.11618361090424223, 0.492832184960149, -0.2949145038394889, 0.175450703085948, -0.3419733470484183]

best_chromosome = asus_run1_bestall

# Testing if the gain change does turn the robot
test_iters = 1
#avg_right = list()
#for i in range(test_iters):
#    # Right leg has smaller stride, go right
#    (x, y) = oscillator_3_test_yaw(best_chromosome, max_time=20.0, gain_l=1.0, gain_r=LOWEST_POSSIBLE_GAIN)
#    avg_right.append(y)
#    log('[STATIC TEST right] iteration: {0} Distance: {1} Deviation: {2}'.format(i, x, y))

avg_straight = list()
for j in range(test_iters):
    # Go straight
    (x, y, gamma) = oscillator_3_test_yaw(best_chromosome, max_time=40.0, gain_l=1.0, gain_r=0.1)
    avg_straight.append(y)
    log('[STATIC TEST straight] iteration: {0} Distance: {1} Deviation: {2} Torso-gamma: {3}'.format(j, x, y, gamma))

#avg_left = list()
#for k in range(test_iters):
#    # Left leg has smaller stride, go left
#    (x, y) = oscillator_3_test_yaw(best_chromosome, max_time=20.0, gain_l=LOWEST_POSSIBLE_GAIN, gain_r=1.0)
#    avg_left.append(y)
#    log('[STATIC TEST left] iteration: {0} Distance: {1} Deviation: {2}'.format(k, x, y))

#log('[Test results] Left: {0}, Straight: {1}, Right: {2}'.format(np.mean(avg_left), np.mean(avg_straight), np.mean(avg_right)))

