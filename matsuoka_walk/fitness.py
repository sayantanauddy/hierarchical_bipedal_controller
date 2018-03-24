import numpy as np
from matsuoka_walk.log import log


def hart6sc(position_vector, **kwargs):
    """
    Implementation of the Hartmann 6-dimensional function (rescaled)
    Adapted from Matlab implementation at https://www.sfu.ca/~ssurjano/Code/hart6scm.html
    Details and optimum value can be found at https://www.sfu.ca/~ssurjano/hart6.html

    :param position_vector: A 6D position vector
    :return: fitness_score
    """

    alpha = np.array([[1.0], [1.2], [3.0], [3.2]])
    A = np.array([[10.0, 3.0, 17.0, 3.50, 1.7, 8.0],
                  [0.05, 10.0, 17.0, 0.1, 8.0, 14.0],
                  [3.0, 3.5, 1.7, 10.0, 17.0, 8.0],
                  [17.0, 8.0, 0.05, 10.0, 0.1, 14.0]])

    P = 10**(-4) * np.array([[1312, 1696, 5569, 124, 8283, 5886],
                              [2329, 4135, 8307, 3736, 1004, 9991],
                              [2348, 1451, 3522, 2883, 3047, 6650],
                              [4047, 8828, 8732, 5743, 1091, 381]])

    outer = 0
    for ii in range(4):
        inner = 0
        for jj in range(6):
            xj = position_vector[jj]
            Aij = A[ii][jj]
            Pij = P[ii][jj]
            inner = inner + Aij * (xj - Pij)**2
        new = alpha[ii] * np.exp(-inner)
        outer = outer + new

    return -outer


def rastigrin(position_vector):

    dim = len(position_vector)

    return 10.0*dim + sum([(xi**2 - 10.0*np.cos(2*np.pi*xi)) for xi in position_vector])

def calc_fitness(start_x, start_y, start_z,
                 end_x, end_y, end_z,
                 avg_z,
                 up_time,
                 fitness_option
                 ):

    log('[FIT] Executing calc_fitness with fitness_option: {0}'.format(fitness_option))
    log('[FIT] Other arguments:')
    log('[FIT] start_x={0}, start_y={1}, start_z={2}'.format(start_x, start_y, start_z))
    log('[FIT] end_x={0}, end_y={1}, end_z={2}'.format(end_x, end_y, end_z))
    log('[FIT] avg_z={0}'.format(avg_z))
    log('[FIT] up_time={0}'.format(up_time))

    x_distance = end_x - start_x
    y_distance = end_y - start_y

    x_vel = x_distance/up_time  # (metres/second)

    fitness = 0.0

    if fitness_option==1:
        # Fitness is the distance moved in the x direction (metres)
        fitness = x_distance
        log('[FIT] fitness = x_distance')

    elif fitness_option==2:
        # Fitness is the distance moved in the x direction (metres) + up_time (minutes)
        fitness = x_distance + up_time/60.0
        log('[FIT] fitness = x_distance + up_time/60.0')

    elif fitness_option == 3:
        # Fitness is the distance moved in the x direction * 0.3 (metres) + up_time (seconds) * 0.7
        # This formula yielded the stable walk in open loop
        fitness = x_distance*0.3 + up_time*0.7
        log('[FIT] fitness = x_distance*0.3 + up_time*0.7')

    elif fitness_option==4:
        # Fitness is the straight line velocity minus a penalty for deviating from the straight line
        # Follows the formula in Cristiano's paper (coefficient values are aplha=80, gamma=100)
        fitness = 80.0*x_vel - 100*abs(y_distance)
        log('[FIT] fitness = 80.0*x_vel - 100*abs(y_distance)')

    elif fitness_option==5:
        # Fitness is the sum of x-distance(m), time(minutes), x-vel(m/s)
        fitness = x_distance + (up_time/60.0) + x_vel
        log('[FIT] fitness = x_distance + (up_time/60.0) + x_vel')

    elif fitness_option == 6:
        # Fitness is the distance moved in the x direction (metres) + up_time (seconds)*0.5
        # This formula yielded the stable walk in open loop
        fitness = x_distance + up_time * 0.5
        log('[FIT] fitness = x_distance + up_time*0.5')

    return fitness
