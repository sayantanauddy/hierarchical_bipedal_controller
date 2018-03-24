"""
Stores temporary results from the monitor
"""


class GaitEvalResult:
    """
    Class for storing global gait evaluation related variables
    """
    fallen = True
    up_time = 0.0
    x = 0.0
    y = 0.0
    avg_footstep = 0.0
    var_torso_orientation_alpha = 0.0
    var_torso_orientation_beta = 0.0
    var_torso_orientation_gamma = 0.0

    def __init__(self, log_dir, log_flag=True):
        GaitEvalResult.fallen = True
        GaitEvalResult.up_time = 0.0
        GaitEvalResult.x = 0.0
        GaitEvalResult.y = 0.0
        GaitEvalResult.avg_footstep = 0.0
        GaitEvalResult.var_torso_orientation_alpha = 0.0
        GaitEvalResult.var_torso_orientation_beta = 0.0
        GaitEvalResult.var_torso_orientation_gamma = 0.0

