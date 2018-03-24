
from pypot.vrep.io import VrepIO
from pypot import vrep
import time
import os

from matsuoka_walk.robots import Nico

home_dir = os.path.expanduser('~')

vrep.close_all_connections()

robot_handle = Nico(sync_sleep_time=0.1,
                    motor_config=os.path.join(home_dir,'computing/repositories/MScThesis_SayantanAuddy_2017_NICOOscillatorWalking/motor_configs/nico_humanoid_full_v1.json'),
                    vrep=True,
                    vrep_host='127.0.0.1',
                    vrep_port=19997,
                    vrep_scene=os.path.join(home_dir,'computing/repositories/MScThesis_SayantanAuddy_2017_NICOOscillatorWalking/vrep_scenes/NICO-Simplified-July2017_standing.ttt')
                    )

angles={'l_shoulder_y': 0.5,
        'r_shoulder_y': 0.5,
        'l_hip_y': -0.3,
        'r_hip_y': -0.3,
        'l_knee_y': 0.3,
        'r_knee_y': 0.3,
        'l_ankle_y': -0.1,
        'r_ankle_y': -0.1}
        
robot_handle.set_angles(angles)

time.sleep(5.0)


