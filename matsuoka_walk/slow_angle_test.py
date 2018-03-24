import os
from matsuoka_walk.robots import Nico

home_dir = os.path.expanduser('~')

robot_handle = Nico(sync_sleep_time=0.1,
                    motor_config=os.path.join(home_dir,'computing/repositories/hierarchical_bipedal_controller/motor_configs/nico_humanoid_full_v1.json'),
                    vrep=True,
                    vrep_host='127.0.0.1',
                    vrep_port=19997,
                    vrep_scene=os.path.join(home_dir,'computing/repositories/hierarchical_bipedal_controller/vrep_scenes/NICO-Simplified-July2017_standing.ttt')
                    )

target_angles = {'l_shoulder_y': -0.6, 'r_shoulder_y':0.6}

robot_handle.set_angles_slow(target_angles, 5.0)