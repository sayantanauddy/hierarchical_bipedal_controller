"""
Script for evaluating the best gait of each type
"""

import os

# Evaluation script for open loop
from oscillator_2_1_eval import oscillator_nw as oscillator_nw_2_1_eval

# Evaluation script for angle feedback
from oscillator_3_eval import oscillator_nw as oscillator_nw_3_eval

# Evaluation script for phase reset using ROS - comment out on wtmpc
from oscillator_5_eval import oscillator_nw as oscillator_nw_5_eval

# Evaluation script for phase reset (without ROS)
from oscillator_5_1_eval import oscillator_nw as oscillator_nw_5_1_eval

from matsuoka_walk import Logger, log

# Set the home directory
home_dir = os.path.expanduser('~')

# Set the logging variables
# This also creates a new log file
Logger(log_dir=os.path.join(home_dir, '.bio_walk/logs/'), log_flag=True)

# How many times to run each gait
max_evals = 100

# How many seconds to walk for
max_duration = 20

# Best chromosomes from wtmpc
wtmpc19_1_best_30 = [0.4487933856539151, 0.41673786908555105, 0.16921354649496317, 0.1860945200781693, 0.6940925174226963, 0.13721648459341165, 0.5818299000288453, -0.16336638678846577, 0.03559516754585279, -0.0147476314724036, 0.01752125444461805]
wtmpc19_1_best_all = [0.4487933856539151, 0.41673786908555105, 0.16921354649496317, 0.1860945200781693, 0.6940925174226963, 0.13721648459341165, 0.5818299000288453, -0.16336638678846577, 0.03559516754585279, -0.0147476314724036, 0.01752125444461805]
wtmpc19_2_best_30 = [0.8144226546318917, 0.7825319796509879, 0.33724011804980913, 0.04886135999886435, 0.6607028866518464, 0.063576825517388, 0.7722552063140246, -0.06659985941535823, 0.30096540191619725, -0.18954930948731608, 0.17735923522104446]
wtmpc19_2_best_all = [0.7969907783221327, 0.7825319796509879, 0.33724011804980913, 0.04886135999886435, 0.7842127577436201, 0.063576825517388, 0.3520099015771251, -0.11670682052423148, 0.2908941617877242, -0.17331908048444455, 0.4086031808441024]
wtmpc19_3_best_30 = [0.446961477181909, 0.6212894131379891, 0.08659780896097678, 0.2070618622562773, 0.7098933998856228, 0.08183000949704482, 0.4865584914422241, -0.17138807926155258, 0.498298846173112, -0.2999437617339941, 0.05749558089524498]
wtmpc19_3_best_all = [0.446961477181909, 0.6212894131379891, 0.08659780896097678, 0.2182480160503531, 0.6988563563590054, 0.08183000949704482, 0.4865584914422241, -0.17138807926155258, 0.498298846173112, -0.2999437617339941, 0.05749558089524498]

wtmpc23_1_best_30 = [0.22582153337762675, 0.5177942185061809, 0.010089761067497238, 0.024517096299628394, 0.4554896626042894, 0.21456587690541598, 0.8918417089438845, -0.15377489976823494, 0.1840621172904282, -0.09727783983604749, 0.010672701006650187, 0.7654131913731539]
wtmpc23_1_best_all = [0.22582153337762675, 0.5177942185061809, 0.010089761067497238, 0.024517096299628394, 0.4565716992558727, 0.21456587690541598, 0.04382822590153418, -0.15377489976823494, 0.1840621172904282, -0.09727783983604749, 0.010672701006650187, 0.7654131913731539]
wtmpc23_2_best_30 = [0.7594003349774361, 0.7708360572478096, 0.08126819962010298, 0.13810885008202248, 0.8222665034505168, 0.02232419404554537, 0.23087525761744018, 0.004576161363345821, 0.08950875610916226, -0.12170022091568816, 0.580570312855465, -0.6129795682547934]
wtmpc23_2_best_all = [0.282089138603028, 0.46334754751752255, 0.018172007674646398, 0.13154986822143147, 0.8956079605368018, 0.22806833680724717, 0.7315744344798211, -0.10306235056915714, 0.34016326831458443, -0.23603252167036476, 0.5687796973564235, 0.5357567195357866]
wtmpc23_3_best_30 = [0.3178385532762875, 0.3777451259604342, 0.023411599863716586, 0.013217696615302215, 0.4566963469455763, 0.20194162123716233, 0.3309010463046798, -0.05187677829896087, 0.09633745660574622, -0.11559976203529859, 0.4814311312157089, 1.5364038978521224]
wtmpc23_3_best_all = [0.30571692865079375, 0.3777451259604342, 0.023411599863716586, 0.013217696615302215, 0.4566963469455763, 0.20194162123716233, 0.643899252518004, -0.08606917370687261, 0.09633745660574622, -0.11559976203529859, 0.4055943477648639, 1.5364038978521224]

wtmpc29_1_best_30 = [0.31579287729984173, 0.6792201907039106, 0.01339778570475527, 0.1477847854805867, 0.4343399413602635, 0.20603829891054057, 0.3597789915592806, -0.13316722855344842, 0.4286740668862156, -0.27801721520551, 0.5847175013549969]
wtmpc29_1_best_all = [0.31579287729984173, 0.5766644050690783, 0.01339778570475527, 0.161794676988227, 0.4343399413602635, 0.20603829891054057, 0.5872236249936905, -0.13316722855344842, 0.3883485718217937, -0.26744236314967945, 0.5847175013549969]
wtmpc29_2_best_30 = [0.28402030795019234, 0.48459484266628056, 0.0469284266586924, 0.02358686049082416, 0.3094144948184207, 0.29360548728816166, 0.0883389870392757, -0.2374508759380406, 0.2508914187307585, -0.14452791424897765, 0.2630386607292723]
wtmpc29_2_best_all = [0.28402030795019234, 0.48459484266628056, 0.0469284266586924, 0.02358686049082416, 0.3094144948184207, 0.29360548728816166, 0.0883389870392757, -0.2374508759380406, 0.2508914187307585, -0.14452791424897765, 0.2630386607292723]
wtmpc29_3_best_30 = [0.24956179605092835, 0.5626372944168968, 0.0174425253314795, 0.014047631285274429, 0.3939484784487431, 0.1844854836208218, 0.9518459829099569, -0.3391738445533632, 0.2575584894549689, -0.10450486683754667, 0.07747260694170832]
wtmpc29_3_best_all = [0.23867777304358195, 0.5626372944168968, 0.0174425253314795, 0.014047631285274429, 0.3939484784487431, 0.1844854836208218, 0.9518459829099569, -0.3391738445533632, 0.1926321520617318, -0.06618960375589938, 0.07747260694170832]

# Best chromosomes for asus
asus_open_loop_run_1_best_30 = [0.5884145884963005, 0.5369511850955603, 0.09131337379133, 0.025252652932915343, 0.3493674052847861, 0.15003845298477264, 0.9377263350996107, -0.3099780890687065, 0.28272635930326756, -0.08259435820020733, 0.22178117552852405]
asus_open_loop_run_1_best_all = [0.5884145884963005, 0.5369511850955603, 0.09131337379133, 0.025252652932915343, 0.3493674052847861, 0.15003845298477264, 0.9377263350996107, -0.3099780890687065, 0.28272635930326756, -0.08259435820020733, 0.22178117552852405]
asus_open_loop_run_2_best_30 = [0.5485784640852793, 0.7970330080026813, 0.038904963880512115, 0.05880702056644107, 0.4610837528910333, 0.3213298732897521, 0.37805877648537417, -0.3094133662873301, 0.2507644239781829, -0.07882557480485719, 0.23676855420461274]
asus_open_loop_run_2_best_all = [0.5485784640852793, 0.7970330080026813, 0.059994036626502364, 0.05880702056644107, 0.4610837528910333, 0.3213298732897521, 0.3264717708845675, -0.4915001197486241, 0.46941851897635467, -0.1513092589135323, 0.23676855420461274]
asus_open_loop_run_3_best_30 = [0.5428521073752984, 0.9371143081664542, 0.04423326455980758, 0.34041014910663603, 0.37865377373344594, 0.19996597492328877, 0.2637554661113346, -0.5351864525917331, 0.4412125822742417, -0.08198294127207817, 0.37568547782029527]
asus_open_loop_run_3_best_all = [0.5429865398935334, 0.9371143081664542, 0.04423326455980758, 0.34041014910663603, 0.37865377373344594, 0.7100302532164159, 0.16001821062067942, -0.5351864525917331, 0.4412125822742417, -0.1322334265170068, 0.37568547782029527]

asus_angle_feedback_run_1_best_30 = [0.7461913734531209, 0.8422944031253159, 0.048102556534769705, 0.13210092576102664, 0.691052361025347, 0.5980055418720059, 0.20172707438248702, -0.11618361090424223, 0.48936183550354523, -0.3117228103439414, 0.5584075581302852, -0.3419733470484183]
asus_angle_feedback_run_1_best_all = [0.7461913734531209, 0.8422944031253159, 0.07043758116681641, 0.14236621222553963, 0.48893497409925746, 0.5980055418720059, 0.740811806645801, -0.11618361090424223, 0.492832184960149, -0.2949145038394889, 0.175450703085948, -0.3419733470484183]
asus_angle_feedback_run_2_best_30 = [0.6450404705811301, 0.7693348395420826, 0.019270973206137753, 0.09411469737444257, 0.49928395481568916, 0.3631655506638598, 0.4917977003145482, -0.333420483018255, 0.46330629494181336, -0.21254793707792774, 0.01824073139337634, -0.17826436059862738]
asus_angle_feedback_run_2_best_all = [0.6450404705811301, 0.7693348395420826, 0.019270973206137753, 0.09411469737444257, 0.49928395481568916, 0.3631655506638598, 0.4917977003145482, -0.333420483018255, 0.46330629494181336, -0.21254793707792774, 0.01824073139337634, -0.17826436059862738]
asus_angle_feedback_run_3_best_30 = [0.664180548936441, 0.525749776983526, 0.0830684196800701, 0.011232309309081695, 0.5196008058922972, 0.7227556069113118, 0.8231251238297248, -0.05749839126674595, 0.20238791074823237, -0.14516829348465654, 0.3215756067598835, -0.7915781389907256]
asus_angle_feedback_run_3_best_all = [0.6784372407889947, 0.5148872951061086, 0.0830684196800701, 0.011232309309081695, 0.5196008058922972, 0.7227556069113118, 0.5293133854016969, -0.06388015409865555, 0.20238791074823237, -0.15175969109697796, 0.3215756067598835, -0.783817540693736]

asus_phase_reset_run_1_best_30 = [0.5146344913630376, 0.7455092402318735, 0.02567339112892708, 0.11122944383094965, 0.45329367561406486, 0.36757287747454864, 0.21053517408377884, -0.3906102732254958, 0.48870071798756914, -0.21330757509005024, 0.11514607350697748]
asus_phase_reset_run_1_best_all = [0.5146344913630376, 0.7455092402318735, -0.0070835220642929495, 0.13405758327500758, 0.6090967083734895, 0.6937935725821373, 0.21053517408377884, -0.3906102732254958, 0.4640728729649564, -0.23084688159919053, 0.7871274895040696]
asus_phase_reset_run_2_best_30 = [0.37798675709152035, 0.8206433683497132, 0.14367821852476442, 0.08404019333004167, 0.3680045188456352, 0.7905291463890434, 0.43844355218981856, -0.35615830232990553, 0.35777347846378393, -0.1088510720271551, 0.24326367020390438]
asus_phase_reset_run_2_best_all = [0.8597258814566455, 0.43299754018667147, 0.14367821852476442, 0.08404019333004167, 0.7634723943690848, 0.39147229949169776, 0.11048570137787744, -0.1408267335975779, 0.08740573084913256, -0.06380256717714489, 0.08800174238458613]
asus_phase_reset_run_3_best_30 = [0.4562623173475021, 0.8786594024400013, 0.019510798669051966, 0.2469325597410287, 0.3530342608249799, 0.6166506101319219, 0.5989007717556899, -0.11838113488614932, 0.48787029062295906, -0.2603644240173303, 0.06758791216505156]
asus_phase_reset_run_3_best_all = [0.4562623173475021, 0.8070710919522855, 0.019510798669051966, 0.2469325597410287, 0.6143840864164016, 0.7639003163395752, 0.5989007717556899, -0.11838113488614932, 0.4929240505481482, -0.27814734264111757, 0.3798933719519648]

def gait_eval(position_vector, description, serial, oscillator_nw, max_evals=max_evals, max_duration=max_duration):
    for i in range(max_evals):
        result = oscillator_nw(position_vector, max_time=max_duration)
        log('[EVAL] Description: {0}, Serial#: {1}, Run#: {2}, Result: << {3}, {4}, {5}, {6}, {7}, {8}, {9}, {10}, {11} >>'
            .format(description,
                    serial,
                    i + 1,
                    result['fitness'],
                    result['fallen'],
                    result['up'],
                    result['x_distance'],
                    result['abs_y_deviation'],
                    result['avg_footstep_x'],
                    result['var_torso_alpha'],
                    result['var_torso_beta'],
                    result['var_torso_gamma']))
    log('#################################################')


# # Evaluate wtmpc open loop
# gait_eval(position_vector=wtmpc19_1_best_30, description='wtmpc19 open loop 30', serial=1, oscillator_nw=oscillator_nw_2_1_eval, max_evals=max_evals, max_duration=max_duration)
# gait_eval(position_vector=wtmpc19_1_best_all, description='wtmpc19 open loop all', serial=1, oscillator_nw=oscillator_nw_2_1_eval, max_evals=max_evals, max_duration=max_duration)
# gait_eval(position_vector=wtmpc19_2_best_30, description='wtmpc19 open loop 30', serial=2, oscillator_nw=oscillator_nw_2_1_eval, max_evals=max_evals, max_duration=max_duration)
# gait_eval(position_vector=wtmpc19_2_best_all, description='wtmpc19 open loop all', serial=2, oscillator_nw=oscillator_nw_2_1_eval, max_evals=max_evals, max_duration=max_duration)
# gait_eval(position_vector=wtmpc19_3_best_30, description='wtmpc19 open loop 30', serial=3, oscillator_nw=oscillator_nw_2_1_eval, max_evals=max_evals, max_duration=max_duration)
# gait_eval(position_vector=wtmpc19_3_best_all, description='wtmpc19 open loop all', serial=3, oscillator_nw=oscillator_nw_2_1_eval, max_evals=max_evals, max_duration=max_duration)
#
# # Evaluate wtmpc angle feedback
# gait_eval(position_vector=wtmpc23_1_best_30, description='wtmpc23 angle feedback 30', serial=1, oscillator_nw=oscillator_nw_3_eval, max_evals=max_evals, max_duration=max_duration)
# gait_eval(position_vector=wtmpc23_1_best_all, description='wtmpc23 angle feedback all', serial=1, oscillator_nw=oscillator_nw_3_eval, max_evals=max_evals, max_duration=max_duration)
# gait_eval(position_vector=wtmpc23_2_best_30, description='wtmpc23 angle feedback 30', serial=2, oscillator_nw=oscillator_nw_3_eval, max_evals=max_evals, max_duration=max_duration)
# gait_eval(position_vector=wtmpc23_2_best_all, description='wtmpc23 angle feedback all', serial=2, oscillator_nw=oscillator_nw_3_eval, max_evals=max_evals, max_duration=max_duration)
# gait_eval(position_vector=wtmpc23_3_best_30, description='wtmpc23 angle feedback 30', serial=3, oscillator_nw=oscillator_nw_3_eval, max_evals=max_evals, max_duration=max_duration)
# gait_eval(position_vector=wtmpc23_3_best_all, description='wtmpc23 angle feedback all', serial=3, oscillator_nw=oscillator_nw_3_eval, max_evals=max_evals, max_duration=max_duration)
#
# # Evaluate wtmpc phase reset
# gait_eval(position_vector=wtmpc29_1_best_30, description='wtmpc29 phase reset 30', serial=1, oscillator_nw=oscillator_nw_5_1_eval, max_evals=max_evals, max_duration=max_duration)
# gait_eval(position_vector=wtmpc29_1_best_all, description='wtmpc29 phase reset all', serial=1, oscillator_nw=oscillator_nw_5_1_eval, max_evals=max_evals, max_duration=max_duration)
# gait_eval(position_vector=wtmpc29_2_best_30, description='wtmpc29 phase reset 30', serial=2, oscillator_nw=oscillator_nw_5_1_eval, max_evals=max_evals, max_duration=max_duration)
# gait_eval(position_vector=wtmpc29_2_best_all, description='wtmpc29 phase reset all', serial=2, oscillator_nw=oscillator_nw_5_1_eval, max_evals=max_evals, max_duration=max_duration)
# gait_eval(position_vector=wtmpc29_3_best_30, description='wtmpc29 phase reset 30', serial=3, oscillator_nw=oscillator_nw_5_1_eval, max_evals=max_evals, max_duration=max_duration)
# gait_eval(position_vector=wtmpc29_3_best_all, description='wtmpc29 phase reset all', serial=3, oscillator_nw=oscillator_nw_5_1_eval, max_evals=max_evals, max_duration=max_duration)


# Evaluate asus open loop
gait_eval(position_vector=asus_open_loop_run_1_best_30, description='asus open loop 30', serial=1, oscillator_nw=oscillator_nw_2_1_eval, max_evals=max_evals, max_duration=max_duration)
gait_eval(position_vector=asus_open_loop_run_1_best_all, description='asus open loop all', serial=1, oscillator_nw=oscillator_nw_2_1_eval, max_evals=max_evals, max_duration=max_duration)
gait_eval(position_vector=asus_open_loop_run_2_best_30, description='asus open loop 30', serial=2, oscillator_nw=oscillator_nw_2_1_eval, max_evals=max_evals, max_duration=max_duration)
gait_eval(position_vector=asus_open_loop_run_2_best_all, description='asus open loop all', serial=2, oscillator_nw=oscillator_nw_2_1_eval, max_evals=max_evals, max_duration=max_duration)
gait_eval(position_vector=asus_open_loop_run_3_best_30, description='asus open loop 30', serial=3, oscillator_nw=oscillator_nw_2_1_eval, max_evals=max_evals, max_duration=max_duration)
gait_eval(position_vector=asus_open_loop_run_3_best_all, description='asus open loop all', serial=3, oscillator_nw=oscillator_nw_2_1_eval, max_evals=max_evals, max_duration=max_duration)

# Evaluate asus angle feedback
gait_eval(position_vector=asus_angle_feedback_run_1_best_30, description='asus angle feedback 30', serial=1, oscillator_nw=oscillator_nw_3_eval, max_evals=max_evals, max_duration=max_duration)
gait_eval(position_vector=asus_angle_feedback_run_1_best_all, description='asus angle feedback all', serial=1, oscillator_nw=oscillator_nw_3_eval, max_evals=max_evals, max_duration=max_duration)
gait_eval(position_vector=asus_angle_feedback_run_2_best_30, description='asus angle feedback 30', serial=2, oscillator_nw=oscillator_nw_3_eval, max_evals=max_evals, max_duration=max_duration)
gait_eval(position_vector=asus_angle_feedback_run_2_best_all, description='asus angle feedback all', serial=2, oscillator_nw=oscillator_nw_3_eval, max_evals=max_evals, max_duration=max_duration)
gait_eval(position_vector=asus_angle_feedback_run_3_best_30, description='asus angle feedback 30', serial=3, oscillator_nw=oscillator_nw_3_eval, max_evals=max_evals, max_duration=max_duration)
gait_eval(position_vector=asus_angle_feedback_run_3_best_all, description='asus angle feedback all', serial=3, oscillator_nw=oscillator_nw_3_eval, max_evals=max_evals, max_duration=max_duration)

# Evaluate asus phase reset
# Run roscore before running this
gait_eval(position_vector=asus_phase_reset_run_1_best_30, description='asus phase reset 30', serial=1, oscillator_nw=oscillator_nw_5_eval, max_evals=max_evals, max_duration=max_duration)
gait_eval(position_vector=asus_phase_reset_run_1_best_all, description='asus phase reset all', serial=1, oscillator_nw=oscillator_nw_5_eval, max_evals=max_evals, max_duration=max_duration)
gait_eval(position_vector=asus_phase_reset_run_2_best_30, description='asus phase reset 30', serial=2, oscillator_nw=oscillator_nw_5_eval, max_evals=max_evals, max_duration=max_duration)
gait_eval(position_vector=asus_phase_reset_run_2_best_all, description='asus phase reset all', serial=2, oscillator_nw=oscillator_nw_5_eval, max_evals=max_evals, max_duration=max_duration)
gait_eval(position_vector=asus_phase_reset_run_3_best_30, description='asus phase reset 30', serial=3, oscillator_nw=oscillator_nw_5_eval, max_evals=max_evals, max_duration=max_duration)
gait_eval(position_vector=asus_phase_reset_run_3_best_all, description='asus phase reset all', serial=3, oscillator_nw=oscillator_nw_5_eval, max_evals=max_evals, max_duration=max_duration)

