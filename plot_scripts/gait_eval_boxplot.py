import csv as csv
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import os

# Set the font size to be used in the plots
matplotlib.rcParams.update({'font.size': 8})

# Set the home directory
home_dir = os.path.expanduser('~')

# Set the directory for saving plots
plot_dir = os.path.join(home_dir, 'computing/repositories/MScThesis_SayantanAuddy_2017_NICOOscillatorWalking/plots')

# Directory for saving plot data files
plot_data_dir = os.path.join(home_dir, 'computing/repositories/MScThesis_SayantanAuddy_2017_NICOOscillatorWalking/plot_data')

# Set the directory where logs are stored
log_dir = os.path.join(home_dir, 'computing/repositories/MScThesis_SayantanAuddy_2017_NICOOscillatorWalking/logs/gait_evaluation_logs')


# Function to create a dataframe from the gait evaluation log file
def create_log_df(search_string, log_file_path):

    # Indices with numeric plot_data
    idx = [14, 16, 17, 18, 19, 20 ,21, 22]

    # Column names for the above indices
    cols = ['fitness', 'up_time', 'x_dist', 'abs_y_dev', 'avg_footstep', 'var_alpha', 'var_beta', 'var_gamma']

    file_list = list()

    with open(log_file_path) as f:
        for line in f:
            if search_string in line:
                file_list.append([float(line.split(' ')[i].replace(',','')) for i in idx])

    df = pd.DataFrame(np.array(file_list).reshape(100,8), columns = cols)

    return df

# The gait evaluation log files (wtmpc)
wtmpc_open_loop_eval_log_path = os.path.join(log_dir, 'log_20171010_184331.txt')
wtmpc_angle_feedback_eval_log_path = os.path.join(log_dir, 'log_20171010_184741.txt')
wtmpc_phase_reset_eval_log_path = os.path.join(log_dir, 'log_20171010_185025.txt')

# The gait evaluation log file (asus)
asus_gait_eval_log_path = os.path.join(log_dir, 'log_20171205_000836.txt')

# Create dataframes for each run (wtmpc)
wtmpc_df_open_loop_run_1_best30 = create_log_df('wtmpc19 open loop 30, Serial#: 1', wtmpc_open_loop_eval_log_path)
wtmpc_df_open_loop_run_1_bestall = create_log_df('wtmpc19 open loop all, Serial#: 1', wtmpc_open_loop_eval_log_path)
wtmpc_df_open_loop_run_2_best30 = create_log_df('wtmpc19 open loop 30, Serial#: 2', wtmpc_open_loop_eval_log_path)
wtmpc_df_open_loop_run_2_bestall = create_log_df('wtmpc19 open loop all, Serial#: 2', wtmpc_open_loop_eval_log_path)
wtmpc_df_open_loop_run_3_best30 = create_log_df('wtmpc19 open loop 30, Serial#: 3', wtmpc_open_loop_eval_log_path)
wtmpc_df_open_loop_run_3_bestall = create_log_df('wtmpc19 open loop all, Serial#: 3', wtmpc_open_loop_eval_log_path)

wtmpc_df_angle_feedback_run_1_best30 = create_log_df('wtmpc23 angle feedback 30, Serial#: 1', wtmpc_angle_feedback_eval_log_path)
wtmpc_df_angle_feedback_run_1_bestall = create_log_df('wtmpc23 angle feedback all, Serial#: 1', wtmpc_angle_feedback_eval_log_path)
wtmpc_df_angle_feedback_run_2_best30 = create_log_df('wtmpc23 angle feedback 30, Serial#: 2', wtmpc_angle_feedback_eval_log_path)
wtmpc_df_angle_feedback_run_2_bestall = create_log_df('wtmpc23 angle feedback all, Serial#: 2', wtmpc_angle_feedback_eval_log_path)
wtmpc_df_angle_feedback_run_3_best30 = create_log_df('wtmpc23 angle feedback 30, Serial#: 3', wtmpc_angle_feedback_eval_log_path)
wtmpc_df_angle_feedback_run_3_bestall = create_log_df('wtmpc23 angle feedback all, Serial#: 3', wtmpc_angle_feedback_eval_log_path)

wtmpc_df_phase_reset_run_1_best30 = create_log_df('wtmpc29 phase reset 30, Serial#: 1', wtmpc_phase_reset_eval_log_path)
wtmpc_df_phase_reset_run_1_bestall = create_log_df('wtmpc29 phase reset all, Serial#: 1', wtmpc_phase_reset_eval_log_path)
wtmpc_df_phase_reset_run_2_best30 = create_log_df('wtmpc29 phase reset 30, Serial#: 2', wtmpc_phase_reset_eval_log_path)
wtmpc_df_phase_reset_run_2_bestall = create_log_df('wtmpc29 phase reset all, Serial#: 2', wtmpc_phase_reset_eval_log_path)
wtmpc_df_phase_reset_run_3_best30 = create_log_df('wtmpc29 phase reset 30, Serial#: 3', wtmpc_phase_reset_eval_log_path)
wtmpc_df_phase_reset_run_3_bestall = create_log_df('wtmpc29 phase reset all, Serial#: 3', wtmpc_phase_reset_eval_log_path)

# Create dataframes for each run (wtmpc)
asus_df_open_loop_run_1_best30 = create_log_df('asus open loop 30, Serial#: 1', asus_gait_eval_log_path)
asus_df_open_loop_run_1_bestall = create_log_df('asus open loop all, Serial#: 1', asus_gait_eval_log_path)
asus_df_open_loop_run_2_best30 = create_log_df('asus open loop 30, Serial#: 2', asus_gait_eval_log_path)
asus_df_open_loop_run_2_bestall = create_log_df('asus open loop all, Serial#: 2', asus_gait_eval_log_path)
asus_df_open_loop_run_3_best30 = create_log_df('asus open loop 30, Serial#: 3', asus_gait_eval_log_path)
asus_df_open_loop_run_3_bestall = create_log_df('asus open loop all, Serial#: 3', asus_gait_eval_log_path)

asus_df_angle_feedback_run_1_best30 = create_log_df('asus angle feedback 30, Serial#: 1', asus_gait_eval_log_path)
asus_df_angle_feedback_run_1_bestall = create_log_df('asus angle feedback all, Serial#: 1', asus_gait_eval_log_path)
asus_df_angle_feedback_run_2_best30 = create_log_df('asus angle feedback 30, Serial#: 2', asus_gait_eval_log_path)
asus_df_angle_feedback_run_2_bestall = create_log_df('asus angle feedback all, Serial#: 2', asus_gait_eval_log_path)
asus_df_angle_feedback_run_3_best30 = create_log_df('asus angle feedback 30, Serial#: 3', asus_gait_eval_log_path)
asus_df_angle_feedback_run_3_bestall = create_log_df('asus angle feedback all, Serial#: 3', asus_gait_eval_log_path)

asus_df_phase_reset_run_1_best30 = create_log_df('asus phase reset 30, Serial#: 1', asus_gait_eval_log_path)
asus_df_phase_reset_run_1_bestall = create_log_df('asus phase reset all, Serial#: 1', asus_gait_eval_log_path)
asus_df_phase_reset_run_2_best30 = create_log_df('asus phase reset 30, Serial#: 2', asus_gait_eval_log_path)
asus_df_phase_reset_run_2_bestall = create_log_df('asus phase reset all, Serial#: 2', asus_gait_eval_log_path)
asus_df_phase_reset_run_3_best30 = create_log_df('asus phase reset 30, Serial#: 3', asus_gait_eval_log_path)
asus_df_phase_reset_run_3_bestall = create_log_df('asus phase reset all, Serial#: 3', asus_gait_eval_log_path)

# Function for concatenating columns from the log dataframes
# All columns of the same metric from different dataframes are concatenated (column-wise) to form a metric dataframe
def create_metric_df(metric_name):
    
    metric_cols = ['open_loop_run_1_best30', #'open_loop_run_1_bestall',
                   'open_loop_run_2_best30', #'open_loop_run_2_bestall',
                   'open_loop_run_3_best30', #'open_loop_run_3_bestall',
                   'angle_feedback_run_1_best30', #'angle_feedback_run_1_bestall',
                   'angle_feedback_run_2_best30', #'angle_feedback_run_2_bestall',
                   'angle_feedback_run_3_best30', #'angle_feedback_run_3_bestall',
                   'phase_reset_run_1_best30', #'phase_reset_run_1_bestall',
                   'phase_reset_run_2_best30', #'phase_reset_run_2_bestall',
                   'phase_reset_run_3_best30', #'phase_reset_run_3_bestall'
                   ]

    df_metric_wtmpc = pd.concat([wtmpc_df_open_loop_run_1_best30[metric_name],
                                 #wtmpc_df_open_loop_run_1_bestall[metric_name],
                                 wtmpc_df_open_loop_run_2_best30[metric_name],
                                 #wtmpc_df_open_loop_run_2_bestall[metric_name],
                                 wtmpc_df_open_loop_run_3_best30[metric_name],
                                 #wtmpc_df_open_loop_run_3_bestall[metric_name],
                                 wtmpc_df_angle_feedback_run_1_best30[metric_name],
                                 #wtmpc_df_angle_feedback_run_1_bestall[metric_name],
                                 wtmpc_df_angle_feedback_run_2_best30[metric_name],
                                 #wtmpc_df_angle_feedback_run_2_bestall[metric_name],
                                 wtmpc_df_angle_feedback_run_3_best30[metric_name],
                                 #wtmpc_df_angle_feedback_run_3_bestall[metric_name],
                                 wtmpc_df_phase_reset_run_1_best30[metric_name],
                                 #wtmpc_df_phase_reset_run_1_bestall[metric_name],
                                 wtmpc_df_phase_reset_run_2_best30[metric_name],
                                 #wtmpc_df_phase_reset_run_2_bestall[metric_name],
                                 wtmpc_df_phase_reset_run_3_best30[metric_name],
                                 #wtmpc_df_phase_reset_run_3_bestall[metric_name]
                                 ],
                                axis=1)

    df_metric_asus = pd.concat([asus_df_open_loop_run_1_best30[metric_name],
                                #asus_df_open_loop_run_1_bestall[metric_name],
                                asus_df_open_loop_run_2_best30[metric_name],
                                #asus_df_open_loop_run_2_bestall[metric_name],
                                asus_df_open_loop_run_3_best30[metric_name],
                                #asus_df_open_loop_run_3_bestall[metric_name],
                                asus_df_angle_feedback_run_1_best30[metric_name],
                                #asus_df_angle_feedback_run_1_bestall[metric_name],
                                asus_df_angle_feedback_run_2_best30[metric_name],
                                #asus_df_angle_feedback_run_2_bestall[metric_name],
                                asus_df_angle_feedback_run_3_best30[metric_name],
                                #asus_df_angle_feedback_run_3_bestall[metric_name],
                                asus_df_phase_reset_run_1_best30[metric_name],
                                #asus_df_phase_reset_run_1_bestall[metric_name],
                                asus_df_phase_reset_run_2_best30[metric_name],
                                #asus_df_phase_reset_run_2_bestall[metric_name],
                                asus_df_phase_reset_run_3_best30[metric_name],
                                #asus_df_phase_reset_run_3_bestall[metric_name]
                                ],
                               axis=1)
    
    df_metric_wtmpc.columns = metric_cols
    df_metric_asus.columns = metric_cols

    return df_metric_wtmpc, df_metric_asus

# Create individual dataframes for each metric and save the data in csv files
df_fitness_wtmpc, df_fitness_asus = create_metric_df('fitness')
df_fitness_wtmpc.to_csv(os.path.join(plot_data_dir, 'wtmpc_gait_eval_fitness.csv'))
df_fitness_asus.to_csv(os.path.join(plot_data_dir, 'asus_gait_eval_fitness.csv'))

df_up_time_wtmpc, df_up_time_asus = create_metric_df('up_time')
df_up_time_wtmpc.to_csv(os.path.join(plot_data_dir, 'wtmpc_gait_eval_up_time.csv'))
df_up_time_asus.to_csv(os.path.join(plot_data_dir, 'asus_gait_eval_up_time.csv'))

df_x_dist_wtmpc, df_x_dist_asus = create_metric_df('x_dist')
df_x_dist_wtmpc.to_csv(os.path.join(plot_data_dir, 'wtmpc_gait_eval_x_dist.csv'))
df_x_dist_asus.to_csv(os.path.join(plot_data_dir, 'asus_gait_eval_x_dist.csv'))

df_abs_y_dev_wtmpc, df_abs_y_dev_asus = create_metric_df('abs_y_dev')
df_abs_y_dev_wtmpc.to_csv(os.path.join(plot_data_dir, 'wtmpc_gait_eval_abs_y_dev.csv'))
df_abs_y_dev_asus.to_csv(os.path.join(plot_data_dir, 'asus_gait_eval_abs_y_dev.csv'))

df_avg_footstep_wtmpc, df_avg_footstep_asus = create_metric_df('avg_footstep')
df_avg_footstep_wtmpc.to_csv(os.path.join(plot_data_dir, 'wtmpc_gait_eval_avg_footstep.csv'))
df_avg_footstep_asus.to_csv(os.path.join(plot_data_dir, 'asus_gait_eval_avg_footstep.csv'))

df_var_alpha_wtmpc, df_var_alpha_asus = create_metric_df('var_alpha')
df_var_alpha_wtmpc.to_csv(os.path.join(plot_data_dir, 'wtmpc_gait_eval_var_alpha.csv'))
df_var_alpha_asus.to_csv(os.path.join(plot_data_dir, 'asus_gait_eval_var_alpha.csv'))

df_var_beta_wtmpc, df_var_beta_asus = create_metric_df('var_beta')
df_var_beta_wtmpc.to_csv(os.path.join(plot_data_dir, 'wtmpc_gait_eval_var_beta.csv'))
df_var_beta_asus.to_csv(os.path.join(plot_data_dir, 'asus_gait_eval_var_beta.csv'))

df_var_gamma_wtmpc, df_var_gamma_asus = create_metric_df('var_gamma')
df_var_gamma_wtmpc.to_csv(os.path.join(plot_data_dir, 'wtmpc_gait_eval_var_gamma.csv'))
df_var_gamma_asus.to_csv(os.path.join(plot_data_dir, 'asus_gait_eval_var_gamma.csv'))

# Create and save the box plots
plt.figure(11)
df_fitness_wtmpc.boxplot(vert=False, showmeans=True,patch_artist=False)
plt.subplots_adjust(left=0.3)
plt.savefig(os.path.join(plot_dir, 'wtmpc_gait_eval_fitness.png'))

plt.figure(12)
df_fitness_asus.boxplot(vert=False, showmeans=True,patch_artist=False)
plt.subplots_adjust(left=0.3)
plt.savefig(os.path.join(plot_dir, 'asus_gait_eval_fitness.png'))

plt.figure(21)
df_up_time_wtmpc.boxplot(vert=False, showmeans=True,patch_artist=False)
plt.subplots_adjust(left=0.3)
plt.savefig(os.path.join(plot_dir, 'wtmpc_gait_eval_up_time.png'))

plt.figure(22)
df_up_time_asus.boxplot(vert=False, showmeans=True,patch_artist=False)
plt.subplots_adjust(left=0.3)
plt.savefig(os.path.join(plot_dir, 'asus_gait_eval_up_time.png'))

plt.figure(31)
df_x_dist_wtmpc.boxplot(vert=False, showmeans=True,patch_artist=False)
plt.subplots_adjust(left=0.3)
plt.savefig(os.path.join(plot_dir, 'wtmpc_gait_eval_x_dist.png'))

plt.figure(32)
df_x_dist_asus.boxplot(vert=False, showmeans=True,patch_artist=False)
plt.subplots_adjust(left=0.3)
plt.savefig(os.path.join(plot_dir, 'asus_gait_eval_x_dist.png'))

plt.figure(41)
df_abs_y_dev_wtmpc.boxplot(vert=False, showmeans=True,patch_artist=False)
plt.subplots_adjust(left=0.3)
plt.savefig(os.path.join(plot_dir, 'wtmpc_gait_eval_abs_y_dev.png'))

plt.figure(42)
df_abs_y_dev_asus.boxplot(vert=False, showmeans=True,patch_artist=False)
plt.subplots_adjust(left=0.3)
plt.savefig(os.path.join(plot_dir, 'asus_gait_eval_abs_y_dev.png'))

plt.figure(51)
df_avg_footstep_wtmpc.boxplot(vert=False, showmeans=True,patch_artist=False)
plt.subplots_adjust(left=0.3)
plt.savefig(os.path.join(plot_dir, 'wtmpc_gait_eval_avg_footstep.png'))

plt.figure(52)
df_avg_footstep_asus.boxplot(vert=False, showmeans=True,patch_artist=False)
plt.subplots_adjust(left=0.3)
plt.savefig(os.path.join(plot_dir, 'asus_gait_eval_avg_footstep.png'))

plt.figure(61)
df_var_alpha_wtmpc.boxplot(vert=False, showmeans=True,patch_artist=False)
plt.subplots_adjust(left=0.3)
plt.savefig(os.path.join(plot_dir, 'wtmpc_gait_eval_var_alpha.png'))

plt.figure(62)
df_var_alpha_asus.boxplot(vert=False, showmeans=True,patch_artist=False)
plt.subplots_adjust(left=0.3)
plt.savefig(os.path.join(plot_dir, 'asus_gait_eval_var_alpha.png'))

plt.figure(71)
df_var_beta_wtmpc.boxplot(vert=False, showmeans=True,patch_artist=False)
plt.subplots_adjust(left=0.3)
plt.savefig(os.path.join(plot_dir, 'wtmpc_gait_eval_var_beta.png'))

plt.figure(72)
df_var_beta_asus.boxplot(vert=False, showmeans=True,patch_artist=False)
plt.subplots_adjust(left=0.3)
plt.savefig(os.path.join(plot_dir, 'asus_gait_eval_var_beta.png'))

plt.figure(81)
df_var_gamma_wtmpc.boxplot(vert=False, showmeans=True,patch_artist=False)
plt.subplots_adjust(left=0.3)
plt.savefig(os.path.join(plot_dir, 'wtmpc_gait_eval_var_gamma.png'))

plt.figure(82)
df_var_gamma_asus.boxplot(vert=False, showmeans=True,patch_artist=False)
plt.subplots_adjust(left=0.3)
plt.savefig(os.path.join(plot_dir, 'asus_gait_eval_var_gamma.png'))

# http://blog.bharatbhole.com/creating-boxplots-with-matplotlib/
# https://matplotlib.org/gallery/statistics/boxplot_color.html?highlight=box%20plots%20custom%20fill%20colors
