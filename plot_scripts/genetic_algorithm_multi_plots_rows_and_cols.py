import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import os

# Machine 1: wtmpc
# Machine 2: asus

asus_openloop_run1 = {'data_fitness': 'ga_fitness_log_20170830_172324_.csv', 'data_max_dist': 'ga_max_dist_log_20170830_172324.csv'}
asus_openloop_run2 = {'data_fitness': 'ga_fitness_log_20171027_225729.csv', 'data_max_dist': 'ga_max_dist_log_20171027_225729.csv'}
asus_openloop_run3 = {'data_fitness': 'ga_fitness_log_20171030_033117.csv', 'data_max_dist': 'ga_max_dist_log_20171030_033117.csv'}

asus_anglefeedback_run1 = {'data_fitness': 'ga_fitness_log_20170908_214752.csv', 'data_max_dist': 'ga_max_dist_log_20170908_214752.csv'}
asus_anglefeedback_run2 = {'data_fitness': 'ga_fitness_log_20170929_131608.csv', 'data_max_dist': 'ga_max_dist_log_20170929_131608.csv'}
asus_anglefeedback_run3 = {'data_fitness': 'ga_fitness_log_20171009_101803.csv', 'data_max_dist': 'ga_max_dist_log_20171009_101803.csv'}

asus_phasereset_run1 = {'data_fitness': 'ga_fitness_log_20170921_110728.csv', 'data_max_dist': 'ga_max_dist_log_20170921_110728.csv'}
asus_phasereset_run2 = {'data_fitness': 'ga_fitness_log_20171001_170505.csv', 'data_max_dist': 'ga_max_dist_log_20171001_170505.csv'}
asus_phasereset_run3 = {'data_fitness': 'ga_fitness_log_20171012_012106.csv', 'data_max_dist': 'ga_max_dist_log_20171012_012106.csv'}

wtmpc_openloop_run1 = {'data_fitness': 'ga_fitness_log_20171002_113858.csv', 'data_max_dist': 'ga_max_dist_log_20171002_113858.csv'}
wtmpc_openloop_run2 = {'data_fitness': 'ga_fitness_log_20171004_094431.csv', 'data_max_dist': 'ga_max_dist_log_20171004_094431.csv'}
wtmpc_openloop_run3 = {'data_fitness': 'ga_fitness_log_20171006_074055.csv', 'data_max_dist': 'ga_max_dist_log_20171006_074055.csv'}

wtmpc_anglefeedback_run1 = {'data_fitness': 'ga_fitness_log_20171002_182720.csv', 'data_max_dist': 'ga_max_dist_log_20171002_182720.csv'}
wtmpc_anglefeedback_run2 = {'data_fitness': 'ga_fitness_log_20171004_173049.csv', 'data_max_dist': 'ga_max_dist_log_20171004_173049.csv'}
wtmpc_anglefeedback_run3 = {'data_fitness': 'ga_fitness_log_20171006_145448.csv', 'data_max_dist': 'ga_max_dist_log_20171006_145448.csv'}

wtmpc_phasereset_run1 = {'data_fitness': 'ga_fitness_log_20171002_120630.csv', 'data_max_dist': 'ga_max_dist_log_20171002_120630.csv'}
wtmpc_phasereset_run2 = {'data_fitness': 'ga_fitness_log_20171004_132604.csv', 'data_max_dist': 'ga_max_dist_log_20171004_132604.csv'}
wtmpc_phasereset_run3 = {'data_fitness': 'ga_fitness_log_20171006_140001.csv', 'data_max_dist': 'ga_max_dist_log_20171006_140001.csv'}

# Set the home directory
home_dir = os.path.expanduser('~')

# Set the directory for saving plots
plot_dir = os.path.join(home_dir, 'computing/repositories/MScThesis_SayantanAuddy_2017_NICOOscillatorWalking/plots')

# Directory for saving plot data files
plot_data_dir = os.path.join(home_dir, 'computing/repositories/MScThesis_SayantanAuddy_2017_NICOOscillatorWalking/plot_data')

# Set the directory where logs are stored
log_dir = os.path.join(home_dir, 'computing/repositories/MScThesis_SayantanAuddy_2017_NICOOscillatorWalking/logs/genetic_algorithm_logs')

# Read the datafiles for fitness
asus_openloop_run1_gen, asus_openloop_run1_min_f, asus_openloop_run1_max_f, asus_openloop_run1_avg_f, asus_openloop_run1_std_f = np.loadtxt(os.path.join(plot_data_dir, asus_openloop_run1['data_fitness']), delimiter=',', unpack=True, skiprows=1)
asus_openloop_run2_gen, asus_openloop_run2_min_f, asus_openloop_run2_max_f, asus_openloop_run2_avg_f, asus_openloop_run2_std_f = np.loadtxt(os.path.join(plot_data_dir, asus_openloop_run2['data_fitness']), delimiter=',', unpack=True, skiprows=1)
asus_openloop_run3_gen, asus_openloop_run3_min_f, asus_openloop_run3_max_f, asus_openloop_run3_avg_f, asus_openloop_run3_std_f = np.loadtxt(os.path.join(plot_data_dir, asus_openloop_run3['data_fitness']), delimiter=',', unpack=True, skiprows=1)

asus_anglefeedback_run1_gen, asus_anglefeedback_run1_min_f, asus_anglefeedback_run1_max_f, asus_anglefeedback_run1_avg_f, asus_anglefeedback_run1_std_f = np.loadtxt(os.path.join(plot_data_dir, asus_anglefeedback_run1['data_fitness']), delimiter=',', unpack=True, skiprows=1)
asus_anglefeedback_run2_gen, asus_anglefeedback_run2_min_f, asus_anglefeedback_run2_max_f, asus_anglefeedback_run2_avg_f, asus_anglefeedback_run2_std_f = np.loadtxt(os.path.join(plot_data_dir, asus_anglefeedback_run2['data_fitness']), delimiter=',', unpack=True, skiprows=1)
asus_anglefeedback_run3_gen, asus_anglefeedback_run3_min_f, asus_anglefeedback_run3_max_f, asus_anglefeedback_run3_avg_f, asus_anglefeedback_run3_std_f = np.loadtxt(os.path.join(plot_data_dir, asus_anglefeedback_run3['data_fitness']), delimiter=',', unpack=True, skiprows=1)

asus_phasereset_run1_gen, asus_phasereset_run1_min_f, asus_phasereset_run1_max_f, asus_phasereset_run1_avg_f, asus_phasereset_run1_std_f = np.loadtxt(os.path.join(plot_data_dir, asus_phasereset_run1['data_fitness']), delimiter=',', unpack=True, skiprows=1)
asus_phasereset_run2_gen, asus_phasereset_run2_min_f, asus_phasereset_run2_max_f, asus_phasereset_run2_avg_f, asus_phasereset_run2_std_f = np.loadtxt(os.path.join(plot_data_dir, asus_phasereset_run2['data_fitness']), delimiter=',', unpack=True, skiprows=1)
asus_phasereset_run3_gen, asus_phasereset_run3_min_f, asus_phasereset_run3_max_f, asus_phasereset_run3_avg_f, asus_phasereset_run3_std_f = np.loadtxt(os.path.join(plot_data_dir, asus_phasereset_run3['data_fitness']), delimiter=',', unpack=True, skiprows=1)

wtmpc_openloop_run1_gen, wtmpc_openloop_run1_min_f, wtmpc_openloop_run1_max_f, wtmpc_openloop_run1_avg_f, wtmpc_openloop_run1_std_f = np.loadtxt(os.path.join(plot_data_dir, wtmpc_openloop_run1['data_fitness']), delimiter=',', unpack=True, skiprows=1)
wtmpc_openloop_run2_gen, wtmpc_openloop_run2_min_f, wtmpc_openloop_run2_max_f, wtmpc_openloop_run2_avg_f, wtmpc_openloop_run2_std_f = np.loadtxt(os.path.join(plot_data_dir, wtmpc_openloop_run2['data_fitness']), delimiter=',', unpack=True, skiprows=1)
wtmpc_openloop_run3_gen, wtmpc_openloop_run3_min_f, wtmpc_openloop_run3_max_f, wtmpc_openloop_run3_avg_f, wtmpc_openloop_run3_std_f = np.loadtxt(os.path.join(plot_data_dir, wtmpc_openloop_run3['data_fitness']), delimiter=',', unpack=True, skiprows=1)

wtmpc_anglefeedback_run1_gen, wtmpc_anglefeedback_run1_min_f, wtmpc_anglefeedback_run1_max_f, wtmpc_anglefeedback_run1_avg_f, wtmpc_anglefeedback_run1_std_f = np.loadtxt(os.path.join(plot_data_dir, wtmpc_anglefeedback_run1['data_fitness']), delimiter=',', unpack=True, skiprows=1)
wtmpc_anglefeedback_run2_gen, wtmpc_anglefeedback_run2_min_f, wtmpc_anglefeedback_run2_max_f, wtmpc_anglefeedback_run2_avg_f, wtmpc_anglefeedback_run2_std_f = np.loadtxt(os.path.join(plot_data_dir, wtmpc_anglefeedback_run2['data_fitness']), delimiter=',', unpack=True, skiprows=1)
wtmpc_anglefeedback_run3_gen, wtmpc_anglefeedback_run3_min_f, wtmpc_anglefeedback_run3_max_f, wtmpc_anglefeedback_run3_avg_f, wtmpc_anglefeedback_run3_std_f = np.loadtxt(os.path.join(plot_data_dir, wtmpc_anglefeedback_run3['data_fitness']), delimiter=',', unpack=True, skiprows=1)

wtmpc_phasereset_run1_gen, wtmpc_phasereset_run1_min_f, wtmpc_phasereset_run1_max_f, wtmpc_phasereset_run1_avg_f, wtmpc_phasereset_run1_std_f = np.loadtxt(os.path.join(plot_data_dir, wtmpc_phasereset_run1['data_fitness']), delimiter=',', unpack=True, skiprows=1)
wtmpc_phasereset_run2_gen, wtmpc_phasereset_run2_min_f, wtmpc_phasereset_run2_max_f, wtmpc_phasereset_run2_avg_f, wtmpc_phasereset_run2_std_f = np.loadtxt(os.path.join(plot_data_dir, wtmpc_phasereset_run2['data_fitness']), delimiter=',', unpack=True, skiprows=1)
wtmpc_phasereset_run3_gen, wtmpc_phasereset_run3_min_f, wtmpc_phasereset_run3_max_f, wtmpc_phasereset_run3_avg_f, wtmpc_phasereset_run3_std_f = np.loadtxt(os.path.join(plot_data_dir, wtmpc_phasereset_run3['data_fitness']), delimiter=',', unpack=True, skiprows=1)

# Read the datafiles for max distance
asus_openloop_run1_gen, asus_openloop_run1_max_d = np.loadtxt(os.path.join(plot_data_dir, asus_openloop_run1['data_max_dist']), delimiter=',', unpack=True, skiprows=1)
asus_openloop_run2_gen, asus_openloop_run2_max_d = np.loadtxt(os.path.join(plot_data_dir, asus_openloop_run2['data_max_dist']), delimiter=',', unpack=True, skiprows=1)
asus_openloop_run3_gen, asus_openloop_run3_max_d = np.loadtxt(os.path.join(plot_data_dir, asus_openloop_run3['data_max_dist']), delimiter=',', unpack=True, skiprows=1)

asus_anglefeedback_run1_gen, asus_anglefeedback_run1_max_d = np.loadtxt(os.path.join(plot_data_dir, asus_anglefeedback_run1['data_max_dist']), delimiter=',', unpack=True, skiprows=1)
asus_anglefeedback_run2_gen, asus_anglefeedback_run2_max_d = np.loadtxt(os.path.join(plot_data_dir, asus_anglefeedback_run2['data_max_dist']), delimiter=',', unpack=True, skiprows=1)
asus_anglefeedback_run3_gen, asus_anglefeedback_run3_max_d = np.loadtxt(os.path.join(plot_data_dir, asus_anglefeedback_run3['data_max_dist']), delimiter=',', unpack=True, skiprows=1)

asus_phasereset_run1_gen, asus_phasereset_run1_max_d = np.loadtxt(os.path.join(plot_data_dir, asus_phasereset_run1['data_max_dist']), delimiter=',', unpack=True, skiprows=1)
asus_phasereset_run2_gen, asus_phasereset_run2_max_d = np.loadtxt(os.path.join(plot_data_dir, asus_phasereset_run2['data_max_dist']), delimiter=',', unpack=True, skiprows=1)
asus_phasereset_run3_gen, asus_phasereset_run3_max_d = np.loadtxt(os.path.join(plot_data_dir, asus_phasereset_run3['data_max_dist']), delimiter=',', unpack=True, skiprows=1)

wtmpc_openloop_run1_gen, wtmpc_openloop_run1_max_d = np.loadtxt(os.path.join(plot_data_dir, wtmpc_openloop_run1['data_max_dist']), delimiter=',', unpack=True, skiprows=1)
wtmpc_openloop_run2_gen, wtmpc_openloop_run2_max_d = np.loadtxt(os.path.join(plot_data_dir, wtmpc_openloop_run2['data_max_dist']), delimiter=',', unpack=True, skiprows=1)
wtmpc_openloop_run3_gen, wtmpc_openloop_run3_max_d = np.loadtxt(os.path.join(plot_data_dir, wtmpc_openloop_run3['data_max_dist']), delimiter=',', unpack=True, skiprows=1)

wtmpc_anglefeedback_run1_gen, wtmpc_anglefeedback_run1_max_d = np.loadtxt(os.path.join(plot_data_dir, wtmpc_anglefeedback_run1['data_max_dist']), delimiter=',', unpack=True, skiprows=1)
wtmpc_anglefeedback_run2_gen, wtmpc_anglefeedback_run2_max_d = np.loadtxt(os.path.join(plot_data_dir, wtmpc_anglefeedback_run2['data_max_dist']), delimiter=',', unpack=True, skiprows=1)
wtmpc_anglefeedback_run3_gen, wtmpc_anglefeedback_run3_max_d = np.loadtxt(os.path.join(plot_data_dir, wtmpc_anglefeedback_run3['data_max_dist']), delimiter=',', unpack=True, skiprows=1)

wtmpc_phasereset_run1_gen, wtmpc_phasereset_run1_max_d = np.loadtxt(os.path.join(plot_data_dir, wtmpc_phasereset_run1['data_max_dist']), delimiter=',', unpack=True, skiprows=1)
wtmpc_phasereset_run2_gen, wtmpc_phasereset_run2_max_d = np.loadtxt(os.path.join(plot_data_dir, wtmpc_phasereset_run2['data_max_dist']), delimiter=',', unpack=True, skiprows=1)
wtmpc_phasereset_run3_gen, wtmpc_phasereset_run3_max_d = np.loadtxt(os.path.join(plot_data_dir, wtmpc_phasereset_run3['data_max_dist']), delimiter=',', unpack=True, skiprows=1)

# Change the font size used in the plots
font_size = 12
plt.rcParams.update({'font.size': font_size})


# Plot 1 contains fitness plots for machine 1
# Rows correspond to runs (1, 2, 3)
# Columns correspond to feedback types (Open Loop, Angle Feedback, Phase Reset)

row_names = ['Run {}'.format(col) for col in range(1, 4)]
col_names = ['{}'.format(row) for row in ['Open Loop', 'Angle Feedback', 'Phase Reset']]

fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(15,10))

for a, col in zip(ax[0], col_names):
    a.set_title(col)

for a, row in zip(ax[:,0], row_names):
    a.annotate(row, xy=(0, 0.5), xytext=(-a.yaxis.labelpad - 5, 0), xycoords=a.yaxis.label, textcoords='offset points', size='large', ha='right', va='center')

# Counters for row and columns
r,c = 1,1

for row in ax:
    for col in row:
        col.set_xlim([1, 30])
        col.set_ylim([-5, 15])
        if r==3: col.set_xlabel('Generation')
        col.set_ylabel('Fitness')
        if c==1:
            # Open loop
            if r==1:
                # Run 1
                col.plot(asus_openloop_run1_gen, asus_openloop_run1_min_f, color='red', linewidth=2.0, label='Minimum Fitness')
                col.plot(asus_openloop_run1_gen, asus_openloop_run1_max_f, color='blue', linewidth=2.0, label='Maximum Fitness')
                col.plot(asus_openloop_run1_gen, asus_openloop_run1_avg_f, color='orange', linewidth=2.0, label='Average Fitness')
                col.plot(asus_openloop_run1_gen, asus_openloop_run1_std_f, color='green', linewidth=2.0, label='Std. Fitness')
            elif r==2:
                # Run 2
                col.plot(asus_openloop_run2_gen, asus_openloop_run2_min_f, color='red', linewidth=2.0, label='Minimum Fitness')
                col.plot(asus_openloop_run2_gen, asus_openloop_run2_max_f, color='blue', linewidth=2.0, label='Maximum Fitness')
                col.plot(asus_openloop_run2_gen, asus_openloop_run2_avg_f, color='orange', linewidth=2.0, label='Average Fitness')
                col.plot(asus_openloop_run2_gen, asus_openloop_run2_std_f, color='green', linewidth=2.0, label='Std. Fitness')
            elif r==3:
                # Run 3
                col.plot(asus_openloop_run3_gen, asus_openloop_run3_min_f, color='red', linewidth=2.0, label='Minimum Fitness')
                col.plot(asus_openloop_run3_gen, asus_openloop_run3_max_f, color='blue', linewidth=2.0, label='Maximum Fitness')
                col.plot(asus_openloop_run3_gen, asus_openloop_run3_avg_f, color='orange', linewidth=2.0, label='Average Fitness')
                col.plot(asus_openloop_run3_gen, asus_openloop_run3_std_f, color='green', linewidth=2.0, label='Std. Fitness')
        elif c==2:
            # Angle Feedback
            if r==1:
                # Run 1
                col.plot(asus_anglefeedback_run1_gen, asus_anglefeedback_run1_min_f, color='red', linewidth=2.0, label='Minimum Fitness')
                col.plot(asus_anglefeedback_run1_gen, asus_anglefeedback_run1_max_f, color='blue', linewidth=2.0, label='Maximum Fitness')
                col.plot(asus_anglefeedback_run1_gen, asus_anglefeedback_run1_avg_f, color='orange', linewidth=2.0, label='Average Fitness')
                col.plot(asus_anglefeedback_run1_gen, asus_anglefeedback_run1_std_f, color='green', linewidth=2.0, label='Std. Fitness')
            elif r==2:
                # Run 2
                col.plot(asus_anglefeedback_run2_gen, asus_anglefeedback_run2_min_f, color='red', linewidth=2.0, label='Minimum Fitness')
                col.plot(asus_anglefeedback_run2_gen, asus_anglefeedback_run2_max_f, color='blue', linewidth=2.0, label='Maximum Fitness')
                col.plot(asus_anglefeedback_run2_gen, asus_anglefeedback_run2_avg_f, color='orange', linewidth=2.0, label='Average Fitness')
                col.plot(asus_anglefeedback_run2_gen, asus_anglefeedback_run2_std_f, color='green', linewidth=2.0, label='Std. Fitness')
            elif r==3:
                # Run 3
                col.plot(asus_anglefeedback_run3_gen, asus_anglefeedback_run3_min_f, color='red', linewidth=2.0, label='Minimum Fitness')
                col.plot(asus_anglefeedback_run3_gen, asus_anglefeedback_run3_max_f, color='blue', linewidth=2.0, label='Maximum Fitness')
                col.plot(asus_anglefeedback_run3_gen, asus_anglefeedback_run3_avg_f, color='orange', linewidth=2.0, label='Average Fitness')
                col.plot(asus_anglefeedback_run3_gen, asus_anglefeedback_run3_std_f, color='green', linewidth=2.0, label='Std. Fitness')
        elif c==3:
            # Phase reset
            if r==1:
                # Run 1
                col.plot(asus_phasereset_run1_gen, asus_phasereset_run1_min_f, color='red', linewidth=2.0, label='Minimum Fitness')
                col.plot(asus_phasereset_run1_gen, asus_phasereset_run1_max_f, color='blue', linewidth=2.0, label='Maximum Fitness')
                col.plot(asus_phasereset_run1_gen, asus_phasereset_run1_avg_f, color='orange', linewidth=2.0, label='Average Fitness')
                col.plot(asus_phasereset_run1_gen, asus_phasereset_run1_std_f, color='green', linewidth=2.0, label='Std. Fitness')
            elif r==2:
                # Run 2
                col.plot(asus_phasereset_run2_gen, asus_phasereset_run2_min_f, color='red', linewidth=2.0, label='Minimum Fitness')
                col.plot(asus_phasereset_run2_gen, asus_phasereset_run2_max_f, color='blue', linewidth=2.0, label='Maximum Fitness')
                col.plot(asus_phasereset_run2_gen, asus_phasereset_run2_avg_f, color='orange', linewidth=2.0, label='Average Fitness')
                col.plot(asus_phasereset_run2_gen, asus_phasereset_run2_std_f, color='green', linewidth=2.0, label='Std. Fitness')
            elif r==3:
                # Run 3
                col.plot(asus_phasereset_run3_gen, asus_phasereset_run3_min_f, color='red', linewidth=2.0, label='Minimum Fitness')
                col.plot(asus_phasereset_run3_gen, asus_phasereset_run3_max_f, color='blue', linewidth=2.0, label='Maximum Fitness')
                col.plot(asus_phasereset_run3_gen, asus_phasereset_run3_avg_f, color='orange', linewidth=2.0, label='Average Fitness')
                col.plot(asus_phasereset_run3_gen, asus_phasereset_run3_std_f, color='green', linewidth=2.0, label='Std. Fitness')

        c += 1
    r += 1
    c = 1

#fig.legend(ncol=2)
fig.subplots_adjust(wspace=0.2, hspace=0.3)
#plt.suptitle('Fitness: Machine 1', fontsize=18)
plt.savefig(os.path.join(plot_dir,'ga_sublot_asus_fitness.eps'))
plt.close()
plt.gcf().clear()

# Plot 2 contains fitness plots for machine 1
# Rows correspond to runs (1, 2, 3)
# Columns correspond to feedback types (Open Loop, Angle Feedback, Phase Reset)

row_names = ['Run {}'.format(col) for col in range(1, 4)]
col_names = ['{}'.format(row) for row in ['Open Loop', 'Angle Feedback', 'Phase Reset']]

fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(15,10))

for a, col in zip(ax[0], col_names):
    a.set_title(col)

for a, row in zip(ax[:,0], row_names):
    a.annotate(row, xy=(0, 0.5), xytext=(-a.yaxis.labelpad - 5, 0), xycoords=a.yaxis.label, textcoords='offset points', size='large', ha='right', va='center')

# Counters for row and columns
r,c = 1,1

for row in ax:
    for col in row:
        col.set_xlim([1, 30])
        col.set_ylim([-5, 15])
        col.set_xlabel('Generation')
        col.set_ylabel('Fitness')
        if c==1:
            # Open loop
            if r==1:
                # Run 1
                col.plot(wtmpc_openloop_run1_gen, wtmpc_openloop_run1_min_f, color='red', linewidth=2.0, label='Minimum Fitness')
                col.plot(wtmpc_openloop_run1_gen, wtmpc_openloop_run1_max_f, color='blue', linewidth=2.0, label='Maximum Fitness')
                col.plot(wtmpc_openloop_run1_gen, wtmpc_openloop_run1_avg_f, color='orange', linewidth=2.0, label='Average Fitness')
                col.plot(wtmpc_openloop_run1_gen, wtmpc_openloop_run1_std_f, color='green', linewidth=2.0, label='Std. Fitness')
            elif r==2:
                # Run 2
                col.plot(wtmpc_openloop_run2_gen, wtmpc_openloop_run2_min_f, color='red', linewidth=2.0, label='Minimum Fitness')
                col.plot(wtmpc_openloop_run2_gen, wtmpc_openloop_run2_max_f, color='blue', linewidth=2.0, label='Maximum Fitness')
                col.plot(wtmpc_openloop_run2_gen, wtmpc_openloop_run2_avg_f, color='orange', linewidth=2.0, label='Average Fitness')
                col.plot(wtmpc_openloop_run2_gen, wtmpc_openloop_run2_std_f, color='green', linewidth=2.0, label='Std. Fitness')
            elif r==3:
                # Run 3
                col.plot(wtmpc_openloop_run3_gen, wtmpc_openloop_run3_min_f, color='red', linewidth=2.0, label='Minimum Fitness')
                col.plot(wtmpc_openloop_run3_gen, wtmpc_openloop_run3_max_f, color='blue', linewidth=2.0, label='Maximum Fitness')
                col.plot(wtmpc_openloop_run3_gen, wtmpc_openloop_run3_avg_f, color='orange', linewidth=2.0, label='Average Fitness')
                col.plot(wtmpc_openloop_run3_gen, wtmpc_openloop_run3_std_f, color='green', linewidth=2.0, label='Std. Fitness')
        elif c==2:
            # Angle Feedback
            if r==1:
                # Run 1
                col.plot(wtmpc_anglefeedback_run1_gen, wtmpc_anglefeedback_run1_min_f, color='red', linewidth=2.0, label='Minimum Fitness')
                col.plot(wtmpc_anglefeedback_run1_gen, wtmpc_anglefeedback_run1_max_f, color='blue', linewidth=2.0, label='Maximum Fitness')
                col.plot(wtmpc_anglefeedback_run1_gen, wtmpc_anglefeedback_run1_avg_f, color='orange', linewidth=2.0, label='Average Fitness')
                col.plot(wtmpc_anglefeedback_run1_gen, wtmpc_anglefeedback_run1_std_f, color='green', linewidth=2.0, label='Std. Fitness')
            elif r==2:
                # Run 2
                col.plot(wtmpc_anglefeedback_run2_gen, wtmpc_anglefeedback_run2_min_f, color='red', linewidth=2.0, label='Minimum Fitness')
                col.plot(wtmpc_anglefeedback_run2_gen, wtmpc_anglefeedback_run2_max_f, color='blue', linewidth=2.0, label='Maximum Fitness')
                col.plot(wtmpc_anglefeedback_run2_gen, wtmpc_anglefeedback_run2_avg_f, color='orange', linewidth=2.0, label='Average Fitness')
                col.plot(wtmpc_anglefeedback_run2_gen, wtmpc_anglefeedback_run2_std_f, color='green', linewidth=2.0, label='Std. Fitness')
            elif r==3:
                # Run 3
                col.plot(wtmpc_anglefeedback_run3_gen, wtmpc_anglefeedback_run3_min_f, color='red', linewidth=2.0, label='Minimum Fitness')
                col.plot(wtmpc_anglefeedback_run3_gen, wtmpc_anglefeedback_run3_max_f, color='blue', linewidth=2.0, label='Maximum Fitness')
                col.plot(wtmpc_anglefeedback_run3_gen, wtmpc_anglefeedback_run3_avg_f, color='orange', linewidth=2.0, label='Average Fitness')
                col.plot(wtmpc_anglefeedback_run3_gen, wtmpc_anglefeedback_run3_std_f, color='green', linewidth=2.0, label='Std. Fitness')
        elif c==3:
            # Phase reset
            if r==1:
                # Run 1
                col.plot(wtmpc_phasereset_run1_gen, wtmpc_phasereset_run1_min_f, color='red', linewidth=2.0, label='Minimum Fitness')
                col.plot(wtmpc_phasereset_run1_gen, wtmpc_phasereset_run1_max_f, color='blue', linewidth=2.0, label='Maximum Fitness')
                col.plot(wtmpc_phasereset_run1_gen, wtmpc_phasereset_run1_avg_f, color='orange', linewidth=2.0, label='Average Fitness')
                col.plot(wtmpc_phasereset_run1_gen, wtmpc_phasereset_run1_std_f, color='green', linewidth=2.0, label='Std. Fitness')
            elif r==2:
                # Run 2
                col.plot(wtmpc_phasereset_run2_gen, wtmpc_phasereset_run2_min_f, color='red', linewidth=2.0, label='Minimum Fitness')
                col.plot(wtmpc_phasereset_run2_gen, wtmpc_phasereset_run2_max_f, color='blue', linewidth=2.0, label='Maximum Fitness')
                col.plot(wtmpc_phasereset_run2_gen, wtmpc_phasereset_run2_avg_f, color='orange', linewidth=2.0, label='Average Fitness')
                col.plot(wtmpc_phasereset_run2_gen, wtmpc_phasereset_run2_std_f, color='green', linewidth=2.0, label='Std. Fitness')
            elif r==3:
                # Run 3
                col.plot(wtmpc_phasereset_run3_gen, wtmpc_phasereset_run3_min_f, color='red', linewidth=2.0, label='Minimum Fitness')
                col.plot(wtmpc_phasereset_run3_gen, wtmpc_phasereset_run3_max_f, color='blue', linewidth=2.0, label='Maximum Fitness')
                col.plot(wtmpc_phasereset_run3_gen, wtmpc_phasereset_run3_avg_f, color='orange', linewidth=2.0, label='Average Fitness')
                col.plot(wtmpc_phasereset_run3_gen, wtmpc_phasereset_run3_std_f, color='green', linewidth=2.0, label='Std. Fitness')

        c += 1
    r += 1
    c = 1

#fig.legend(ncol=2)
fig.subplots_adjust(wspace=0.2, hspace=0.3)
#plt.suptitle('Fitness: Machine 2', fontsize=18)
plt.savefig(os.path.join(plot_dir,'ga_sublot_wtmpc_fitness.eps'))
plt.close()
plt.gcf().clear()

# Plot 3 contains distance plots for machine 1
# Rows correspond to runs (1, 2, 3)
# Columns correspond to feedback types (Open Loop, Angle Feedback, Phase Reset)

row_names = ['Run {}'.format(col) for col in range(1, 4)]
col_names = ['{}'.format(row) for row in ['Open Loop', 'Angle Feedback', 'Phase Reset']]

fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(15,10))

for a, col in zip(ax[0], col_names):
    a.set_title(col)

for a, row in zip(ax[:,0], row_names):
    a.annotate(row, xy=(0, 0.5), xytext=(-a.yaxis.labelpad - 5, 0), xycoords=a.yaxis.label, textcoords='offset points', size='large', ha='right', va='center')

# Counters for row and columns
r,c = 1,1

for row in ax:
    for col in row:
        col.set_xlim([1, 30])
        col.set_ylim([-1, 5])
        col.set_xlabel('Generation')
        col.set_ylabel('Maximum Distance')
        if c==1:
            # Open loop
            if r==1:
                # Run 1
                col.plot(asus_openloop_run1_gen, asus_openloop_run1_max_d, color='blue', linewidth=2.0, label='Maximum Distance')
            elif r==2:
                # Run 2
                col.plot(asus_openloop_run2_gen, asus_openloop_run2_max_d, color='blue', linewidth=2.0, label='Maximum Distance')
            elif r==3:
                # Run 3
                col.plot(asus_openloop_run3_gen, asus_openloop_run3_max_d, color='blue', linewidth=2.0, label='Maximum Distance')
        elif c==2:
            # Angle Feedback
            if r==1:
                # Run 1
                col.plot(asus_anglefeedback_run1_gen, asus_anglefeedback_run1_max_d, color='blue', linewidth=2.0, label='Maximum Distance')
            elif r==2:
                # Run 2
                col.plot(asus_anglefeedback_run2_gen, asus_anglefeedback_run2_max_d, color='blue', linewidth=2.0, label='Maximum Distance')
            elif r==3:
                # Run 3
                col.plot(asus_anglefeedback_run3_gen, asus_anglefeedback_run3_max_d, color='blue', linewidth=2.0, label='Maximum Distance')
        elif c==3:
            # Phase reset
            if r==1:
                # Run 1
                col.plot(asus_phasereset_run1_gen, asus_phasereset_run1_max_d, color='blue', linewidth=2.0, label='Maximum Distance')
            elif r==2:
                # Run 2
                col.plot(asus_phasereset_run2_gen, asus_phasereset_run2_max_d, color='blue', linewidth=2.0, label='Maximum Distance')
            elif r==3:
                # Run 3
                col.plot(asus_phasereset_run3_gen, asus_phasereset_run3_max_d, color='blue', linewidth=2.0, label='Maximum Distance')

        c += 1
    r += 1
    c = 1

#fig.legend(ncol=1, fontsize=14)
fig.subplots_adjust(wspace=0.2, hspace=0.3)
#plt.suptitle('Maximum Distance: Machine 1', fontsize=18)
plt.savefig(os.path.join(plot_dir,'ga_sublot_asus_distance.eps'))
plt.close()
plt.gcf().clear()

# Plot 4 contains distance plots for machine 2
# Rows correspond to runs (1, 2, 3)
# Columns correspond to feedback types (Open Loop, Angle Feedback, Phase Reset)

row_names = ['Run {}'.format(col) for col in range(1, 4)]
col_names = ['{}'.format(row) for row in ['Open Loop', 'Angle Feedback', 'Phase Reset']]

fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(15,10))

for a, col in zip(ax[0], col_names):
    a.set_title(col)

for a, row in zip(ax[:,0], row_names):
    a.annotate(row, xy=(0, 0.5), xytext=(-a.yaxis.labelpad - 5, 0), xycoords=a.yaxis.label, textcoords='offset points', size='large', ha='right', va='center')

# Counters for row and columns
r,c = 1,1

for row in ax:
    for col in row:
        col.set_xlim([1, 30])
        col.set_ylim([-1, 5])
        col.set_xlabel('Generation')
        col.set_ylabel('Maximum Distance')
        if c==1:
            # Open loop
            if r==1:
                # Run 1
                col.plot(wtmpc_openloop_run1_gen, wtmpc_openloop_run1_max_d, color='blue', linewidth=2.0, label='Maximum Distance')
            elif r==2:
                # Run 2
                col.plot(wtmpc_openloop_run2_gen, wtmpc_openloop_run2_max_d, color='blue', linewidth=2.0, label='Maximum Distance')
            elif r==3:
                # Run 3
                col.plot(wtmpc_openloop_run3_gen, wtmpc_openloop_run3_max_d, color='blue', linewidth=2.0, label='Maximum Distance')
        elif c==2:
            # Angle Feedback
            if r==1:
                # Run 1
                col.plot(wtmpc_anglefeedback_run1_gen, wtmpc_anglefeedback_run1_max_d, color='blue', linewidth=2.0, label='Maximum Distance')
            elif r==2:
                # Run 2
                col.plot(wtmpc_anglefeedback_run2_gen, wtmpc_anglefeedback_run2_max_d, color='blue', linewidth=2.0, label='Maximum Distance')
            elif r==3:
                # Run 3
                col.plot(wtmpc_anglefeedback_run3_gen, wtmpc_anglefeedback_run3_max_d, color='blue', linewidth=2.0, label='Maximum Distance')
        elif c==3:
            # Phase reset
            if r==1:
                # Run 1
                col.plot(wtmpc_phasereset_run1_gen, wtmpc_phasereset_run1_max_d, color='blue', linewidth=2.0, label='Maximum Distance')
            elif r==2:
                # Run 2
                col.plot(wtmpc_phasereset_run2_gen, wtmpc_phasereset_run2_max_d, color='blue', linewidth=2.0, label='Maximum Distance')
            elif r==3:
                # Run 3
                col.plot(wtmpc_phasereset_run3_gen, wtmpc_phasereset_run3_max_d, color='blue', linewidth=2.0, label='Maximum Distance')

        c += 1
    r += 1
    c = 1

#fig.legend(ncol=1, fontsize=14)
fig.subplots_adjust(wspace=0.2, hspace=0.3)
#plt.suptitle('Maximum Distance: Machine 2', fontsize=18)
plt.savefig(os.path.join(plot_dir,'ga_sublot_wtmpc_distance.eps'))
plt.close()
plt.gcf().clear()
