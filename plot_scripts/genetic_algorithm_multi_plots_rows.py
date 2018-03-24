import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import os

# Machine 1: wtmpc
# Machine 2: asus

asus_openloop_run1 = {'data_fitness': 'ga_fitness_log_20170830_172324.csv', 'data_max_dist': 'ga_max_dist_log_20170830_172324.csv'}
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


def gen_fitness_plot(run1_gen, openloop_run1_max_f, openloop_run1_avg_f,
                     run2_gen, openloop_run2_max_f, openloop_run2_avg_f,
                     run3_gen, openloop_run3_max_f, openloop_run3_avg_f,
                     plot_name,
                     grid=False):
    """
    Takes the generations and fitness data of 3 runs and creates the plot
    :param run1_gen: List of generations of run 1
    :param openloop_run1_max_f: List of max fitness of run 1
    :param openloop_run1_avg_f:  List of avg fitness of run 1
    :param run2_gen: List of generations of run 2
    :param openloop_run2_max_f: List of max fitness of run 2
    :param openloop_run2_avg_f: List of avg fitness of run 2
    :param run3_gen: List of generations of run 3
    :param openloop_run3_max_f: List of max fitness of run 3
    :param openloop_run3_avg_f: List of avg fitness of run 3
    :param plot_name: Name of the plot image file to be saved
    :return: none
    """
    # Figure 1 for open loop
    fig, ax = plt.subplots(3, figsize=(9.75,10), sharex=True, sharey=True)

    # Change the font size used in the plots
    font_size = 20
    plt.rcParams.update({'font.size': font_size})

    # Set the line width
    linewidth=3.5

    # Counters for row and columns
    r,c = 1,1

    # Set the x-axis limits
    ax[0].set_xlim([1, 30])
    ax[1].set_xlim([1, 30])
    ax[2].set_xlim([1, 30])

    # Set the y-axis limits
    ax[0].set_ylim([0, 15])
    ax[1].set_ylim([0, 15])
    ax[2].set_ylim([0, 15])

    # Set the axis labels (since x-axis is shared, only the last subplot has an xlabel)
    ax[2].set_xlabel('Generation', fontsize=24)

    ax[0].set_ylabel('  Run 1 \n Fitness', fontsize=24, horizontalalignment='center')
    ax[0].yaxis.set_ticks_position('both')
    ax[0].axes.yaxis.set_ticklabels([0,'',4,'',8,'',12,''])

    #ax[0].tick_params(axis='y', which='both', labelleft='on', labelright='on')
    ax[1].set_ylabel('  Run 2 \n Fitness', fontsize=24, horizontalalignment='center')
    ax[1].yaxis.set_ticks_position('both')
    ax[1].axes.yaxis.set_ticklabels([0,'',4,'',8,'',12,''])

    #ax[1].tick_params(axis='y', which='both', labelleft='on', labelright='on')
    ax[2].set_ylabel('  Run 3 \n Fitness', fontsize=24, horizontalalignment='center')
    ax[2].yaxis.set_ticks_position('both')
    ax[2].axes.yaxis.set_ticklabels([0,'',4,'',8,'',12,''])

    #ax[2].tick_params(axis='y', which='both', labelleft='on', labelright='on')

    # First subplot contains max and avg fitness for run 1
    ax[0].plot(run1_gen, openloop_run1_max_f, color='blue', linewidth=linewidth, label='Max. Fitness')
    ax[0].plot(run1_gen, openloop_run1_avg_f, color='red', linewidth=linewidth, label='Avg. Fitness')
    if grid: ax[0].grid(color='#222222', linestyle='dotted', linewidth=1)

    # Second subplot contains max and avg fitness for run 2
    ax[1].plot(run2_gen, openloop_run2_max_f, color='blue', linewidth=linewidth, label='Max. Fitness')
    ax[1].plot(run2_gen, openloop_run2_avg_f, color='red', linewidth=linewidth, label='Avg. Fitness')
    if grid: ax[1].grid(color='#222222', linestyle='dotted', linewidth=1)

    # Third subplot contains max and avg fitness for run 3
    ax[2].plot(run3_gen, openloop_run3_max_f, color='blue', linewidth=linewidth, label='Max. Fitness')
    ax[2].plot(run3_gen, openloop_run3_avg_f, color='red', linewidth=linewidth, label='Avg. Fitness')
    if grid: ax[2].grid(color='#222222', linestyle='dotted', linewidth=1)

    plt.subplots_adjust(hspace=0.05)
    plt.legend(bbox_to_anchor=(1.018, 3.45), loc='upper right', ncol=2)
    plt.savefig(os.path.join(plot_dir, plot_name), bbox_inches='tight')
    #plt.show()

def gen_maxdist_plot(run1_gen, openloop_run1_max_d,
                     run2_gen, openloop_run2_max_d,
                     run3_gen, openloop_run3_max_d,
                     plot_name,
                     grid=False):
    """
    Takes the generations and fitness data of 3 runs and creates the plot
    :param run1_gen: List of generations of run 1
    :param openloop_run1_max_d: List of max fitness of run 1
    :param run2_gen: List of generations of run 2
    :param openloop_run2_max_d: List of max distance of run 2
    :param run3_gen: List of generations of run 3
    :param openloop_run3_max_d: List of max distance of run 3
    :param plot_name: Name of the plot image file to be saved
    :return: none
    """
    # Figure 1 for open loop
    fig, ax = plt.subplots(3, figsize=(9.75,10), sharex=True, sharey=True)

    # Change the font size used in the plots
    font_size = 20
    plt.rcParams.update({'font.size': font_size})

    # Set the line width
    linewidth=3.5

    # Counters for row and columns
    r,c = 1,1

    # Set the x-axis limits
    ax[0].set_xlim([1, 30])
    ax[1].set_xlim([1, 30])
    ax[2].set_xlim([1, 30])

    # Set the y-axis limits
    ax[0].set_ylim([0, 4.8])
    ax[1].set_ylim([0, 4.8])
    ax[2].set_ylim([0, 4.8])

    # Set the axis labels (since x-axis is shared, only the last subplot has an xlabel)
    ax[2].set_xlabel('Generation', fontsize=24)

    ax[0].set_ylabel('Distance (m)', fontsize=24, horizontalalignment='center')
    ax[0].yaxis.set_ticks_position('both')
    ax[1].set_ylabel('Distance (m)', fontsize=24, horizontalalignment='center')
    ax[1].yaxis.set_ticks_position('both')
    ax[2].set_ylabel('Distance (m)', fontsize=24, horizontalalignment='center')
    ax[2].yaxis.set_ticks_position('both')

    # First subplot contains max and avg fitness for run 1
    ax[0].plot(run1_gen, openloop_run1_max_d, color='green', linewidth=linewidth, label='Distance')
    if grid: ax[0].grid(color='#222222', linestyle='dotted', linewidth=1)

    # Second subplot contains max and avg fitness for run 2
    ax[1].plot(run2_gen, openloop_run2_max_d, color='green', linewidth=linewidth, label='Distance')
    if grid: ax[1].grid(color='#222222', linestyle='dotted', linewidth=1)

    # Third subplot contains max and avg fitness for run 3
    ax[2].plot(run3_gen, openloop_run3_max_d, color='green', linewidth=linewidth, label='Distance')
    if grid: ax[2].grid(color='#222222', linestyle='dotted', linewidth=1)

    plt.subplots_adjust(hspace=0.05)
    plt.legend(bbox_to_anchor=(1.018, 3.45), loc='upper right', ncol=2)
    plt.savefig(os.path.join(plot_dir, plot_name), bbox_inches='tight')
    #plt.show()

# 3 figures will be generated for each setup, each containing 3 plots for runs 1,2 and 3.

# wtmpc plots
# Generate fitness plot for open loop (wtmpc)
gen_fitness_plot(wtmpc_openloop_run1_gen, wtmpc_openloop_run1_max_f, wtmpc_openloop_run1_avg_f,
                 wtmpc_openloop_run2_gen, wtmpc_openloop_run2_max_f, wtmpc_openloop_run2_avg_f,
                 wtmpc_openloop_run3_gen, wtmpc_openloop_run3_max_f, wtmpc_openloop_run3_avg_f,
                 'wtmpc_openloop_fitness.svg',
                 grid=True)

# Generate fitness plot for angle feedback (wtmpc)
gen_fitness_plot(wtmpc_anglefeedback_run1_gen, wtmpc_anglefeedback_run1_max_f, wtmpc_anglefeedback_run1_avg_f,
                 wtmpc_anglefeedback_run2_gen, wtmpc_anglefeedback_run2_max_f, wtmpc_anglefeedback_run2_avg_f,
                 wtmpc_anglefeedback_run3_gen, wtmpc_anglefeedback_run3_max_f, wtmpc_anglefeedback_run3_avg_f,
                 'wtmpc_anglefeedback_fitness.svg',
                 grid=True)

# Generate fitness plot for phase reset (wtmpc)
gen_fitness_plot(wtmpc_phasereset_run1_gen, wtmpc_phasereset_run1_max_f, wtmpc_phasereset_run1_avg_f,
                 wtmpc_phasereset_run2_gen, wtmpc_phasereset_run2_max_f, wtmpc_phasereset_run2_avg_f,
                 wtmpc_phasereset_run3_gen, wtmpc_phasereset_run3_max_f, wtmpc_phasereset_run3_avg_f,
                 'wtmpc_phasereset_fitness.svg',
                 grid=True)

# Generate distance plot for open loop (wtmpc)
gen_maxdist_plot(wtmpc_openloop_run1_gen, wtmpc_openloop_run1_max_d,
                 wtmpc_openloop_run2_gen, wtmpc_openloop_run2_max_d,
                 wtmpc_openloop_run3_gen, wtmpc_openloop_run3_max_d,
                 'wtmpc_openloop_distance.svg',
                 grid=True)

# Generate fitness plot for angle feedback (wtmpc)
gen_maxdist_plot(wtmpc_anglefeedback_run1_gen, wtmpc_anglefeedback_run1_max_d,
                 wtmpc_anglefeedback_run2_gen, wtmpc_anglefeedback_run2_max_d,
                 wtmpc_anglefeedback_run3_gen, wtmpc_anglefeedback_run3_max_d,
                 'wtmpc_anglefeedback_distance.svg',
                 grid=True)

# Generate fitness plot for phase reset (wtmpc)
gen_maxdist_plot(wtmpc_phasereset_run1_gen, wtmpc_phasereset_run1_max_d,
                 wtmpc_phasereset_run2_gen, wtmpc_phasereset_run2_max_d,
                 wtmpc_phasereset_run3_gen, wtmpc_phasereset_run3_max_d,
                 'wtmpc_phasereset_distance.svg',
                 grid=True)

### Asus

# asus plots
# Generate fitness plot for open loop (asus)
gen_fitness_plot(asus_openloop_run1_gen, asus_openloop_run1_max_f, asus_openloop_run1_avg_f,
                 asus_openloop_run2_gen, asus_openloop_run2_max_f, asus_openloop_run2_avg_f,
                 asus_openloop_run3_gen, asus_openloop_run3_max_f, asus_openloop_run3_avg_f,
                 'asus_openloop_fitness.svg',
                 grid=True)

# Generate fitness plot for angle feedback (asus)
gen_fitness_plot(asus_anglefeedback_run1_gen, asus_anglefeedback_run1_max_f, asus_anglefeedback_run1_avg_f,
                 asus_anglefeedback_run2_gen, asus_anglefeedback_run2_max_f, asus_anglefeedback_run2_avg_f,
                 asus_anglefeedback_run3_gen, asus_anglefeedback_run3_max_f, asus_anglefeedback_run3_avg_f,
                 'asus_anglefeedback_fitness.svg',
                 grid=True)

# Generate fitness plot for phase reset (asus)
gen_fitness_plot(asus_phasereset_run1_gen, asus_phasereset_run1_max_f, asus_phasereset_run1_avg_f,
                 asus_phasereset_run2_gen, asus_phasereset_run2_max_f, asus_phasereset_run2_avg_f,
                 asus_phasereset_run3_gen, asus_phasereset_run3_max_f, asus_phasereset_run3_avg_f,
                 'asus_phasereset_fitness.svg',
                 grid=True)

# Generate distance plot for open loop (asus)
gen_maxdist_plot(asus_openloop_run1_gen, asus_openloop_run1_max_d,
                 asus_openloop_run2_gen, asus_openloop_run2_max_d,
                 asus_openloop_run3_gen, asus_openloop_run3_max_d,
                 'asus_openloop_distance.svg',
                 grid=True)

# Generate fitness plot for angle feedback (asus)
gen_maxdist_plot(asus_anglefeedback_run1_gen, asus_anglefeedback_run1_max_d,
                 asus_anglefeedback_run2_gen, asus_anglefeedback_run2_max_d,
                 asus_anglefeedback_run3_gen, asus_anglefeedback_run3_max_d,
                 'asus_anglefeedback_distance.svg',
                 grid=True)

# Generate fitness plot for phase reset (asus)
gen_maxdist_plot(asus_phasereset_run1_gen, asus_phasereset_run1_max_d,
                 asus_phasereset_run2_gen, asus_phasereset_run2_max_d,
                 asus_phasereset_run3_gen, asus_phasereset_run3_max_d,
                 'asus_phasereset_distance.svg',
                 grid=True)

