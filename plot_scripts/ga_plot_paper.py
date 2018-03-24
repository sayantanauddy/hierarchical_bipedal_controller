import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import os
import matplotlib

# For avoiding type-3 fonts
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# Set the home directory
home_dir = os.path.expanduser('~')

# Set the directory for saving plots
plot_dir = os.path.join(home_dir, 'computing/repositories/MScThesis_SayantanAuddy_2017_NICOOscillatorWalking/plots')

# Directory for saving plot data files
plot_data_dir = os.path.join(home_dir, 'computing/repositories/MScThesis_SayantanAuddy_2017_NICOOscillatorWalking/plot_data')

# Set the directory where logs are stored
log_dir = os.path.join(home_dir, 'computing/repositories/MScThesis_SayantanAuddy_2017_NICOOscillatorWalking/logs/genetic_algorithm_logs')


def gen_fitness_plot(gen, max_f, avg_f, max_d, plot_name, grid=False):

    # Figure 1 for open loop
    fig, ax = plt.subplots(1,2,figsize=(20,5))

    # Change the font size used in the plots
    font_size = 36
    plt.rcParams.update({'font.size': font_size})

    # Set the line width
    linewidth=3.5

    # Counters for row and columns
    r,c = 1,1

    # Set the x-axis limits
    ax[0].set_xlim([1, 30])
    ax[0].set_xlabel('Generation', fontsize=font_size)
    ax[1].set_xlim([1, 30])
    ax[1].set_xlabel('Generation', fontsize=font_size)

    # Set the y-axis limits
    ax[0].set_ylim([0, 14.8])
    ax[0].set_ylabel('Fitness', fontsize=font_size, horizontalalignment='center')

    ax[1].set_ylim([0, 4.8])
    ax[1].set_ylabel('Max. Distance (m)', fontsize=font_size, horizontalalignment='center')

    # Set the axis labels (since x-axis is shared, only the last subplot has an xlabel)

    ax[0].yaxis.set_ticks_position('both')
    ax[0].axes.yaxis.set_ticklabels([0,'',4,'',8,'',12,''])
    ax[0].tick_params(axis='y', pad=15)
    ax[0].tick_params(axis='x', pad=15)

    ax[1].yaxis.set_ticks_position('both')
    #ax2.axes.yaxis.set_ticklabels([0, 1, 2, 3, 4])
    ax[1].tick_params(axis='y', pad=15)
    ax[1].tick_params(axis='x', pad=15)

    # First subplot contains max and avg fitness for run 1
    l0, = ax[0].plot(gen, max_f, color='blue', linewidth=linewidth, label='Max. Fitness')
    l1, = ax[0].plot(gen, avg_f, color='red', linewidth=linewidth, label='Avg. Fitness')
    leg = ax[0].legend(loc='lower right', fontsize=font_size-4)
    leg.get_frame().set_alpha(0.5)

    l2, = ax[1].plot(gen, max_d, color='green', linewidth=linewidth, label='Max. Distance')
    if grid:
        ax[0].grid(color='#222222', linestyle='dotted', linewidth=1)
        ax[1].grid(color='#222222', linestyle='dotted', linewidth=1)

    plt.subplots_adjust(wspace=0.25)
    #plt.legend(handles=[l0, l1, l2], bbox_to_anchor=(1.025, 1.4), loc='upper right', ncol=3, fontsize=font_size-4)
    plt.savefig(os.path.join(plot_dir, plot_name), bbox_inches='tight')
    plt.show()

wtmpc_anglefeedback_run1 = {'data_fitness': 'ga_fitness_log_20171002_182720.csv', 'data_max_dist': 'ga_max_dist_log_20171002_182720.csv'}
wtmpc_anglefeedback_run2 = {'data_fitness': 'ga_fitness_log_20171004_173049.csv', 'data_max_dist': 'ga_max_dist_log_20171004_173049.csv'}
wtmpc_anglefeedback_run3 = {'data_fitness': 'ga_fitness_log_20171006_145448.csv', 'data_max_dist': 'ga_max_dist_log_20171006_145448.csv'}

gen, min_f, max_f, avg_f, std_f = np.loadtxt(os.path.join(plot_data_dir, wtmpc_anglefeedback_run3['data_fitness']), delimiter=',', unpack=True, skiprows=1)
gen, max_d = np.loadtxt(os.path.join(plot_data_dir, wtmpc_anglefeedback_run3['data_max_dist']), delimiter=',', unpack=True, skiprows=1)

gen_fitness_plot(gen, max_f, avg_f, max_d, plot_name='ga_plot_paper.eps', grid=True)
