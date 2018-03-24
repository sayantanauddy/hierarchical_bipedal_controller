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
log_dir = os.path.join(home_dir, 'computing/repositories/MScThesis_SayantanAuddy_2017_NICOOscillatorWalking/logs/reinforcement_learning_logs')

'rl_test_data_boxplot_distance.csv', 'rl_test_data_boxplot_deviation.csv', 'rl_test_data_boxplot_torso_orientation.csv'
'rl_test_data_boxplot_paper.eps', [(-1, 7), (-3, 5), (-1, 3)]

def save_boxplot(datafile_distance, datafile_deviation, datafile_orientation, plotfile, limits):
    font_size = 40
    plt.rcParams.update({'font.size': font_size})

    linewidth = 1
    # Create box plots using the data files saved above
    i_dist, setup_control_dist, setup1_dist, setup2_dist, setup3_dist, setup4_dist = np.loadtxt(os.path.join(plot_data_dir, datafile_distance), delimiter=',', unpack=True, skiprows=1)
    i_dev, setup_control_dev, setup1_dev, setup2_dev, setup3_dev, setup4_dev = np.loadtxt(os.path.join(plot_data_dir, datafile_deviation), delimiter=',', unpack=True, skiprows=1)
    i_or, setup_control_or, setup1_or, setup2_or, setup3_or, setup4_or = np.loadtxt(os.path.join(plot_data_dir, datafile_orientation), delimiter=',', unpack=True, skiprows=1)

    fig, ax = plt.subplots(3, 1, figsize=(24,20), sharey=False)
    #ax = fig.add_subplot(111)
    setup_list_dist = [setup_control_dist, setup1_dist, setup2_dist, setup3_dist, setup4_dist]
    setup_list_dev = [setup_control_dev, setup1_dev, setup2_dev, setup3_dev, setup4_dev]
    setup_list_or = [setup_control_or, setup1_or, setup2_or, setup3_or, setup4_or]

    setup_list_dist.reverse()
    setup_list_dev.reverse()
    setup_list_or.reverse()

    bp_dist = ax[0].boxplot(setup_list_dist,
                            showmeans=False, patch_artist=True, vert=False,
                            meanprops=dict(marker='D', markeredgecolor='black', markerfacecolor='yellow'),
                            flierprops=dict(marker='o', markerfacecolor='#888888', markersize=5, linestyle='none', markeredgecolor='black'))

    bp_dev = ax[1].boxplot(setup_list_dev,
                           showmeans=False, patch_artist=True, vert=False,
                           meanprops=dict(marker='D', markeredgecolor='black', markerfacecolor='yellow'),
                           flierprops=dict(marker='o', markerfacecolor='#888888', markersize=5, linestyle='none', markeredgecolor='black'))

    bp_or = ax[2].boxplot(setup_list_or,
                          showmeans=False, patch_artist=True, vert=False,
                          meanprops=dict(marker='D', markeredgecolor='black', markerfacecolor='yellow'),
                          flierprops=dict(marker='o', markerfacecolor='#888888', markersize=5, linestyle='none', markeredgecolor='black'))

    box_count = 0
    for boxes in zip(bp_dist['boxes'], bp_dev['boxes'], bp_or['boxes']):
        # change outline color
        for box in boxes: box.set(color='#000000', linewidth=linewidth)
        # change fill color
        if box_count == 4:
            for box in boxes: box.set_facecolor('#FA8C8C')
        else:
            for box in boxes: box.set_facecolor('#9D98FA')
        box_count += 1

    ## change color and linewidth of the whiskers
    for whiskers in zip(bp_dist['whiskers'], bp_dev['whiskers'], bp_or['whiskers']):
        for whisker in whiskers: whisker.set(color='#000000', linewidth=2.5*linewidth)

    ## change color and linewidth of the caps
    for caps in zip(bp_dist['caps'], bp_dev['caps'], bp_or['caps']):
        for cap in caps: cap.set(color='#000000', linewidth=linewidth)

    ## change color and linewidth of the medians
    for medians in zip(bp_dist['medians'], bp_dev['medians'], bp_or['medians']):
        for median in medians: median.set(color='#000000', linewidth=linewidth)

    #for mean in bp['means']:
        #mean.set(color='#000000', linewidth=linewidth)
        #mean.set_facecolor('#FF0000')

    ## change the style of fliers and their fill
    for fliers in zip(bp_dist['fliers'], bp_dev['fliers'], bp_or['fliers']):
        for flier in fliers: flier.set(marker='o', color='#E0636F', alpha=1.0, linewidth=linewidth)

    ax[0].set_xlim(limits[0])
    ax[0].set_xlabel('Distance after 40s (m)')
    ax[0].tick_params(axis='y', pad=15)
    ax[0].tick_params(axis='x', pad=15)

    ax[1].set_xlim(limits[1])
    ax[1].set_xlabel('Deviation after 40s (m)')
    ax[1].tick_params(axis='y', pad=15)
    ax[1].tick_params(axis='x', pad=15)

    ax[2].set_xlim(limits[2])
    ax[2].set_xlabel('Orientation after 40s (radians)')
    ax[2].tick_params(axis='y', pad=15)
    ax[2].tick_params(axis='x', pad=15)

    ylabel_list = ['S0', 'S1', 'S2', 'S3', 'S4']
    ylabel_list.reverse()
    for i in range(3):
        ax[i].set_yticklabels(ylabel_list, fontsize=font_size)
        # Color the Y labels according to the box colors
        colors = ['b', 'b', 'b', 'b', 'r']
        for ytick, color in zip(ax[i].get_yticklabels(), colors):
            ytick.set_color(color)

        ax[i].axvline(x=0.0, color='black', linestyle='-', linewidth=0.5)
        ax[i].grid(color='#222222', linestyle='dotted', linewidth=1)

    #plt.show()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, plotfile), bbox_inches='tight')

save_boxplot('rl_test_data_boxplot_distance.csv',
             'rl_test_data_boxplot_deviation.csv',
             'rl_test_data_boxplot_torso_orientation.csv',
             'rl_test_data_boxplot_paper.eps',
             [(-1, 7), (-3, 5), (-1, 3)])
