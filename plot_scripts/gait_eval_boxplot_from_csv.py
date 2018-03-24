import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import os

# Set the home directory
home_dir = os.path.expanduser('~')

# Set the directory for saving plots
plot_dir = os.path.join(home_dir, 'computing/repositories/MScThesis_SayantanAuddy_2017_NICOOscillatorWalking/plots')

# Directory for saving plot data files
plot_data_dir = os.path.join(home_dir, 'computing/repositories/MScThesis_SayantanAuddy_2017_NICOOscillatorWalking/plot_data')

# Set the directory where logs are stored
log_dir = os.path.join(home_dir, 'computing/repositories/MScThesis_SayantanAuddy_2017_NICOOscillatorWalking/logs/reinforcement_learning_logs')

global fignum
fignum=1

def create_boxplot_from_csv(datafile, ax_label, ax_limit, plot_title, figsize=(10,10), vert=False, axis_label_show=True):

    global fignum

    plot_file = os.path.join(plot_dir,datafile.replace('.csv', '.svg'))

    i, \
    open_loop_run_1, \
    open_loop_run_2, \
    open_loop_run_3,\
    angle_feedback_run_1,\
    angle_feedback_run_2,\
    angle_feedback_run_3,\
    phase_reset_run_1,\
    phase_reset_run_2, \
    phase_reset_run_3 = np.loadtxt(os.path.join(plot_data_dir, datafile), delimiter=',', unpack=True, skiprows=1)

    # Calculate the medians of each column
    col_name = ''
    if 'fitness' in datafile:
        col_name = '$f$'
    elif 'up_time' in datafile:
        col_name = '$t_{up}$ (s)'
    elif 'x_dist' in datafile:
        col_name = '$distance_x$ (m)'
    elif 'abs_y_dev' in datafile:
        col_name = '$dev^{abs}_{y}$ (m)'
    elif 'avg_footstep' in datafile:
        col_name = '$stride^{avg}$ (m)'
    elif 'var_alpha' in datafile:
        col_name = '$torso_\\alpha^{var}$ (rad$^2$)'
    elif 'var_beta' in datafile:
        col_name = '$torso_\\beta^{var}$ (rad$^2$)'
    elif 'var_gamma' in datafile:
        col_name = '$torso_\\gamma^{var}$ (rad$^2$)'

    print '\multicolumn{1}{|c||}{' + col_name + '} & ' + '{} & {} & {} & {} & {} & {} & {} & {} & {} \\\\ \\hline'.format(
        round(np.median(open_loop_run_1), 3),
        round(np.median(open_loop_run_2), 3),
        round(np.median(open_loop_run_3), 3),
        round(np.median(angle_feedback_run_1), 3),
        round(np.median(angle_feedback_run_2), 3),
        round(np.median(angle_feedback_run_3), 3),
        round(np.median(phase_reset_run_1), 3),
        round(np.median(phase_reset_run_2), 3),
        round(np.median(phase_reset_run_3), 3))

    fig=plt.figure(fignum, figsize=figsize)
    fignum += 1
    ax = fig.add_subplot(111)
    bp = ax.boxplot([open_loop_run_1,
                     open_loop_run_2,
                     open_loop_run_3,
                     angle_feedback_run_1,
                     angle_feedback_run_2,
                     angle_feedback_run_3,
                     phase_reset_run_1,
                     phase_reset_run_2,
                     phase_reset_run_3],
                    showmeans=False,
                    patch_artist=True,
                    meanprops=dict(marker='D', markeredgecolor='black', markersize=8, markerfacecolor='yellow'),
                    flierprops=dict(marker='o', markerfacecolor='#888888', markersize=5, linestyle='none', markeredgecolor='black'),
                    vert=vert,
                    )

    linewidth=1
    font_size = 20
    plt.rcParams.update({'font.size': font_size})

    box_count = 0
    for box in bp['boxes']:
        # change outline color
        box.set(color='#000000', linewidth=linewidth)
        # change fill color
        if box_count <= 2:
            box.set_facecolor('#FA8C8C')
        elif box_count > 2 and box_count <= 5:
            box.set_facecolor('#9D98FA')
        else:
            box.set_facecolor('#6BE092')
        box_count += 1

    ## change color and linewidth of the whiskers
    for whisker in bp['whiskers']:
        whisker.set(color='#000000', linewidth=linewidth)

    ## change color and linewidth of the caps
    for cap in bp['caps']:
        cap.set(color='#000000', linewidth=linewidth*2.0)

    ## change color and linewidth of the medians
    for median in bp['medians']:
        median.set(color='#000000', linewidth=linewidth)

    for mean in bp['means']:
        mean.set(color='#000000', linewidth=linewidth)
        # mean.set_facecolor('#FF0000')

    ## change the style of fliers and their fill
    #for flier in bp['fliers']:
    #    flier.set(marker='o', color='#ABCDEF', alpha=0.5, linewidth=linewidth)

    #ax.set_ylim(limits)
    if not vert:
        ax.set_xlabel(ax_label, fontsize=20)
        if axis_label_show:
            ax.set_ylabel('Chromosomes of setups')
        ax.xaxis.labelpad = 20
        ax.yaxis.labelpad = 20

        ax.set_yticklabels(['OL'+r'$_1$', 'OL'+r'$_2$', 'OL'+r'$_3$',
                            'AF'+r'$_1$', 'AF'+r'$_2$', 'AF'+r'$_3$',
                            'PR'+r'$_1$', 'PR'+r'$_2$', 'PR'+r'$_3$',
                            ])

        # Hide axis labels
        if not axis_label_show:
            #ax.set_yticklabels([])
            #ax.yaxis.set_label_position('right')
            ax.yaxis.set_tick_params(labelright='on', labelleft='off')

        # Color the Y labels according to the box colors
        colors = ['r', 'r', 'r', 'b', 'b', 'b', 'g', 'g', 'g']
        for ytick, color in zip(ax.get_yticklabels(), colors):
            ytick.set_color(color)

        #ax.set_title(plot_title)
        #ax.axhline(y=0.0, color='black', linestyle='-', linewidth=0.5)
        ax.set_xlim(ax_limit)

        # If this is a smaller plot, align the axis labels to the left to allow the side-by-side plots to be closer
        plt.sca(ax)
        if figsize==(8,6):
            plt.xticks(ha='left')
        else:
            plt.xticks(ha='center')

    else:
        ax.set_ylabel(ax_label, fontsize=20)
        if axis_label_show:
            ax.set_ylabel('Chromosomes of setups')
        ax.xaxis.labelpad = 20
        ax.yaxis.labelpad = 20
        ax.set_xticklabels(['OL' + r'$_1$', 'OL' + r'$_2$', 'OL' + r'$_3$',
                            'AF' + r'$_1$', 'AF' + r'$_2$', 'AF' + r'$_3$',
                            'PR' + r'$_1$', 'PR' + r'$_2$', 'PR' + r'$_3$',
                            ])
        # If this is a smaller plot, align the axis labels to the left to allow the side-by-side plots to be closer
        y_labels = []
        for ytick in ax.get_yticklabels():
            y_labels.append(ytick)
        ax.set_xticklabels(y_labels, ha='left')

        # Color the X labels according to the box colors
        colors = ['r', 'r', 'r', 'b', 'b', 'b', 'g', 'g', 'g']
        for xtick, color in zip(ax.get_xticklabels(), colors):
            xtick.set_color(color)

        # Hide axis labels
        if not axis_label_show:
            ax.set_xticklabels([])

        # ax.set_title(plot_title)
        # ax.axhline(y=0.0, color='black', linestyle='-', linewidth=0.5)
        ax.set_ylim(ax_limit)

        # If this is a smaller plot, align the axis labels to the left to allow the side-by-side plots to be closer
        plt.sca(ax)
        if figsize == (8, 6):
            plt.yticks(ha='right')
        else:
            plt.yticks(ha='center')

    ax.grid(color='#222222', linestyle='dotted', linewidth=1)

    plt.savefig(os.path.join(plot_dir, plot_file), bbox_inches='tight')  # Change the font size used in the plots


# The latex table with the median values is printed
# header of the table
print r'\begin{table}[hbtp]'
print r'\centering'
print r'\caption[Median values of the gait evaluation parameters]{Median values of the gait evaluation parameters over 100 trials.}'
print r'\label{tab:gait_eval_median}'
print r'\resizebox{\textwidth}{!}{'
print r'\begin{tabular}{|c||c|c|c||c|c|c||c|c|c|}'
print r'\cline{1-10}'
print r'\multirow{2}{*}{\begin{tabular}[|c]{@{}l@{}}Parameter \end{tabular}}         & \multicolumn{3}{c||}{Open loop} & \multicolumn{3}{c||}{Angle feedback} & \multicolumn{3}{c|}{Phase reset} \\ \cline{2-10}'
print r'\multicolumn{1}{|c||}{     }          & $OL_1$   & $OL_2$   & $OL_3$   & $AF_1$     & $AF_2$     & $AF_3$    & $PR_1$    & $PR_2$    & $PR_3$   \\ \hline \hline'

# Create boxplots for asus data files
create_boxplot_from_csv(datafile='asus_gait_eval_fitness.csv',ax_label='Fitness score ', ax_limit=[0,15], plot_title='Fitness', figsize=(16,6), vert=False, axis_label_show=True)
create_boxplot_from_csv(datafile='asus_gait_eval_up_time.csv',ax_label='Up time (s)', ax_limit=[0,21], plot_title='Up time', figsize=(16,6), vert=False, axis_label_show=True)
create_boxplot_from_csv(datafile='asus_gait_eval_x_dist.csv',ax_label='Forward distance (m)', ax_limit=[-1.0,4.4], plot_title='X-distance', figsize=(8,6), vert=False, axis_label_show=True)
create_boxplot_from_csv(datafile='asus_gait_eval_abs_y_dev.csv',ax_label='Absolute Y-deviation (m)', ax_limit=[0,4.4], plot_title='Absolute Y-deviation', figsize=(8,6), vert=False, axis_label_show=False)
create_boxplot_from_csv(datafile='asus_gait_eval_avg_footstep.csv',ax_label='Average stride length (m)', ax_limit=[0,0.3], plot_title='Average footstep size', figsize=(16,6), vert=False, axis_label_show=True)
create_boxplot_from_csv(datafile='asus_gait_eval_var_alpha.csv',ax_label='Variance of ' + r'$torso_\alpha$' + ' (rad' + r'$^2$)', ax_limit=[0,0.0799], plot_title='Variance in alpha', figsize=(8,6), vert=False, axis_label_show=True)
create_boxplot_from_csv(datafile='asus_gait_eval_var_beta.csv',ax_label='Variance of ' + r'$torso_\beta$' + ' (rad' + r'$^2$)', ax_limit=[0,0.0799], plot_title='Variance in beta', figsize=(8,6), vert=False, axis_label_show=False)
create_boxplot_from_csv(datafile='asus_gait_eval_var_gamma.csv',ax_label='Variance of ' + r'$torso_\gamma$' + ' (rad' + r'$^2$)', ax_limit=[0,0.5], plot_title='Variance in gamma', figsize=(16,6), vert=False, axis_label_show=True)

# footer of the table
print r'\end{tabular}}'
print r'\end{table}'

print '##############################################'

# The latex table with the median values is printed
# header of the table
print '\\begin\{table\}[hbtp]'
print '\\centering'
print '\\caption[Median values of the gait evaluation parameters]{Median values of the gait evaluation parameters over 100 trials.}'
print '\\label{tab:gait_eval_median}'
print '\\resizebox{\\textwidth}{!}{'
print '\\begin{tabular}{|c||c|c|c||c|c|c||c|c|c|}'
print '\\cline{1-10}'
print '\\multirow{2}{*}{\\begin{tabular}[|c]{@{}l@{}}Parameter \\end{tabular}}         & \\\\multicolumn{3}{c\|\|}{Open loop} & \\multicolumn{3}{c\|\|}{Angle feedback} & \\multicolumn{3}{c\|}{Phase reset} \\\\ \\cline{2-10}'
print '\\multicolumn{1}{|c||}{     }          & $OL_1$   & $OL_2$   & $OL_3$   & $AF_1$     & $AF_2$     & $AF_3$    & $PR_1$    & $PR_2$    & $PR_3$   \\\\ \\hline \\hline'

# Create boxplots for wtmpc data files
create_boxplot_from_csv(datafile='wtmpc_gait_eval_fitness.csv',ax_label='Fitness score ', ax_limit=[0,15], plot_title='Fitness', figsize=(16,6), vert=False, axis_label_show=True)
create_boxplot_from_csv(datafile='wtmpc_gait_eval_up_time.csv',ax_label='Up time (s)', ax_limit=[0,21], plot_title='Up time', figsize=(16,6), vert=False, axis_label_show=True)
create_boxplot_from_csv(datafile='wtmpc_gait_eval_x_dist.csv',ax_label='Forward distance (m)', ax_limit=[-1.0,4.4], plot_title='X-distance', figsize=(8,6), vert=False, axis_label_show=True)
create_boxplot_from_csv(datafile='wtmpc_gait_eval_abs_y_dev.csv',ax_label='Absolute Y-deviation (m)', ax_limit=[0,4.4], plot_title='Absolute Y-deviation', figsize=(8,6), vert=False, axis_label_show=False)
create_boxplot_from_csv(datafile='wtmpc_gait_eval_avg_footstep.csv',ax_label='Average stride length (m)', ax_limit=[0,0.3], plot_title='Average footstep size', figsize=(16,6), vert=False, axis_label_show=True)
create_boxplot_from_csv(datafile='wtmpc_gait_eval_var_alpha.csv',ax_label='Variance of ' + r'$torso_\alpha$' + ' (rad' + r'$^2$)', ax_limit=[0,0.0799], plot_title='Variance in alpha', figsize=(8,6), vert=False, axis_label_show=True)
create_boxplot_from_csv(datafile='wtmpc_gait_eval_var_beta.csv',ax_label='Variance of ' + r'$torso_\beta$' + ' (rad' + r'$^2$)', ax_limit=[0,0.0799], plot_title='Variance in beta', figsize=(8,6), vert=False, axis_label_show=False)
create_boxplot_from_csv(datafile='wtmpc_gait_eval_var_gamma.csv',ax_label='Variance of ' + r'$torso_\gamma$' + ' (rad' + r'$^2$)', ax_limit=[0,0.5], plot_title='Variance in gamma', figsize=(16,6), vert=False, axis_label_show=True)

# footer of the table
print '\\end{tabular}}'
print '\\end{table}'